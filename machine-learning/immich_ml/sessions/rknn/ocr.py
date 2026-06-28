from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

from immich_ml.config import log

from .rknnpool import is_available, soc_name


def _init_rknn_lite(model_path: str) -> Any:
    """加载 .rknn 模型并初始化 NPU 运行时（单实例，非线程池）。

    RKNNLite.init_runtime 默认 core_mask=NPU_CORE_AUTO，由 NPU 驱动在
    RK3588 三核（NPU_CORE_0/1/2）间自动调度。OCR 单图同步推理场景下，
    AUTO 与固定单核性能相当（实测 median 147ms vs 151ms），保持默认即可。

    多核真正受益场景是多请求并行（如人脸/CLIP 的 RknnPoolExecutor
    加载多份模型副本），OCR 为单实例同步处理，无需显式设置 core_mask。
    """
    if not is_available:
        raise RuntimeError("RKNN is not available on this device")
    from rknnlite.api import RKNNLite

    rknn_lite = RKNNLite()
    rknn_lite.rknn_log.logger.setLevel(logging.ERROR)
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model: {model_path}")
    # 默认 core_mask=NPU_CORE_AUTO，由 NPU 驱动在 RK3588 三核间自动调度
    # 实测 AUTO 与固定单核 NPU_CORE_0 在 OCR 单图同步推理场景下性能相当
    # （AUTO median 147ms vs NPU_CORE_0 median 151ms），保持默认即可
    ret = rknn_lite.init_runtime()
    if ret != 0:
        raise RuntimeError("Failed to initialize RKNN runtime environment")
    return rknn_lite


class RknnOcrSession:
    """OCR 专用的 RKNN session。

    与 OrtSession 接口兼容（run/get_inputs/get_outputs），
    同时提供 __call__ 以便与 rapidocr 的 InferSession 协议对接。

    与 RknnSession 的区别：
    - 不使用多线程池（OCR 为同步单图/单批处理）
    - 不依赖 input_output_mapping（IO 由 .rknn 模型自身决定）
    - 适用于 PPOCR det/rec 模型
    """

    def __init__(self, model_path: Path, input_shape: tuple[int, ...] | None = None) -> None:
        self.model_path = Path(model_path)
        self._rknn_lite: Any = None
        # .rknn 为 static shape，RKNNLite 无 API 查询，由调用方传入
        # det: (1, 3, 480, 480)，rec: (1, 3, 48, 320)
        self._input_shape = input_shape
        # 首次成功推理标志，用于输出"首次推理成功"日志
        self._first_inference_done = False
        log.info(f"Loading OCR RKNN model from {model_path}, input_shape={input_shape}")
        self._rknn_lite = _init_rknn_lite(self.model_path.as_posix())
        log.info(f"Loaded OCR RKNN model from {model_path}")

    def _resize_input(self, arr: NDArray[np.float32]) -> NDArray[np.float32]:
        """若指定了 input_shape 且与传入数据不一致，则 resize 到模型期望的 H/W。

        - 仅处理 NCHW 且 H/W 为正数的静态 shape
        - 其它情况保持原样
        """
        shape = self._input_shape
        if shape is None or len(shape) != 4:
            return arr
        n, c, h, w = shape
        if n < 0 or c < 0 or h <= 0 or w <= 0:
            return arr
        if arr.ndim != 4:
            return arr
        a_n, a_c, a_h, a_w = arr.shape
        if a_n == n and a_c == c and a_h == h and a_w == w:
            return arr
        if a_c != c:
            log.warning(f"RKNN input channel mismatch: model={c}, input={a_c}, skip resize")
            return arr
        # 双线性 resize H/W 到模型期望尺寸
        try:
            import cv2

            resized = cv2.resize(
                arr[0].transpose(1, 2, 0).astype(np.float32),
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            out = resized.transpose(2, 0, 1)[np.newaxis, ...]
            return np.ascontiguousarray(out, dtype=arr.dtype)
        except Exception as e:
            log.warning(f"RKNN input resize failed: {e}")
            return arr

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        # RKNN 模型为 static shape (1, C, H, W)，不支持 batch>1
        # 若传入 batch>1，则逐个推理后拼接结果
        # 注意：_resize_input 会截断 batch>1 的输入，所以先检查原始输入
        raw_inputs = list(input_feed.values())
        first_raw = raw_inputs[0]
        if first_raw.ndim == 4 and first_raw.shape[0] > 1:
            # batch>1：逐个推理（每个样本单独 resize 后推理）
            batch_outputs = []
            for i in range(first_raw.shape[0]):
                single_input = np.ascontiguousarray(self._resize_input(first_raw[i:i+1]))
                outputs = self._rknn_lite.inference(inputs=[single_input], data_format="nchw")
                if outputs is None:
                    raise RuntimeError("RKNN OCR inference returned None")
                batch_outputs.append(np.asarray(outputs[0]))
            # 沿 batch 维度拼接
            results = [np.concatenate(batch_outputs, axis=0)]
        else:
            # batch=1：直接推理
            inputs_list = [np.ascontiguousarray(self._resize_input(v)) for v in raw_inputs]
            outputs = self._rknn_lite.inference(inputs=inputs_list, data_format="nchw")
            if outputs is None:
                raise RuntimeError("RKNN OCR inference returned None")
            results = [np.asarray(o) for o in outputs]

        # 首次成功推理日志（按项目规范，参考 RknnSession 的加载日志风格）
        if not self._first_inference_done:
            self._first_inference_done = True
            input_shape = tuple(first_raw.shape) if first_raw.ndim >= 1 else None
            output_shapes = [tuple(r.shape) for r in results]
            log.info(
                f"First OCR RKNN inference ok: model={self.model_path.name}, "
                f"input_shape={input_shape}, output_shapes={output_shapes}"
            )
        return results

    def __call__(self, input_content: NDArray[np.float32]) -> NDArray[np.float32]:
        """rapidocr InferSession 兼容接口：单输入 → 单输出。"""
        return self.run(None, {"x": input_content})[0]

    def get_inputs(self) -> list["RknnOcrNode"]:
        shape = self._input_shape if self._input_shape else (-1, -1, -1, -1)
        return [RknnOcrNode(name="x", shape=shape)]

    def get_outputs(self) -> list["RknnOcrNode"]:
        return [RknnOcrNode(name="output", shape=(-1, -1))]

    def get_input_names(self) -> list[str]:
        return ["x"]

    def get_output_names(self) -> list[str]:
        return ["output"]

    def release(self) -> None:
        if self._rknn_lite is not None:
            log.info(f"Releasing OCR RKNN model from {self.model_path}")
            self._rknn_lite.release()
            self._rknn_lite = None
            log.info(f"Released OCR RKNN model from {self.model_path}")

    def __del__(self) -> None:
        self.release()


class RknnOcrNode(NamedTuple):
    name: str | None
    shape: tuple[int, ...]


__all__ = ["RknnOcrSession", "RknnOcrNode", "is_available", "soc_name"]
