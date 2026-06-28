from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rapidocr.ch_ppocr_det.utils import DBPostProcess
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
from rapidocr.utils.typings import EngineType, LangDet, OCRVersion, TaskType
from rapidocr.utils.typings import ModelType as RapidModelType

from immich_ml.config import log
from immich_ml.models.base import InferenceModel
from immich_ml.schemas import ModelFormat, ModelSession, ModelTask, ModelType
from immich_ml.sessions.ort import OrtSession

from .schemas import TextDetectionOutput


class TextDetector(InferenceModel):
    depends = []
    identity = (ModelType.DETECTION, ModelTask.OCR)

    def __init__(self, model_name: str, min_score: float = 0.5, **model_kwargs: Any) -> None:
        super().__init__(model_name.split("__")[-1], **model_kwargs)
        self.max_resolution = 736
        self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.std_inv = np.float32(1.0) / (np.array([0.5, 0.5, 0.5], dtype=np.float32) * 255.0)
        self._empty: TextDetectionOutput = {
            "boxes": np.empty(0, dtype=np.float32),
            "scores": np.empty(0, dtype=np.float32),
        }
        # 参考 rknn_model_zoo PPOCR-Det 官方配置：
        # thresh=0.3, box_thresh=0.6, unclip_ratio=1.5, use_dilation=False, score_mode='fast'
        self.postprocess = DBPostProcess(
            thresh=0.3,
            box_thresh=0.6,
            max_candidates=1000,
            unclip_ratio=1.5,
            use_dilation=False,
            score_mode="fast",
        )

    def model_path_for_format(self, model_format: ModelFormat) -> Path:
        # OCR 的 .rknn 由本地转换脚本生成，不使用 rknpu/{soc} 前缀
        if model_format == ModelFormat.RKNN:
            return self.model_dir / f"model.{model_format}"
        return super().model_path_for_format(model_format)

    def _download(self) -> None:
        if self.model_format == ModelFormat.RKNN:
            # 从 fork 仓库的 GitHub raw 地址自动下载预转换的 .rknn 模型
            # 仓库：https://github.com/Ra1nyHouse/immich (feature/rknn-orc 分支)
            # 模型由 scripts/convert_ocr_rknn.py 转换，FP16，det 输入 480x480
            rknn_url = (
                "https://raw.githubusercontent.com/Ra1nyHouse/immich/feature/rknn-orc/"
                f"machine-learning/models/rknn/ocr/{self.model_name}/detection/model.rknn"
            )
            log.info(f"Downloading OCR RKNN detection model from {rknn_url}")
            DownloadFile.run(
                DownloadFileInput(
                    file_url=rknn_url,
                    sha256=None,  # GitHub raw 不提供 sha256，跳过校验
                    save_path=self.model_path,
                    logger=log,
                )
            )
            return
        model_info = InferSession.get_model_url(
            FileInfo(
                engine_type=EngineType.ONNXRUNTIME,
                ocr_version=OCRVersion.PPOCRV5,
                task_type=TaskType.DET,
                lang_type=LangDet.CH,
                model_type=RapidModelType.MOBILE if "mobile" in self.model_name else RapidModelType.SERVER,
            )
        )
        download_params = DownloadFileInput(
            file_url=model_info["model_dir"],
            sha256=model_info["SHA256"],
            save_path=self.model_path,
            logger=log,
        )
        DownloadFile.run(download_params)

    def _load(self) -> ModelSession:
        if self.model_format == ModelFormat.RKNN:
            from immich_ml.sessions.rknn.ocr import RknnOcrSession
            # RKNN det 模型为 static shape 480×480（参考 rknn_model_zoo DET_INPUT_SHAPE = [480, 480]）
            return RknnOcrSession(self.model_path, input_shape=(1, 3, 480, 480))
        return OrtSession(self.model_path)

    # partly adapted from RapidOCR and rknn_model_zoo PPOCR-Det
    def _predict(self, inputs: Image.Image) -> TextDetectionOutput:
        w, h = inputs.size
        if w < 32 or h < 32:
            return self._empty
        transformed = self._transform(inputs)
        out = self.session.run(None, {"x": transformed})[0]

        # RKNN 路径：使用保持宽高比 + pad 策略（参考 insightface RetinaFace）
        # 模型输入 480×480，其中实际图像区域为 (resize_h, resize_w)，右下 pad 0
        # DBPostProcess 内部会做 box.x/width*dest_width 映射，其中 width/height 是
        # 模型输出 bitmap 尺寸（480×480），所以 ori_shape 必须传 (480, 480)
        # 让 box 坐标保持在模型输入坐标系，再单独映射到原图
        if self.model_format == ModelFormat.RKNN:
            src_w, src_h = inputs.size
            ratio = min(480 / src_w, 480 / src_h)
            resize_w = int(src_w * ratio)
            resize_h = int(src_h * ratio)
            # ori_shape 传 (480, 480)，box 坐标基于模型输入坐标系
            boxes_raw, scores = self.postprocess(out, (480, 480))
            if len(boxes_raw) > 0:
                # box 坐标基于 (480, 480) 模型输入坐标系：
                # - x 方向：实际图像填满 480 宽，x 坐标按 src_w/resize_w=src_w/(src_w*ratio)=1/ratio 还原
                # - y 方向：实际图像占 resize_h 高，y 坐标先确认 < resize_h（在有效区域），再按 src_h/resize_h=1/ratio 还原
                boxes = boxes_raw.astype(np.float32).copy()
                scale = 1.0 / ratio  # 等比缩放因子
                # 过滤掉 y >= resize_h 的 box（pad 区域的误检）
                valid_mask = boxes[:, :, 1].max(axis=1) < resize_h
                boxes = boxes[valid_mask]
                scores = [s for s, v in zip(scores, valid_mask) if v]
                if len(boxes) == 0:
                    return self._empty
                # 等比缩放回原图坐标
                boxes[:, :, 0] *= scale
                boxes[:, :, 1] *= scale
            else:
                boxes = boxes_raw
        else:
            boxes, scores = self.postprocess(out, (h, w))
        if len(boxes) == 0:
            return self._empty
        return {
            "boxes": self.sorted_boxes(boxes),
            "scores": np.array(scores, dtype=np.float32),
        }

    # adapted from RapidOCR and rknn_model_zoo PPOCR-Det
    def _transform(self, img: Image.Image) -> NDArray[np.float32]:
        if self.model_format == ModelFormat.RKNN:
            # 参考 insightface RetinaFace detect() 的"保持宽高比 + pad"策略，
            # 避免 OCR 文字拉伸变形导致检测精度下降：
            # 1. 按 min(480/src_w, 480/src_h) 等比缩放，短边填满 480
            # 2. 缩放后图像贴到 480×480 画布左上角，右下 pad 0（黑色）
            # 3. 后处理用实际图像区域尺寸反算坐标，pad 区域不产生检测
            src_w, src_h = img.size
            ratio = min(480 / src_w, 480 / src_h)
            new_w = int(src_w * ratio)
            new_h = int(src_h * ratio)
            resized_img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
            # 创建 480×480 黑色画布，把缩放后的图像贴到左上角
            canvas = Image.new("RGB", (480, 480), (0, 0, 0))
            canvas.paste(resized_img, (0, 0))
            # 转为 BGR float32，不归一化（保持 0-255 范围，归一化在模型内部完成）
            img_np: NDArray[np.float32] = cv2.cvtColor(
                np.array(canvas, dtype=np.float32), cv2.COLOR_RGB2BGR
            )  # type: ignore
            img_np = np.transpose(img_np, (2, 0, 1))
            return np.expand_dims(img_np, axis=0)
        else:
            if img.height < img.width:
                ratio = float(self.max_resolution) / img.height
            else:
                ratio = float(self.max_resolution) / img.width
            ratio = min(ratio, 1.0)

            resize_h = int(img.height * ratio)
            resize_w = int(img.width * ratio)

            resize_h = int(round(resize_h / 32) * 32)
            resize_w = int(round(resize_w / 32) * 32)
            resized_img = img.resize((int(resize_w), int(resize_h)), resample=Image.Resampling.LANCZOS)

            img_np = cv2.cvtColor(np.array(resized_img, dtype=np.float32), cv2.COLOR_RGB2BGR)  # type: ignore
            # PPOCR det 标准预处理：归一化到 [-1,1]（mean=0.5, std=0.5）
            img_np -= self.mean
            img_np *= self.std_inv
            img_np = np.transpose(img_np, (2, 0, 1))
            return np.expand_dims(img_np, axis=0)

    def sorted_boxes(self, dt_boxes: NDArray[np.float32]) -> NDArray[np.float32]:
        if len(dt_boxes) == 0:
            return dt_boxes

        # Sort by y, then identify lines, then sort by (line, x)
        y_order = np.argsort(dt_boxes[:, 0, 1], kind="stable")
        sorted_y = dt_boxes[y_order, 0, 1]

        line_ids = np.empty(len(dt_boxes), dtype=np.int32)
        line_ids[0] = 0
        np.cumsum(np.abs(np.diff(sorted_y)) >= 10, out=line_ids[1:])

        # Create composite sort key for final ordering
        # Shift line_ids by large factor, add x for tie-breaking
        sort_key = line_ids[y_order] * 1e6 + dt_boxes[y_order, 0, 0]
        final_order = np.argsort(sort_key, kind="stable")
        sorted_boxes: NDArray[np.float32] = dt_boxes[y_order[final_order]]
        return sorted_boxes

    def configure(self, **kwargs: Any) -> None:
        if (max_resolution := kwargs.get("maxResolution")) is not None:
            self.max_resolution = max_resolution
        if (min_score := kwargs.get("minScore")) is not None:
            self.postprocess.box_thresh = min_score
        if (score_mode := kwargs.get("scoreMode")) is not None:
            self.postprocess.score_mode = score_mode
