#!/usr/bin/env python3
"""PPOCR det/rec ONNX → RKNN 转换脚本。

由于 rknn-toolkit2（完整版）要求 numpy<=1.26.4，与 immich-ml 的 numpy>=2.4.0 冲突，
本脚本需在独立环境运行，不能在项目主 venv 中执行。

推荐运行方式（uv 隔离环境）：
    uv run --with "rknn-toolkit2>=2.3.0,<3" --with "numpy<2" --python 3.12 \\
        scripts/convert_ocr_rknn.py --output-dir /tmp/ocr-rknn

或在独立 venv 中：
    python -m venv /tmp/rknn-convert && /tmp/rknn-convert/bin/pip install rknn-toolkit2
    /tmp/rknn-convert/bin/python scripts/convert_ocr_rknn.py --output-dir /tmp/ocr-rknn

参考: https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# 模型下载地址（rknn_model_zoo 官方示例）
MODEL_URLS = {
    "det": "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/PPOCR/ppocrv4_det.onnx",
    "rec": "https://ftrg.zbox.filez.com/v2/delivery/data/95f00b0fc900458ba134f8b180b3f7a1/examples/PPOCR/ppocrv4_rec.onnx",
}

DEFAULT_TARGET = "rk3588"


def download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[skip] {dest} already exists")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def convert_onnx_to_rknn(onnx_path: Path, output_path: Path, target: str = DEFAULT_TARGET) -> Path:
    """使用 rknn-toolkit2（完整版）将 ONNX 转换为 RKNN。

    参考 rknn_model_zoo 官方实现：
    - det: 输入 480×480，ImageNet mean/std，INT8 量化（需要 dataset）
    - rec: 输入 48×320，mean=0/std=1，FP16（不量化）

    由于 INT8 量化需要 dataset 图片，默认用 FP16（不量化）以简化流程。
    如需 INT8 量化获得更好性能，请参考官方 rknn_model_zoo 仓库。
    """
    from rknn.api import RKNN

    rknn = RKNN()

    is_det = "det" in onnx_path.name

    if is_det:
        # det 模型：参考官方 convert.py
        # mean_values/std_values 在模型内部处理归一化，推理时直接传 0-255 像素值
        rknn.config(
            mean_values=[[123.675, 116.28, 103.53]],
            std_values=[[58.395, 57.12, 57.375]],
            target_platform=target,
        )
    else:
        # rec 模型：参考官方 convert.py，不设置 mean/std
        # 归一化（pixel/255）在推理时由外部完成
        rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[1, 1, 1]],
            target_platform=target,
        )

    # 指定输入 shape：det 为 (1,3,480,480)，rec 为 (1,3,48,320)
    # 官方 det 用 480×480（DET_INPUT_SHAPE = [480, 480]）
    # inputs 是输入名称列表，input_size_list 是对应的 size
    input_size_list = None
    if is_det:
        input_size_list = [[1, 3, 480, 480]]
    else:
        input_size_list = [[1, 3, 48, 320]]

    # PPOCR ONNX 模型输入名称为 "x"
    ret = rknn.load_onnx(
        model=onnx_path.as_posix(),
        inputs=["x"],
        input_size_list=input_size_list,
    )
    if ret != 0:
        raise RuntimeError(f"load_onnx failed: {ret}")

    # 不做量化（fp16），保留精度
    # 官方默认 INT8 量化，但需要 dataset，这里用 FP16 简化
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        raise RuntimeError(f"build failed: {ret}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(output_path.as_posix())
    if ret != 0:
        raise RuntimeError(f"export_rknn failed: {ret}")

    rknn.release()
    print(f"[convert] {onnx_path.name} -> {output_path} (target={target}, fp16)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert PPOCR ONNX models to RKNN format")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/ocr-rknn"), help="Output directory for .rknn files")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, choices=["rk3566", "rk3568", "rk3576", "rk3588"],
                        help="Target platform")
    parser.add_argument("--models", nargs="*", choices=["det", "rec"], default=["det", "rec"],
                        help="Which models to convert")
    args = parser.parse_args()

    for name in args.models:
        url = MODEL_URLS[name]
        onnx_path = args.output_dir / f"ppocrv4_{name}.onnx"
        rknn_path = args.output_dir / f"ppocrv4_{name}.rknn"

        download(url, onnx_path)
        convert_onnx_to_rknn(onnx_path, rknn_path, target=args.target)

    print("\n[done] RKNN models generated:")
    for name in args.models:
        rknn_path = args.output_dir / f"ppocrv4_{name}.rknn"
        print(f"  {name}: {rknn_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
