#!/usr/bin/env python3
"""OCR 端到端测试脚本：支持 CPU (ONNX) 和 RKNN (NPU) 双路径。

用法:
    # CPU (ONNX) 模式
    MACHINE_LEARNING_RKNN=false uv run --python 3.12 scripts/test_ocr.py --image test.png

    # RKNN (NPU) 模式（需先转换 .rknn 模型）
    LD_LIBRARY_PATH=~/.local/share/lib \
        MACHINE_LEARNING_RKNN=true uv run --python 3.12 --extra rknn scripts/test_ocr.py

    # 不传 --image 则生成一张包含文字的测试图片
    LD_LIBRARY_PATH=~/.local/share/lib \
        MACHINE_LEARNING_RKNN=true uv run --python 3.12 --extra rknn scripts/test_ocr.py

前置条件:
    - CPU 模式: 自动下载 PPOCRv5 ONNX 模型
    - RKNN 模式: 需先运行转换脚本生成 .rknn 文件:
        LD_LIBRARY_PATH=~/.local/share/lib \
            uv run --with 'rknn-toolkit2>=2.3.0,<3' --with 'numpy<2' --python 3.12 \
            scripts/convert_ocr_rknn.py --output-dir /tmp/ocr-rknn
        # 然后复制到模型缓存目录:
        cp /tmp/ocr-rknn/ppocrv4_det.rknn ~/.cache/immich_ml/ocr/PP-OCRv5_mobile/detection/model.rknn
        cp /tmp/ocr-rknn/ppocrv4_rec.rknn ~/.cache/immich_ml/ocr/PP-OCRv5_mobile/recognition/model.rknn
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from immich_ml.models.ocr.detection import TextDetector
from immich_ml.models.ocr.recognition import TextRecognizer
from immich_ml.sessions.rknn import is_available as rknn_available


def make_test_image() -> Image.Image:
    """生成包含中英文文字的测试图片。"""
    img = Image.new("RGB", (640, 200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((20, 30), "Hello OCR 2024", fill="black", font=font)
    draw.text((20, 100), "Immich RKNN Test", fill="black", font=font)
    return img


def run_ocr(image: Image.Image, cache_dir: str | None) -> None:
    """运行 OCR 检测 + 识别并打印结果。"""
    fmt = "RKNN (NPU)" if rknn_available else "CPU (ONNX)"
    print(f"\n{'='*60}")
    print(f"  OCR Mode: {fmt}")
    print(f"  RKNN available: {rknn_available}")
    print(f"{'='*60}\n")

    model_name = "PP-OCRv5_mobile"

    print("[1/2] Loading TextDetector...")
    t0 = time.perf_counter()
    detector = TextDetector(model_name, min_score=0.3, cache_dir=cache_dir)
    detector.load()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s (format={detector.model_format})")

    print("[2/2] Loading TextRecognizer...")
    t0 = time.perf_counter()
    recognizer = TextRecognizer(model_name, min_score=0.3, cache_dir=cache_dir)
    recognizer.load()
    print(f"  loaded in {time.perf_counter() - t0:.2f}s (format={recognizer.model_format})")

    print("\nRunning detection...")
    t0 = time.perf_counter()
    det_result = detector.predict(image)
    print(f"  detection: {time.perf_counter() - t0:.3f}s, {len(det_result['boxes'])} boxes")

    if len(det_result["boxes"]) == 0:
        print("  No text detected!")
        return

    print("\nRunning recognition...")
    t0 = time.perf_counter()
    rec_result = recognizer.predict(image, det_result)
    print(f"  recognition: {time.perf_counter() - t0:.3f}s")

    print(f"\n{'='*60}")
    print(f"  Results ({len(rec_result['text'])} texts)")
    print(f"{'='*60}")
    for i, (text, score) in enumerate(zip(rec_result["text"], rec_result["textScore"])):
        print(f"  [{i}] \"{text}\" (score: {score:.3f})")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Test OCR with CPU/RKNN backend")
    parser.add_argument("--image", type=Path, help="Path to test image (default: generate one)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Model cache directory (default: ~/.cache/immich_ml/ocr/<model_name>)")
    args = parser.parse_args()

    if args.image:
        print(f"Loading image: {args.image}")
        image = Image.open(args.image).convert("RGB")
    else:
        print("Generating test image...")
        image = make_test_image()
        image.save("/tmp/ocr_test.png")
        print(f"  saved to /tmp/ocr_test.png")

    run_ocr(image, args.cache_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
