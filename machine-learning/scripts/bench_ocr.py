#!/usr/bin/env python3
"""OCR RKNN 性能基准测试：多次运行统计 det/rec 耗时。

用法:
    LD_LIBRARY_PATH=~/.local/share/lib \
        MACHINE_LEARNING_RKNN=true uv run --python 3.12 --extra rknn \
        scripts/bench_ocr.py [--rounds 20]
"""

from __future__ import annotations

import argparse
import statistics
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


def bench(rounds: int, warmup: int) -> int:
    fmt = "RKNN (NPU)" if rknn_available else "CPU (ONNX)"
    print(f"\n{'='*70}")
    print(f"  OCR Benchmark | Mode: {fmt} | Rounds: {rounds} (warmup: {warmup})")
    print(f"{'='*70}\n")

    model_name = "PP-OCRv5_mobile"

    print("[1/2] Loading TextDetector...")
    t0 = time.perf_counter()
    detector = TextDetector(model_name, min_score=0.3)
    detector.load()
    print(f"  loaded in {time.perf_counter() - t0:.3f}s (format={detector.model_format})")

    print("[2/2] Loading TextRecognizer...")
    t0 = time.perf_counter()
    recognizer = TextRecognizer(model_name, min_score=0.3)
    recognizer.load()
    print(f"  loaded in {time.perf_counter() - t0:.3f}s (format={recognizer.model_format})")

    image = make_test_image()

    # Warmup
    print(f"\nWarming up ({warmup} runs)...")
    for _ in range(warmup):
        det_result = detector.predict(image)
        if len(det_result["boxes"]) > 0:
            recognizer.predict(image, det_result)

    # Benchmark
    print(f"Benchmarking ({rounds} runs)...\n")
    det_times: list[float] = []
    rec_times: list[float] = []
    total_times: list[float] = []
    texts: list[str] = []

    for i in range(rounds):
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        det_result = detector.predict(image)
        t_det = time.perf_counter() - t0
        det_times.append(t_det)

        t0 = time.perf_counter()
        if len(det_result["boxes"]) > 0:
            rec_result = recognizer.predict(image, det_result)
            t_rec = time.perf_counter() - t0
            rec_times.append(t_rec)
            if i == 0:
                texts = list(rec_result["text"])
        else:
            rec_times.append(0.0)

        total_times.append(time.perf_counter() - t_start)

    # Stats
    def stats(name: str, times: list[float]) -> None:
        n = len(times)
        avg = statistics.mean(times)
        med = statistics.median(times)
        mn = min(times)
        mx = max(times)
        std = statistics.stdev(times) if n > 1 else 0.0
        print(f"  {name:12s}: n={n}, avg={avg*1000:7.2f}ms, median={med*1000:7.2f}ms, "
              f"min={mn*1000:7.2f}ms, max={mx*1000:7.2f}ms, std={std*1000:6.2f}ms")

    print(f"{'='*70}")
    print(f"  Results")
    print(f"{'='*70}")
    stats("Detection", det_times)
    stats("Recognition", rec_times)
    stats("Total", total_times)

    print(f"\n  Throughput: {1.0 / statistics.median(total_times):.2f} FPS (median)")
    print(f"  Latency:    {statistics.median(total_times) * 1000:.2f} ms (median)")

    if texts:
        print(f"\n  Recognized texts:")
        for i, t in enumerate(texts):
            print(f"    [{i}] {t!r}")

    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="OCR RKNN Performance Benchmark")
    parser.add_argument("--rounds", type=int, default=20, help="Number of benchmark rounds (default: 20)")
    parser.add_argument("--warmup", type=int, default=3, help="Number of warmup runs (default: 3)")
    args = parser.parse_args()

    return bench(args.rounds, args.warmup)


if __name__ == "__main__":
    sys.exit(main())
