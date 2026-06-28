# Immich Machine Learning

- CLIP embeddings
- Facial recognition
- OCR (CPU + RKNN NPU)

## RK3588 OCR 适配（RKNN NPU + CPU 双路径）

### 环境初始化

```bash
# 系统依赖
sudo apt install build-essential python3-dev -y

# Python 环境（必须 3.12，rknn-toolkit-lite2 不支持 3.13）
uv sync --python 3.12 --extra rknn
```

### 1. 转换 ONNX → RKNN 模型

rknn-toolkit2（完整版）与项目 numpy>=2.4.0 冲突，需在隔离环境运行：

```bash
uv run --with 'rknn-toolkit2>=2.3.0,<3' --with 'numpy<2' --python 3.12 \
    scripts/convert_ocr_rknn.py --output-dir /tmp/ocr-rknn
```

转换后生成 `ppocrv4_det.rknn` 和 `ppocrv4_rec.rknn`。
将它们复制到模型缓存目录并重命名为 `model.rknn`（det/rec 分别存于不同子目录）：

```bash
CACHE_DIR=~/.cache/immich_ml/ocr/PP-OCRv5_mobile
mkdir -p "$CACHE_DIR/detection" "$CACHE_DIR/recognition"
cp /tmp/ocr-rknn/ppocrv4_det.rknn "$CACHE_DIR/detection/model.rknn"
cp /tmp/ocr-rknn/ppocrv4_rec.rknn "$CACHE_DIR/recognition/model.rknn"
# 字符表 (ppocr_keys_v1.txt) 会在首次加载识别模型时自动下载
```

### 2. 运行 OCR

#### CPU (ONNX) 模式

```bash
MACHINE_LEARNING_RKNN=false uv run --python 3.12 scripts/test_ocr.py --image test.png
```

#### RKNN (NPU) 模式

```bash
uv run --python 3.12 --extra rknn scripts/test_ocr.py --image test.png
```

不传 `--image` 则自动生成测试图片。

### 3. 启动服务

```bash
# CPU 模式
MACHINE_LEARNING_RKNN=false uv run --python 3.12 gunicorn immich_ml.main:app

# RKNN 模式
uv run --python 3.12 --extra rknn gunicorn immich_ml.main:app
```

### 架构说明

| 组件 | 文件 | 说明 |
|------|------|------|
| RknnOcrSession | `immich_ml/sessions/rknn/ocr.py` | OCR 专用 RKNN session（单实例，非线程池） |
| TextDetector | `immich_ml/models/ocr/detection.py` | 检测模型，按 model_format 走 ONNX 或 RKNN |
| TextRecognizer | `immich_ml/models/ocr/recognition.py` | 识别模型，RKNN 路径绕过 RapidTextRecognizer |
| 转换脚本 | `scripts/convert_ocr_rknn.py` | ONNX→RKNN 转换（独立环境运行） |
| 测试脚本 | `scripts/test_ocr.py` | 端到端 OCR 测试 |

模型格式由 `MACHINE_LEARNING_RKNN` 环境变量控制：
- `true`（默认）: RK3588 上自动使用 RKNN
- `false`: 强制使用 CPU/ONNX


# Setup

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/), so be sure to install it first.
Running `uv sync --extra cpu` will install everything you need in an isolated virtual environment.
CUDA, ROCM and OpenVINO are supported as acceleration APIs. To use them, you can replace `--extra cpu` with either of `--extra cuda`, `--extra rocm` or `--extra openvino`. In the case of CUDA, a [compute capability](https://developer.nvidia.com/cuda-gpus) of 5.2 or higher is required.

To add or remove dependencies, you can use the commands `uv add $PACKAGE_NAME` and `uv remove $PACKAGE_NAME`, respectively.
Be sure to commit the `uv.lock` and `pyproject.toml` files with `uv lock` to reflect any changes in dependencies.

# Load Testing

To measure inference throughput and latency, you can use [Locust](https://locust.io/) using the provided `locustfile.py`.
Locust works by querying the model endpoints and aggregating their statistics, meaning the app must be deployed.
You can change the models or adjust options like score thresholds through the Locust UI.

To get started, you can simply run `locust --web-host 127.0.0.1` and open `localhost:8089` in a browser to access the UI. See the [Locust documentation](https://docs.locust.io/en/stable/index.html) for more info on running Locust.

Note that in Locust's jargon, concurrency is measured in `users`, and each user runs one task at a time. To achieve a particular per-endpoint concurrency, multiply that number by the number of endpoints to be queried. For example, if there are 3 endpoints and you want each of them to receive 8 requests at a time, you should set the number of users to 24.

# Facial Recognition

## Acknowledgements

This project utilizes facial recognition models from the [InsightFace](https://github.com/deepinsight/insightface/tree/master/model_zoo) project. We appreciate the work put into developing these models, which have been beneficial to the machine learning part of this project.

### Used Models

- antelopev2
- buffalo_l
- buffalo_m
- buffalo_s

## License and Use Restrictions

We have received permission to use the InsightFace facial recognition models in our project, as granted via email by Jia Guo (guojia@insightface.ai) on 18th March 2023. However, it's important to note that this permission does not extend to the redistribution or commercial use of their models by third parties. Users and developers interested in using these models should review the licensing terms provided in the InsightFace GitHub repository.

For more information on the capabilities of the InsightFace models and to ensure compliance with their license, please refer to their [official repository](https://github.com/deepinsight/insightface). Adhering to the specified licensing terms is crucial for the respectful and lawful use of their work.
