# OCR RKNN 适配说明

本文档说明 immich-ml 在 RK3588 平台上将 PPOCR 模型从 CPU (ONNX Runtime) 切换到 RKNN (NPU) 的适配方案。

参考实现：[rknn_model_zoo PPOCR](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR)

---

## 1. 总体架构

```
┌─────────────────────────────────────────────────────────┐
│                   TextDetector                          │
│  Image ──► _transform ──► session.run ──► DBPostProcess │
│                  │                                       │
│         ┌────────┴────────┐                             │
│         ▼                 ▼                             │
│    ONNX (CPU)        RKNN (NPU)                         │
│    OrtSession        RknnOcrSession                     │
│    mean=0.5,std=0.5  mean=123.675,std=58.395 (ImageNet) │
│    736×动态尺寸       480×480 静态                       │
│    保持宽高比          保持宽高比 + pad（参考 RetinaFace）│
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   TextRecognizer                        │
│  Crop ──► _resize_norm ──► session.run ──► CTCDecode   │
│                  │                                       │
│         ┌────────┴────────┐                             │
│         ▼                 ▼                             │
│    ONNX (CPU)        RKNN (NPU)                         │
│    RapidTextRec      RknnOcrSession                     │
│    [-1,1] 归一化     [-1,1] 归一化 (与 ONNX 完全一致)    │
│    (3,48,320)        (3,48,320) 静态                    │
│    batch 并行        batch 串行（逐样本推理后拼接）      │
└─────────────────────────────────────────────────────────┘
```

双路径同时保留：
- `MACHINE_LEARNING_RKNN=false`（默认）→ CPU (ONNX) 路径
- `MACHINE_LEARNING_RKNN=true` → RKNN (NPU) 路径

> **注意**：`MACHINE_LEARNING_RKNN` 是**全局开关**，同时影响人脸检测/识别、CLIP、OCR 所有模型。并非 OCR 专用开关。若只想单独控制 OCR 是否走 RKNN，需要修改 [immich_ml/models/base.py](immich_ml/models/base.py) 中的 `_model_format_default` 逻辑，新增 OCR 专用环境变量（如 `MACHINE_LEARNING_OCR_USE_RKNN`）。
>
> 在测试脚本 [scripts/test_ocr.py](scripts/test_ocr.py) 中，由于测试时只加载 OCR 模型，`MACHINE_LEARNING_RKNN=true` 的效果等同于"OCR 走 RKNN"，不会影响其他模型。

---

## 2. 图片转化规则

### 2.1 Detection（检测）模型预处理

| 步骤 | CPU (ONNX) 路径 | RKNN (NPU) 路径 |
|---|---|---|
| **resize** | 保持宽高比，长边缩放到 `max_resolution=736`，并对齐到 32 的倍数 | **保持宽高比 + pad**：按 `ratio = min(480/src_w, 480/src_h)` 等比缩放，贴到 480×480 黑色画布左上角，右下 pad 0 |
| **颜色转换** | RGB → BGR | RGB → BGR |
| **归一化** | `(pixel - 0.5) / (0.5 * 255)` → 范围 [-1, 1] | **不归一化**，保持 0-255 像素值（归一化由模型内部 `mean_values`/`std_values` 完成） |
| **数据格式** | NCHW `(1, 3, H, W)` | NCHW `(1, 3, 480, 480)` |
| **dtype** | float32 | float32 |

**保持宽高比 + pad 策略**（参考 [insightface RetinaFace](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/retinaface.py)）：

```python
src_w, src_h = img.size
ratio = min(480 / src_w, 480 / src_h)
new_w = int(src_w * ratio)
new_h = int(src_h * ratio)
resized_img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)
canvas = Image.new("RGB", (480, 480), (0, 0, 0))   # 黑色画布
canvas.paste(resized_img, (0, 0))                  # 贴到左上角
```

**为什么不用直接 resize 到 480×480？**

直接 resize 会导致文字宽高比失真（横向拉伸或纵向压缩），DBNet 检测精度显著下降。保持宽高比 + pad 的策略与人脸检测（RetinaFace）一致：
- 短边等比缩放到 480，长边按比例缩放
- 缩放后的图像贴到 480×480 画布左上角
- 右下区域 pad 0（黑色），后处理时过滤该区域的误检框

**关键差异**：RKNN 模型在转换时通过 `rknn.config(mean_values=[[123.675, 116.28, 103.53]], std_values=[[58.395, 57.12, 57.375]])` 将 ImageNet 标准归一化内置到模型中，因此推理时直接传 0-255 像素值即可。

### 2.2 Detection 后处理

| 参数 | CPU (ONNX) | RKNN (NPU) | 说明 |
|---|---|---|---|
| `thresh` | 0.3 | 0.3 | 二值化阈值 |
| `box_thresh` | 0.5（可配置） | **0.6** | 框阈值，官方默认值 |
| `unclip_ratio` | 2.5 | **1.5** | 框扩展比例，官方默认值 |
| `use_dilation` | True | **False** | 是否膨胀，官方默认值 |
| `score_mode` | fast | fast | 评分模式 |

**坐标映射**（保持宽高比 + pad 策略）：

```python
# 1. postprocess 传入 (480, 480) 作为 ori_shape，box 坐标基于模型输入坐标系
boxes_raw, scores = self.postprocess(out, (480, 480))

# 2. 过滤 pad 区域的误检框（y 坐标 >= resize_h 的框在 pad 区域）
valid_mask = boxes[:, :, 1].max(axis=1) < resize_h
boxes = boxes[valid_mask]
scores = [s for s, v in zip(scores, valid_mask) if v]

# 3. 等比缩放回原图坐标（x 和 y 用同一个 scale）
scale = 1.0 / ratio   # ratio = min(480/src_w, 480/src_h)
boxes[:, :, 0] *= scale
boxes[:, :, 1] *= scale
```

**关键点**：
- `ori_shape` 必须传 `(480, 480)`，因为 DBPostProcess 内部会做 `box.x / width * dest_width` 映射，width/height 是模型输出 bitmap 尺寸（480×480）
- pad 区域（`y >= resize_h`）会产生误检框，必须过滤
- x 和 y 用**同一个 scale** 等比缩放（因为预处理是保持宽高比缩放）

### 2.3 Recognition（识别）模型预处理

| 步骤 | CPU (ONNX) 路径 | RKNN (NPU) 路径 |
|---|---|---|
| **crop** | `get_crop_img_list` 透视变换裁剪 | 同左 |
| **resize** | 保持宽高比，高度 resize 到 `imgH=48`，宽度按比例计算（上限 `imgW=320`） | 同左 |
| **归一化** | `(pixel/255 - 0.5) / 0.5` → 范围 [-1, 1] | **`(pixel/255 - 0.5) / 0.5`** → 范围 [-1, 1]（与 ONNX 完全一致） |
| **padding** | 宽度不足 `imgW` 时右侧 pad 0 | 同左 |
| **数据格式** | NCHW `(N, 3, 48, 320)` | NCHW `(N, 3, 48, 320)` |
| **dtype** | float32 | float32 |

**关键差异**：RKNN 路径归一化与 ONNX 路径**完全一致**，都是 `[-1, 1]` 归一化。早期版本 RKNN 路径曾用 `pixel/255`（[0,1] 归一化），但会导致识别错误（如 "Hello OCR 2024" 被识别为 "Hello OC R 2024"），改为 [-1,1] 后修复。

> **注意**：RKNN 模型转换时 `mean_values=[[0,0,0]], std_values=[[1,1,1]]`（不内置归一化），归一化完全由推理代码外部完成。这样 RKNN 路径可以与 ONNX 路径共用同一套归一化逻辑，避免差异。

### 2.4 Batch 推理处理

RKNN 静态 shape 模型不支持 batch>1，`RknnOcrSession.run` 中实现了逐样本推理后拼接：

```python
if first_raw.ndim == 4 and first_raw.shape[0] > 1:
    # batch>1：逐个推理后拼接（RKNN 静态 shape 不支持 batch）
    batch_outputs = []
    for i in range(first_raw.shape[0]):
        single_input = np.ascontiguousarray(self._resize_input(first_raw[i:i+1]))
        outputs = self._rknn_lite.inference(inputs=[single_input], data_format="nchw")
        batch_outputs.append(np.asarray(outputs[0]))
    return [np.concatenate(batch_outputs, axis=0)]
```

**性能影响**：RKNN 静态 shape 不支持 batch，`batch=N` 时实际上是**串行推理 N 次**。实测 rec 模型：
- `batch=1`：约 33ms
- `batch=2`：约 61ms（≈ 2×33ms，没有并行加速）

相比之下 ONNX Runtime 支持 batch 并行（2 个样本只比 1 个样本略慢）。因此 RKNN 在 batch>1 场景下的优势会被削弱，但单样本延迟仍优于 CPU。

> **优化方向**：若需进一步提升 batch 性能，可参考人脸/CLIP 的 `RknnPoolExecutor`，加载 N 份模型副本到 N 个 NPU 核心，用 `ThreadPoolExecutor` 并行推理 batch 中的多个样本。代价是内存占用 ×N（每份 rec 模型约 50MB）。

---

## 3. CPU vs RKNN 结果对比

### 3.1 测试图片

使用脚本生成包含两行英文文字的图片（`scripts/test_ocr.py` 默认行为）：
- "Hello OCR 2024"
- "Immich RKNN Test"

### 3.2 测试结果（20 轮 warmup 后 median）

| 指标 | CPU (ONNX) | RKNN (NPU) |
|---|---|---|
| 检测耗时 | 55.19ms | 88.84ms |
| 识别耗时（batch=2） | 94.16ms | **62.63ms** (提速 1.5x) |
| 总耗时 | 149.51ms | 151.49ms |
| 吞吐量 | 6.69 FPS | 6.60 FPS |
| 检测框数量 | 2 | 2 |
| 识别结果 1 | "Immich RKNN Test" (score: 0.992) | "Immich RKNN Test" (score: 0.702) |
| 识别结果 2 | "Hello OCR 2024" (score: 0.966) | "Hello OCR 2024" (score: 0.705) |

### 3.3 结果分析

- **识别正确性**：两路径均正确识别两行文字，结果完全一致
- **检测性能**：RKNN 略慢于 CPU（88ms vs 55ms），因为 det 模型在 NPU 上需要额外的 pad 预处理，且 DBNet 后处理在 CPU 上执行
- **识别性能**：RKNN 比 CPU 快 **1.5 倍**（62ms vs 94ms），NPU 在序列模型上优势明显
- **总体性能**：两路径总耗时基本持平（150ms 左右），RKNN 在识别阶段的增益被检测阶段的损耗抵消
- **置信度**：RKNN 略低（0.70 vs 0.97），因为使用 FP16 量化（未做 INT8 量化），但识别结果完全正确

### 3.4 性能数据真实性说明

> **重要**：早期文档曾记录 RKNN 识别耗时 41ms（相比 CPU 105ms 提速 2.5x），这是**虚假数据**。原因是早期 `RknnOcrSession.run` 的 batch 处理存在 bug，`batch=2` 时实际只推理了第一个样本（漏识别第二个文本），导致看起来很快。

修复 batch 串行推理 bug 后，`batch=2` 时 RKNN 会正确推理两个样本（耗时 61ms ≈ 2×33ms），与 CPU（94ms，batch 并行）相比仍快 1.5 倍，这是真实性能数据。

### 3.5 性能优化建议

如需进一步提升性能和精度，可参考官方进行 **INT8 量化**：
1. 准备量化数据集（约 100-200 张代表性图片）
2. 修改 `scripts/convert_ocr_rknn.py`，启用 `do_quantization=True` 并传入 `dataset` 文件
3. INT8 量化后速度可再提升 2-3 倍，精度接近 FP32

如需提升 batch 性能：
1. 启用 `RknnPoolExecutor` 多副本并行（参考人脸/CLIP 实现）
2. 加载 N 份 rec 模型到 N 个 NPU 核心，`batch=N` 时并行推理

---

## 4. 调试 Python 运行方法

### 4.1 环境准备

#### 4.1.1 安装 RKNN 运行时库

```bash
# 检查 librknnrt.so 是否已安装
ls -la /usr/lib/librknnrt.so

# 若未安装，从 rknn_model_zoo 仓库获取
# 参考: https://github.com/airockchip/rknn_model_zoo/blob/main/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so
sudo cp /tmp/librknnrt.so /usr/lib/librknnrt.so
```

#### 4.1.2 安装 OpenCV 依赖（libGL.so.1）

```bash
# 若报错: ImportError: libGL.so.1: cannot open shared object file
# 方案1: apt 安装
sudo apt install -y libgl1

# 方案2: 手动下载 .deb 并提取（无 sudo 权限时）
# 参考 machine-learning 项目的 ~/.local/share/lib 目录
export LD_LIBRARY_PATH=~/.local/share/lib:$LD_LIBRARY_PATH
```

#### 4.1.3 同步项目依赖

```bash
cd machine-learning
uv sync --python 3.12 --extra rknn
```

> **注意**：必须使用 Python 3.12（不是 3.13），因为 `rknn-toolkit-lite2` 仅支持到 3.12。项目的 `.python-version` 已设置为 3.12。

### 4.2 模型转换（首次必须）

由于 `rknn-toolkit2`（完整版）要求 `numpy<=1.26.4`，与 immich-ml 的 `numpy>=2.4.0` 冲突，转换需在隔离环境运行：

```bash
cd machine-learning

# 设置 libGL 路径（若需要）
export LD_LIBRARY_PATH=~/.local/share/lib

# 转换 det + rec 模型
uv run --with "rknn-toolkit2>=2.3.0,<3" --with "numpy<2" --python 3.12 \
    scripts/convert_ocr_rknn.py --output-dir /tmp/ocr-rknn

# 复制到模型缓存目录
cp /tmp/ocr-rknn/ppocrv4_det.rknn ~/.cache/immich_ml/ocr/PP-OCRv5_mobile/detection/model.rknn
cp /tmp/ocr-rknn/ppocrv4_rec.rknn ~/.cache/immich_ml/ocr/PP-OCRv5_mobile/recognition/model.rknn
```

### 4.3 运行测试

#### 4.3.1 RKNN (NPU) 模式

```bash
cd machine-learning

LD_LIBRARY_PATH=~/.local/share/lib \
    MACHINE_LEARNING_RKNN=true \
    uv run --python 3.12 --extra rknn \
    scripts/test_ocr.py
```

#### 4.3.2 CPU (ONNX) 模式

```bash
cd machine-learning

MACHINE_LEARNING_RKNN=false \
    uv run --python 3.12 \
    scripts/test_ocr.py
```

#### 4.3.3 指定测试图片

```bash
LD_LIBRARY_PATH=~/.local/share/lib \
    MACHINE_LEARNING_RKNN=true \
    uv run --python 3.12 --extra rknn \
    scripts/test_ocr.py --image /path/to/your/image.png
```

### 4.4 调试技巧

#### 4.4.1 VS Code 调试配置

在 `.vscode/launch.json` 中添加：

```json
{
    "name": "OCR RKNN Test",
    "type": "debugpy",
    "request": "launch",
    "program": "${workspaceFolder}/scripts/test_ocr.py",
    "console": "integratedTerminal",
    "cwd": "${workspaceFolder}",
    "env": {
        "MACHINE_LEARNING_RKNN": "true",
        "LD_LIBRARY_PATH": "~/.local/share/lib"
    },
    "python": "${workspaceFolder}/.venv/bin/python"
}
```

#### 4.4.2 添加诊断日志

临时在以下位置添加 print 调试：

- **det 输出检查**：[immich_ml/models/ocr/detection.py](immich_ml/models/ocr/detection.py) 的 `_predict` 方法
  ```python
  print(f"  [det] input={transformed.shape}, output={out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")
  ```

- **rec 推理检查**：[immich_ml/models/ocr/recognition.py](immich_ml/models/ocr/recognition.py) 的 `_predict_rknn` 方法
  ```python
  print(f"  [rec] batch={norm_img_batch.shape}, preds={preds.shape}, range=[{preds.min():.4f}, {preds.max():.4f}]")
  ```

- **RKNN session 检查**：[immich_ml/sessions/rknn/ocr.py](immich_ml/sessions/rknn/ocr.py) 的 `run` 方法
  ```python
  print(f"  [rknn] input={inputs_list[0].shape}, output={outputs[0].shape}")
  ```

#### 4.4.3 检查中间结果

```bash
# 测试后检查生成的图片
ls -la /tmp/ocr_test.png              # 原始测试图片
ls -la /tmp/ocr_crop_*.png            # crop 后的文本区域
```

#### 4.4.4 常见问题排查

**问题 1：`input size(XXX) < model input size(YYY)`**
- 原因：传入数据尺寸与 RKNN 模型静态 shape 不匹配
- 解决：检查 `RknnOcrSession` 的 `input_shape` 参数，确保 det 为 `(1, 3, 480, 480)`，rec 为 `(1, 3, 48, 320)`

**问题 2：识别结果为空或乱码（如 "Hello OCR 2024" → "Hello OC R 2024"）**
- 检查 det 模型归一化：RKNN 路径应**不归一化**（保持 0-255），由模型内部处理
- 检查 rec 模型归一化：RKNN 路径应使用 **[-1,1] 归一化**（`pixel/255 - 0.5` 再 `/0.5`），与 ONNX 路径完全一致
  - ⚠️ 不要用 `[0,1]` 归一化（`pixel/255`），会导致识别错误
- 检查 det 模型是否采用"保持宽高比 + pad"策略，直接 resize 到 480×480 会导致文字变形失真
- 检查 det 后处理是否过滤 pad 区域误检框（`y >= resize_h` 的框应丢弃）
- 检查字符表：`ppocr_keys_v1.txt` 是否存在于 `~/.cache/immich_ml/ocr/PP-OCRv5_mobile/recognition/`

**问题 3：batch>1 时只返回一个结果**
- 原因：RKNN 静态 shape 不支持 batch>1，`_resize_input` 会截断
- 解决：已修复，`RknnOcrSession.run` 现在会逐样本推理后拼接

**问题 4：`libGL.so.1: cannot open shared object file`**
- 解决：`export LD_LIBRARY_PATH=~/.local/share/lib:$LD_LIBRARY_PATH`

---

## 5. 关键文件清单

| 文件 | 作用 |
|---|---|
| [scripts/convert_ocr_rknn.py](scripts/convert_ocr_rknn.py) | ONNX → RKNN 模型转换脚本 |
| [scripts/test_ocr.py](scripts/test_ocr.py) | OCR 端到端测试脚本（CPU/RKNN 双路径） |
| [immich_ml/models/ocr/detection.py](immich_ml/models/ocr/detection.py) | OCR 检测模型（双路径预处理） |
| [immich_ml/models/ocr/recognition.py](immich_ml/models/ocr/recognition.py) | OCR 识别模型（双路径预处理） |
| [immich_ml/sessions/rknn/ocr.py](immich_ml/sessions/rknn/ocr.py) | RKNN OCR Session（NPU 推理封装） |

---

## 6. 参考资料

- [rknn_model_zoo PPOCR-Det](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR/PPOCR-Det)
- [rknn_model_zoo PPOCR-Rec](https://github.com/airockchip/rknn_model_zoo/tree/main/examples/PPOCR/PPOCR-Rec)
- [PPOCR-Det python demo (ppocr_det.py)](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/PPOCR/PPOCR-Det/python/ppocr_det.py)
- [PPOCR-Rec python demo (ppocr_rec.py)](https://github.com/airockchip/rknn_model_zoo/blob/main/examples/PPOCR/PPOCR-Rec/python/ppocr_rec.py)
- [RKNN-Toolkit2 文档](https://github.com/airockchip/rknn-toolkit2)
- [RapidOCR 项目](https://github.com/RapidAI/RapidOCR)
