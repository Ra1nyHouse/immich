import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from rapidocr.ch_ppocr_rec import TextRecInput
from rapidocr.ch_ppocr_rec import TextRecognizer as RapidTextRecognizer
from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode
from rapidocr.inference_engine.base import FileInfo, InferSession
from rapidocr.utils.download_file import DownloadFile, DownloadFileInput
from rapidocr.utils.typings import EngineType, LangRec, OCRVersion, TaskType
from rapidocr.utils.typings import ModelType as RapidModelType
from rapidocr.utils.vis_res import VisRes

from immich_ml.config import log, settings
from immich_ml.models.base import InferenceModel
from immich_ml.models.transforms import pil_to_cv2
from immich_ml.schemas import ModelFormat, ModelSession, ModelTask, ModelType
from immich_ml.sessions.ort import OrtSession

from .schemas import OcrOptions, TextDetectionOutput, TextRecognitionOutput

# ppocr_keys_v1.txt 默认下载地址（rapidocr 兜底）
_PPOCR_KEYS_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v2.0.7/paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_infer/ppocr_keys_v1.txt"


class TextRecognizer(InferenceModel):
    depends = [(ModelType.DETECTION, ModelTask.OCR)]
    identity = (ModelType.RECOGNITION, ModelTask.OCR)

    def __init__(self, model_name: str, min_score: float = 0.9, **model_kwargs: Any) -> None:
        self.language = LangRec[model_name.split("__")[0]] if "__" in model_name else LangRec.CH
        self.min_score = model_kwargs.get("minScore", min_score)
        self._empty: TextRecognitionOutput = {
            "box": np.empty(0, dtype=np.float32),
            "boxScore": np.empty(0, dtype=np.float32),
            "text": [],
            "textScore": np.empty(0, dtype=np.float32),
        }
        VisRes.__init__ = lambda self, **kwargs: None  # pyright: ignore[reportAttributeAccessIssue]
        super().__init__(model_name, **model_kwargs)
        # RKNN 路径专用属性
        self._rknn_rec_batch_num = 6
        self._rknn_rec_img_shape = (3, 48, 320)
        self._rknn_postprocess: CTCLabelDecode | None = None

    def model_path_for_format(self, model_format: ModelFormat) -> Path:
        if model_format == ModelFormat.RKNN:
            return self.model_dir / f"model.{model_format}"
        return super().model_path_for_format(model_format)

    def _download(self) -> None:
        if self.model_format == ModelFormat.RKNN:
            # 从 fork 仓库的 GitHub raw 地址自动下载预转换的 .rknn 模型
            # 仓库：https://github.com/Ra1nyHouse/immich (feature/rknn-orc 分支)
            # 模型由 scripts/convert_ocr_rknn.py 转换，FP16，rec 输入 48x320
            rknn_url = (
                "https://raw.githubusercontent.com/Ra1nyHouse/immich/feature/rknn-orc/"
                f"machine-learning/models/rknn/ocr/{self.model_name}/recognition/model.rknn"
            )
            log.info(f"Downloading OCR RKNN recognition model from {rknn_url}")
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
                task_type=TaskType.REC,
                lang_type=self.language,
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
            return self._load_rknn()
        return self._load_onnx()

    def _load_onnx(self) -> ModelSession:
        session = OrtSession(self.model_path)
        max_batch_size = settings.max_batch_size and settings.max_batch_size.ocr
        self.model = RapidTextRecognizer(
            OcrOptions(
                session=session.session,
                rec_batch_num=max_batch_size if max_batch_size else 6,
                rec_img_shape=(3, 48, 320),
                lang_type=self.language,
                model_root_dir=self.cache_dir,
            )
        )
        return session

    def _load_rknn(self) -> ModelSession:
        from immich_ml.sessions.rknn.ocr import RknnOcrSession

        # RKNN rec 模型为 static shape (1, 3, 48, 320)
        session = RknnOcrSession(self.model_path, input_shape=(1, 3, 48, 320))
        max_batch_size = settings.max_batch_size and settings.max_batch_size.ocr
        self._rknn_rec_batch_num = max_batch_size if max_batch_size else 6

        # 加载字符表（.rknn 无 ONNX 元数据，使用 ppocr_keys_v1.txt）
        dict_path = self.model_dir / "ppocr_keys_v1.txt"
        if not dict_path.exists():
            log.info(f"Downloading character dictionary to {dict_path}")
            DownloadFile.run(
                DownloadFileInput(
                    file_url=_PPOCR_KEYS_URL,
                    sha256=None,
                    save_path=dict_path,
                    logger=log,
                )
            )
        self._rknn_postprocess = CTCLabelDecode(character_path=dict_path.as_posix())
        self.model = None  # 标记使用 RKNN 路径
        return session

    def _predict(self, img: Image.Image, texts: TextDetectionOutput) -> TextRecognitionOutput:
        boxes, box_scores = texts["boxes"], texts["scores"]
        if boxes.shape[0] == 0:
            return self._empty

        if self.model_format == ModelFormat.RKNN:
            txts, text_scores = self._predict_rknn(img, boxes)
        else:
            rec = self.model(TextRecInput(img=self.get_crop_img_list(img, boxes)))
            if rec.txts is None:
                return self._empty
            txts = list(rec.txts)
            text_scores = np.array(rec.scores)

        boxes[:, :, 0] /= img.width
        boxes[:, :, 1] /= img.height

        valid_text_score_idx = text_scores > self.min_score
        valid_score_idx_list = valid_text_score_idx.tolist()
        return {
            "box": boxes.reshape(-1, 8)[valid_text_score_idx].reshape(-1),
            "text": [txts[i] for i in range(len(txts)) if valid_score_idx_list[i]],
            "boxScore": box_scores[valid_text_score_idx],
            "textScore": text_scores[valid_text_score_idx],
        }

    def _predict_rknn(self, img: Image.Image, boxes: NDArray[np.float32]) -> tuple[list[str], NDArray[np.float32]]:
        """RKNN 识别路径：resize_norm_img → NPU 推理 → CTC 解码。"""
        img_list = self.get_crop_img_list(img, boxes)
        img_num = len(img_list)
        batch_num = self._rknn_rec_batch_num
        imgC, imgH, imgW = self._rknn_rec_img_shape[:3]

        txts: list[str] = [""] * img_num
        scores: list[float] = [0.0] * img_num

        # 按宽高比排序以加速批量推理（与 RapidTextRecognizer 一致）
        width_list = [im.shape[1] / float(im.shape[0]) for im in img_list]
        indices = np.argsort(np.array(width_list))

        # RKNN rec 模型为 static shape (48, 320)，_resize_norm_img 会 padding 到 imgW
        # max_wh_ratio 用真实值（上限 imgW/imgH 防止超过模型固定宽度），
        # padding 自然填充到模型固定宽度
        max_wh_ratio_limit = imgW / imgH
        for beg in range(0, img_num, batch_num):
            end = min(img_num, beg + batch_num)
            max_wh_ratio = max_wh_ratio_limit
            wh_ratio_list = []
            for ino in range(beg, end):
                h, w = img_list[indices[ino]].shape[:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            # 上限保护：防止 img_width 超过模型固定宽度
            max_wh_ratio = min(max_wh_ratio, max_wh_ratio_limit)

            norm_img_batch = []
            for ino in range(beg, end):
                # 参考 rknn_model_zoo PPOCR-Rec 官方预处理：
                # PRE_PROCESS_CONFIG: mean=0, std=1, scale=1./255.
                # 即归一化到 [0,1]（不是 [-1,1]）
                # RKNN 转换时 mean=0/std=1，归一化由外部完成
                norm_img = self._resize_norm_img(img_list[indices[ino]], max_wh_ratio, imgC, imgH, imgW, normalize=True)
                norm_img_batch.append(norm_img[np.newaxis, :])
            norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

            preds = self.session.run(None, {"x": norm_img_batch})[0]
            line_results, _ = self._rknn_postprocess(
                preds,
                False,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )
            for rno, (text, score) in enumerate(line_results):
                txts[indices[beg + rno]] = text
                scores[indices[beg + rno]] = score

        return txts, np.array(scores, dtype=np.float32)

    @staticmethod
    def _resize_norm_img(img: NDArray[np.uint8], max_wh_ratio: float, imgC: int, imgH: int, imgW: int,
                         normalize: bool = True) -> NDArray[np.float32]:
        """与 RapidTextRecognizer.resize_norm_img 一致。

        Args:
            normalize: True 归一化到 [-1,1]（与 RapidOCR ONNX 路径一致：
                       pixel/255 - 0.5 再 /0.5）；
                       False 保留 0-255 原始像素值。
        """
        img_width = int(imgH * max_wh_ratio)
        h, w = img.shape[:2]
        ratio = w / float(h)
        resized_w = img_width if math.ceil(imgH * ratio) > img_width else int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1))
        if normalize:
            # 与 RapidOCR ONNX 路径完全一致：归一化到 [-1, 1]
            # RapidOCR resize_norm_img: /255 - 0.5 再 /0.5
            resized_image = resized_image / 255.0
            resized_image -= 0.5
            resized_image /= 0.5

        padding_im = np.zeros((imgC, imgH, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def get_crop_img_list(self, img: Image.Image, boxes: NDArray[np.float32]) -> list[NDArray[np.uint8]]:
        img_crop_width = np.maximum(
            np.linalg.norm(boxes[:, 1] - boxes[:, 0], axis=1), np.linalg.norm(boxes[:, 2] - boxes[:, 3], axis=1)
        ).astype(np.int32)
        img_crop_height = np.maximum(
            np.linalg.norm(boxes[:, 0] - boxes[:, 3], axis=1), np.linalg.norm(boxes[:, 1] - boxes[:, 2], axis=1)
        ).astype(np.int32)
        pts_std = np.zeros((img_crop_width.shape[0], 4, 2), dtype=np.float32)
        pts_std[:, 1:3, 0] = img_crop_width[:, None]
        pts_std[:, 2:4, 1] = img_crop_height[:, None]

        img_crop_sizes = np.stack([img_crop_width, img_crop_height], axis=1)
        all_coeffs = self._get_perspective_transform(pts_std, boxes)
        imgs: list[NDArray[np.uint8]] = []
        for coeffs, dst_size in zip(all_coeffs, img_crop_sizes):
            dst_img = img.transform(
                size=tuple(dst_size),
                method=Image.Transform.PERSPECTIVE,
                data=tuple(coeffs),
                resample=Image.Resampling.BICUBIC,
            )

            dst_width, dst_height = dst_img.size
            if dst_height * 1.0 / dst_width >= 1.5:
                dst_img = dst_img.rotate(90, expand=True)
            imgs.append(pil_to_cv2(dst_img))

        return imgs

    def _get_perspective_transform(self, src: NDArray[np.float32], dst: NDArray[np.float32]) -> NDArray[np.float32]:
        N = src.shape[0]
        x, y = src[:, :, 0], src[:, :, 1]
        u, v = dst[:, :, 0], dst[:, :, 1]
        A = np.zeros((N, 8, 9), dtype=np.float32)

        # Fill even rows (0, 2, 4, 6): [x, y, 1, 0, 0, 0, -u*x, -u*y, -u]
        A[:, ::2, 0] = x
        A[:, ::2, 1] = y
        A[:, ::2, 2] = 1
        A[:, ::2, 6] = -u * x
        A[:, ::2, 7] = -u * y
        A[:, ::2, 8] = -u

        # Fill odd rows (1, 3, 5, 7): [0, 0, 0, x, y, 1, -v*x, -v*y, -v]
        A[:, 1::2, 3] = x
        A[:, 1::2, 4] = y
        A[:, 1::2, 5] = 1
        A[:, 1::2, 6] = -v * x
        A[:, 1::2, 7] = -v * y
        A[:, 1::2, 8] = -v

        # Solve using SVD for all matrices at once
        _, _, Vt = np.linalg.svd(A)
        H = Vt[:, -1, :].reshape(N, 3, 3)
        H = H / H[:, 2:3, 2:3]

        # Extract the 8 coefficients for each transformation
        return np.column_stack(
            [H[:, 0, 0], H[:, 0, 1], H[:, 0, 2], H[:, 1, 0], H[:, 1, 1], H[:, 1, 2], H[:, 2, 0], H[:, 2, 1]]
        )  # pyright: ignore[reportReturnType]

    def configure(self, **kwargs: Any) -> None:
        self.min_score = kwargs.get("minScore", self.min_score)
