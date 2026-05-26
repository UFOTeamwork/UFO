# -*- coding: utf-8 -*-
"""
face_similarity.py
==================
符合 ArcFace 标准流程的人脸相似度评估器：

  1. 背景分割 (mediapipe selfie segmentation) → 白底  [可选, 默认关闭]
  2. MTCNN / RetinaFace 检测人脸
  3. 提取 5 点 landmarks
  4. 五点对齐 (Umeyama similarity transform)
  5. Affine crop → 112×112
  6. 标准预处理 (ToTensor + Normalize[-1,1])
  7. ArcFace backbone 提取特征 → L2 归一化 → 余弦相似度

修正说明
--------
  [Fix 1] MTCNN 在 FaceDetector.__init__ 中初始化一次，批量评估不重复加载
  [Fix 2] get_embedding 绕过 arcface 包装层，自行做标准预处理后直接调 backbone.forward()
          预处理: ToTensor → Normalize(mean=0.5, std=0.5) → [-1, 1]
          与 utils.py preprocess_input(x /= 255; x -= 0.5; x /= 0.5) 完全等价
  [Fix 3] remove_bg 默认改为 False；背景分割改用 mediapipe（更快更准）
  [Fix 4] Arcface wrapper 属性名是 .net 而非 .model；
          cuda=True 时 .net 被 DataParallel 包装，需要 .module 解包；
          import 时用 ArcfaceWrapper 别名避免与本文件类名冲突
  [Compat] compute_similarity 保持原接口语义：返回 L2 distance，调用方零改动

Usage
-----
python face_similarity.py \
  --ref  /path/to/ref.png \
  --gen  /path/to/gen.png \
  --arcface_root /path/to/arcface-pytorch \
  --model_path   /path/to/arcface_mobilefacenet.pth \
  --cuda

eg.
python src/evaluator_pipeline/face_similarity.py \
  --ref "/data/zhangdanning/Projects/ref-ldm/data/FFHQ-Ref/images1024x1024/00078.png" \
  --gen "/data/zhangdanning/Projects/ref-ldm/data/FFHQ-Ref/images1024x1024/03201.png" \
  --arcface_root /data/zhangdanning/Projects/Evaluation/arcface-pytorch \
  --model_path /data/zhangdanning/Projects/Evaluation/arcface-pytorch/model_data/arcface_mobilefacenet.pth \
  --remove_bg \
  --cuda

python src/evaluator_pipeline/face_similarity.py \
  --ref "/data/zhuangshuhan/benchmark/our_bench_final/Human/Face/15.png" \
  --gen "/data/zhuangshuhan/benchmark/eval_results/omnigen2/Human/Face/Complex-Editing/15_template0_sample0.png" \
  --arcface_root /data/zhangdanning/Projects/Evaluation/arcface-pytorch \
  --model_path /data/zhangdanning/Projects/Evaluation/arcface-pytorch/model_data/arcface_mobilefacenet.pth \
  --remove_bg \
  --cuda

    --gen "/data/zhuangshuhan/benchmark/eval_results/qwen/Human/Face/Complex-Editing/15_template0_sample3.png" \

    python src/evaluator_pipeline/face_similarity.py \
  --ref "/data/zhuangshuhan/benchmark/our_bench_final/Human/Face/15.png" \
  --gen "/data/zhuangshuhan/benchmark/eval_results/qwen/Human/Face/Complex-Editing/15_template0_sample3.png" \
  --arcface_root /data/zhangdanning/Projects/Evaluation/arcface-pytorch \
  --model_path /data/zhangdanning/Projects/Evaluation/arcface-pytorch/model_data/arcface_mobilefacenet.pth \
  --remove_bg \
  --debug_dir ./tmp/debug_15_qwen \
  --cuda
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

# ── Environment defaults ────────────────────────────────────────────────────
DEFAULT_ARCFACE_ROOT = os.getenv("UFO_ARCFACE_ROOT")
DEFAULT_ARCFACE_MODEL_PATH = os.getenv("UFO_ARCFACE_MODEL_PATH")

# ── ArcFace 标准 5 点参考坐标 (112×112)  InsightFace 官方值 ────────────────
ARCFACE_REF_POINTS = np.array(
    [
        [38.2946, 51.6963],  # 左眼
        [73.5318, 51.5014],  # 右眼
        [56.0252, 71.7366],  # 鼻尖
        [41.5493, 92.3655],  # 左嘴角
        [70.7299, 92.2041],  # 右嘴角
    ],
    dtype=np.float32,
)

# ── ArcFace 标准图像预处理 ──────────────────────────────────────────────────
# 与 arcface-pytorch/utils/utils.py preprocess_input 完全等价：
#   image /= 255.0      →  ToTensor 完成
#   image -= 0.5        →  Normalize(mean=0.5) 完成
#   image /= 0.5        →  Normalize(std=0.5)  完成
# 结果: uint8 [0,255] → float [-1.0, 1.0]
ARCFACE_TRANSFORM = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


# ── 工具函数 ────────────────────────────────────────────────────────────────

def clean_path(path: Optional[str], name: str) -> str:
    if path is None:
        raise ValueError(f"{name} is None. Please set it in config or env.")
    path = str(path).strip().strip('"').strip("'")
    if not path:
        raise ValueError(f"{name} is empty.")
    return os.path.abspath(path)


# ── Step 1: 背景分割 → 白底 (可选) ─────────────────────────────────────────

MEDIAPIPE_MODEL_PATH = "/tmp/selfie_segmenter_landscape.tflite"

def remove_background_to_white(
    img: Image.Image,
    mediapipe_model_path: str = MEDIAPIPE_MODEL_PATH,
) -> Image.Image:
    """
    背景分割 → 白底。
    fallback 链：mediapipe 0.10+ tasks API → mediapipe 旧版 solutions → rembg → 原图+警告

    mediapipe 0.10+ 需要 tflite 模型，下载命令：
      wget -O /tmp/selfie_segmenter_landscape.tflite \
        https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite
    """
    img_rgb = np.array(img.convert("RGB"))

    # ── 优先：mediapipe 0.10+ tasks API ─────────────────────────────────
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        if not os.path.exists(mediapipe_model_path):
            raise RuntimeError(
                f"mediapipe tflite model not found: {mediapipe_model_path}\n"
                "Download with:\n"
                "  wget -O /tmp/selfie_segmenter_landscape.tflite \\\n"
                "    https://storage.googleapis.com/mediapipe-models/"
                "image_segmenter/selfie_segmenter_landscape/float16/latest/"
                "selfie_segmenter_landscape.tflite"
            )

        base_options = mp_python.BaseOptions(model_asset_path=mediapipe_model_path)
        options = mp_vision.ImageSegmenterOptions(
            base_options=base_options,
            output_confidence_masks=True,  # confidence_masks[0] 是前景置信度
        )
        with mp_vision.ImageSegmenter.create_from_options(options) as segmenter:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            seg_result = segmenter.segment(mp_image)

        # confidence_masks[0]: (H, W, 1) float32，> 0.5 为人像前景
        conf = seg_result.confidence_masks[0].numpy_view()
        mask = (conf > 0.5).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[:, :, 0]           # (H, W, 1) → (H, W)
        mask3 = mask[:, :, np.newaxis]     # (H, W) → (H, W, 1)
        white_bg = np.full_like(img_rgb, 255)
        out = np.where(mask3, img_rgb, white_bg).astype(np.uint8)
        logger.info("Background removed via mediapipe 0.10+ tasks API")
        return Image.fromarray(out)

    except RuntimeError as e:
        logger.warning(str(e))
    except (ImportError, Exception) as e:
        logger.warning(f"mediapipe tasks API failed ({e}). Trying solutions API.")

    # ── fallback 1：mediapipe 旧版 solutions API (<0.10) ─────────────────
    try:
        import mediapipe as mp
        mp_selfie = mp.solutions.selfie_segmentation          # type: ignore
        with mp_selfie.SelfieSegmentation(model_selection=1) as seg:
            seg_result = seg.process(img_rgb)
        mask = (seg_result.segmentation_mask > 0.5).astype(np.uint8)
        mask3 = mask[:, :, np.newaxis]
        white_bg = np.full_like(img_rgb, 255)
        out = np.where(mask3, img_rgb, white_bg).astype(np.uint8)
        logger.info("Background removed via mediapipe solutions API (<0.10)")
        return Image.fromarray(out)

    except (ImportError, AttributeError, Exception) as e:
        logger.warning(f"mediapipe solutions API failed ({e}). Trying rembg.")

    # ── fallback 2：rembg ────────────────────────────────────────────────
    try:
        from rembg import remove as rembg_remove
        rgba: Image.Image = rembg_remove(img.convert("RGBA"))
        white_bg_pil = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        white_bg_pil.paste(rgba, mask=rgba.split()[3])
        logger.info("Background removed via rembg")
        return white_bg_pil.convert("RGB")

    except ImportError:
        logger.warning(
            "No background removal backend available. "
            "Install with: pip install rembg"
        )
        return img.convert("RGB")
    
# ── Step 2–3: 人脸检测 + 5 点 landmarks ────────────────────────────────────

class FaceDetector:
    """
    [Fix 1] 统一检测器封装，初始化时加载模型一次，供整个评估过程复用。
    支持 backend: "mtcnn" | "retinaface"
    """

    def __init__(self, detector: str = "mtcnn", use_cuda: bool = False):
        self.detector = detector
        self.device = "cuda" if use_cuda else "cpu"
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if self.detector == "mtcnn":
            try:
                from facenet_pytorch import MTCNN
            except ImportError:
                raise ImportError(
                    "facenet-pytorch not installed. "
                    "Install with: pip install facenet-pytorch"
                )
            # [Fix 1] 只在此处实例化一次，不在 detect() 内部重复 new
            self._model = MTCNN(
                keep_all=False,    # 只返回最高置信度的人脸
                min_face_size=20,  # 过滤极小误检
                device=self.device,
            )
            logger.info(f"MTCNN loaded on {self.device}")

        elif self.detector == "retinaface":
            try:
                from retinaface import RetinaFace  # noqa: F401
            except ImportError:
                raise ImportError(
                    "retinaface not installed. "
                    "Install with: pip install retina-face"
                )
            # RetinaFace 通过静态方法调用，无需实例化
            self._model = "retinaface_ready"
            logger.info("RetinaFace ready")

        else:
            raise ValueError(
                f"Unknown detector: '{self.detector}'. "
                "Choose 'mtcnn' or 'retinaface'."
            )

    def detect(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        """
        检测人脸，返回 5 点 landmarks，shape (5, 2) float32。
        无人脸时返回 None。
        """
        if self.detector == "mtcnn":
            return self._detect_mtcnn(img_rgb)
        return self._detect_retinaface(img_rgb)

    def _detect_mtcnn(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        pil_img = Image.fromarray(img_rgb)
        boxes, probs, landmarks = self._model.detect(pil_img, landmarks=True)

        if landmarks is None or len(landmarks) == 0:
            return None

        # landmarks: (N, 5, 2)，取置信度最高的人脸
        best = int(np.argmax(probs))
        return landmarks[best].astype(np.float32)  # (5, 2)

    def _detect_retinaface(self, img_rgb: np.ndarray) -> Optional[np.ndarray]:
        from retinaface import RetinaFace

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        faces = RetinaFace.detect_faces(img_bgr)

        if not isinstance(faces, dict) or len(faces) == 0:
            return None

        # 取面积最大的人脸（主体人脸通常最大）
        best_face = max(
            faces.values(),
            key=lambda f: (
                (f["facial_area"][2] - f["facial_area"][0])
                * (f["facial_area"][3] - f["facial_area"][1])
            ),
        )

        lm = best_face["landmarks"]
        pts = np.array(
            [
                lm["left_eye"],
                lm["right_eye"],
                lm["nose"],
                lm["mouth_left"],
                lm["mouth_right"],
            ],
            dtype=np.float32,
        )
        return pts  # (5, 2)


# ── Step 4–5: 五点 Similarity Transform + Affine Crop ─────────────────────

def estimate_similarity_transform(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """
    Umeyama (1991) 最小二乘相似变换 (s·R + t)。
    src_pts: 检测到的 landmarks (5, 2)
    dst_pts: ArcFace 标准参考点  (5, 2)
    返回:    2×3 仿射矩阵 float64
    """
    assert src_pts.shape == dst_pts.shape == (5, 2)

    src = src_pts.T  # (2, 5)
    dst = dst_pts.T  # (2, 5)

    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)

    src_c = src - src_mean
    dst_c = dst - dst_mean

    src_var = (src_c ** 2).mean()
    cov = (dst_c @ src_c.T) / 5  # (2, 2)

    U, S, Vt = np.linalg.svd(cov)

    # 防反射：保证行列式 > 0（纯旋转，非镜像）
    det_sign = np.linalg.det(U @ Vt)
    D = np.diag([1.0, det_sign])

    R = U @ D @ Vt                                      # (2, 2) 旋转矩阵
    scale = (S * D.diagonal()).sum() / (src_var + 1e-8)
    t = dst_mean - scale * R @ src_mean                 # (2, 1) 平移向量

    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = t.ravel()
    return M


def align_face(
    img_rgb: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 112,
) -> np.ndarray:
    """
    Affine warp 人脸到 112×112，越界区域填充白色。
    返回: (112, 112, 3) uint8 RGB
    """
    M = estimate_similarity_transform(landmarks, ARCFACE_REF_POINTS)
    aligned = cv2.warpAffine(
        img_rgb,
        M,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),  # 白色填充，与白底背景一致
    )
    return aligned  # (112, 112, 3) uint8 RGB


# ── Step 6: ArcFace 特征提取 ────────────────────────────────────────────────

class ArcFaceExtractor:
    """
    [Fix 2 + Fix 4] 直接调用 backbone.forward()，绕过 arcface 包装类。

    标准预处理流程（与 arcface-pytorch/utils/utils.py preprocess_input 等价）：
      uint8 RGB [0, 255]
        → ToTensor       → float [0.0, 1.0]
        → Normalize(0.5) → float [-1.0, 1.0]
        → backbone.forward() → (1, 512)
        → L2 normalize   → (512,)
    """

    def __init__(
        self,
        arcface_root: str,
        model_path: str,
        backbone: str = "mobilefacenet",
        use_cuda: bool = False,
    ):
        arcface_root = clean_path(arcface_root, "arcface_root")
        model_path = clean_path(model_path, "model_path")

        if not os.path.isdir(arcface_root):
            raise FileNotFoundError(f"ArcFace root not found: {arcface_root}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace model not found: {model_path}")

        if arcface_root not in sys.path:
            sys.path.insert(0, arcface_root)

        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.backbone = self._load_backbone(backbone, model_path)
        self.backbone.to(self.device)
        self.backbone.eval()

        logger.info(f"ArcFace backbone '{backbone}' loaded on {self.device}")

    def _load_backbone(self, backbone: str, model_path: str) -> torch.nn.Module:
        """
        从 arcface-pytorch 加载 backbone。
        方式 1: 直接 import nets.arcface 模块（推荐，权重自行加载，无副作用）
        方式 2: 通过 Arcface wrapper 取出内部 .net（fallback）

        [Fix 4] 方式 2 的三处修正：
          - import 时用 ArcfaceWrapper 别名，避免与本文件 class 名冲突
          - wrapper 属性名是 .net（来自 generate() 中 self.net = arcface(...)）
          - cuda=True 时 .net 被 DataParallel 包装，需 .module 解包取原始 nn.Module
        """
        # ── 方式 1: 直接导入 nets.arcface backbone ───────────────────────
        try:
            from nets.arcface import Arcface as ArcfaceNet  # type: ignore

            # arcface-pytorch 的 nets/arcface.py 中 Arcface 即 backbone 网络本身
            # mode="predict" 关闭训练时的 ArcFace head，只保留特征提取部分
            net = ArcfaceNet(backbone=backbone, mode="predict")

            # 加载权重（兼容 DataParallel 训练产生的 'module.' 前缀）
            state = torch.load(model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            state = {k.replace("module.", ""): v for k, v in state.items()}
            net.load_state_dict(state, strict=False)

            logger.info("Backbone loaded via nets.arcface (method 1)")
            return net

        except (ImportError, Exception) as e:
            logger.warning(
                f"Method 1 failed ({e}). "
                "Falling back to Arcface wrapper (method 2)."
            )

        # ── 方式 2: 通过 Arcface wrapper 取出 backbone ───────────────────
        # [Fix 4-a] 用别名 ArcfaceWrapper，避免与本文件顶层 class 名冲突
        try:
            from arcface import Arcface as ArcfaceWrapper  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Failed to import Arcface from arcface_root. "
                "Ensure arcface-pytorch is at the given path."
            ) from e

        # wrapper 构造函数使用 **kwargs，cuda=False 避免 DataParallel 包装
        # model_path 在 wrapper 内部是相对路径，需切换工作目录
        # 注意：wrapper 会打印 Configurations 表格，属正常输出
        wrapper = ArcfaceWrapper(model_path=model_path, backbone=backbone, cuda=False)

        # [Fix 4-b] 属性名是 .net，不是 .model
        #           来源: arcface-pytorch/arcface.py generate() 中 self.net = arcface(...)
        raw_net = wrapper.net

        # [Fix 4-c] cuda=False 时通常无 DataParallel；保险起见统一判断解包
        if isinstance(raw_net, torch.nn.DataParallel):
            raw_net = raw_net.module

        logger.info("Backbone loaded via Arcface wrapper .net (method 2)")
        return raw_net

    def get_embedding(self, aligned_rgb: np.ndarray) -> np.ndarray:
        """
        输入:  已对齐的 112×112 RGB uint8 ndarray
        输出:  L2 归一化特征向量 (512,) float64

        预处理与 arcface-pytorch/utils/utils.py preprocess_input 完全等价：
          uint8 [0,255] → /255 → -0.5 → /0.5  等价于  Normalize(0.5, 0.5)
        """
        pil_img = Image.fromarray(aligned_rgb)            # 确保 PIL RGB
        tensor = ARCFACE_TRANSFORM(pil_img).unsqueeze(0)  # (1, 3, 112, 112)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            feat = self.backbone(tensor)                  # (1, 512)

        feat = feat.cpu().numpy().ravel().astype(np.float64)

        # L2 归一化（余弦相似度的前提）
        norm = np.linalg.norm(feat)
        if norm > 1e-8:
            feat /= norm

        return feat  # (512,)


# ── 主评估器 ────────────────────────────────────────────────────────────────

class FaceSimilarityEvaluator:
    """
    完整 ArcFace 标准流程评估器。

    对外接口（与原代码完全兼容）
    ──────────────────────────────
    evaluator = FaceSimilarityEvaluator(
        arcface_root=..., model_path=..., backbone=...,
        use_cuda=..., enabled=True,
    )
    distance = evaluator.compute_similarity(ref_img, gen_img)   # float, 越小越相似
    similarity_score = max(0.0, 1.0 - distance)                 # 原调用方无需修改

    新增接口
    ────────
    result = evaluator.compute_similarity_with_debug(ref_img, gen_img, save_dir=...)
    # result["cosine_similarity"]  ∈ [-1, 1]，越大越相似（推荐新调用方使用）
    # result["l2_distance"]        与 compute_similarity 返回值相同
    """

    def __init__(
        self,
        arcface_root: str = DEFAULT_ARCFACE_ROOT,
        model_path: str = DEFAULT_ARCFACE_MODEL_PATH,
        backbone: str = "mobilefacenet",
        use_cuda: bool = False,
        enabled: bool = True,
        detector: str = "mtcnn",   # "mtcnn" | "retinaface"
        remove_bg: bool = False,   # [Fix 3] 默认关闭，需要时显式开启
         mediapipe_model_path: str = MEDIAPIPE_MODEL_PATH,  # ← 新增
        **kwargs,                  # 保留，吸收调用方传入的未知参数
    ):
        self.enabled = enabled
        self.remove_bg = remove_bg
        self.mediapipe_model_path = mediapipe_model_path   # ← 新增
        self._detector: Optional[FaceDetector] = None
        self._extractor: Optional[ArcFaceExtractor] = None

        if not self.enabled:
            return

        # [Fix 1] 检测器在此处初始化一次，整个评估过程复用
        self._detector = FaceDetector(detector=detector, use_cuda=use_cuda)

        self._extractor = ArcFaceExtractor(
            arcface_root=arcface_root,
            model_path=model_path,
            backbone=backbone,
            use_cuda=use_cuda,
        )

    # ── 内部：单张图完整预处理 → 112×112 对齐图 + landmarks ─────────────
    def _preprocess(self, img: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 1~5 完整流程。
        返回: (aligned_rgb_112x112, landmarks_5x2)
        检测失败时抛出 RuntimeError。
        """
        # Step 1: 背景→白（可选）
        if self.remove_bg:
            img = remove_background_to_white(img, self.mediapipe_model_path)  # ← 透传

        img_rgb = np.array(img.convert("RGB"))

        # Step 2–3: 检测 5 点 landmarks
        lm = self._detector.detect(img_rgb)
        if lm is None:
            raise RuntimeError("No face detected in the image.")

        # Step 4–5: Umeyama 对齐 + affine crop → 112×112
        aligned = align_face(img_rgb, lm, output_size=112)
        return aligned, lm

    
    # ── 原有接口（向后兼容，调用方零改动）──────────────────────────────────
    def compute_similarity(
        self,
        ref_img: Image.Image,
        gen_img: Image.Image,
    ) -> float:
        """
        与原代码接口完全兼容。
        返回: ArcFace L2 distance (float)，越小越相似。
        内部已走完整标准流程，原调用方无需任何修改。
        """
        if not self.enabled:
            return 0.0

        if not isinstance(ref_img, Image.Image):
            raise TypeError("ref_img must be PIL.Image")
        if not isinstance(gen_img, Image.Image):
            raise TypeError("gen_img must be PIL.Image")

        ref_aligned, _ = self._preprocess(ref_img)
        gen_aligned, _ = self._preprocess(gen_img)

        ref_feat = self._extractor.get_embedding(ref_aligned)
        gen_feat = self._extractor.get_embedding(gen_aligned)

        l2_dist = float(np.linalg.norm(ref_feat - gen_feat))
        return l2_dist

    # ── 新增接口（推荐新调用方使用）────────────────────────────────────────
    def compute_similarity_with_debug(
        self,
        ref_img: Image.Image,
        gen_img: Image.Image,
        save_dir: Optional[str] = None,
    ) -> dict:
        """
        返回完整中间结果。

        Returns
        -------
        {
            "l2_distance":       float,  # 同 compute_similarity() 返回值
            "cosine_similarity": float,  # ∈ [-1,1]，越大越相似，推荐新调用方用此值
            "ref_landmarks":     list,   # 参考图 5 点坐标 [[x,y], ...]
            "gen_landmarks":     list,   # 生成图 5 点坐标
        }

        两者在归一化特征空间下的关系（数学等价）:
            l2_distance = sqrt(2 - 2 * cosine_similarity)
        """
        ref_aligned, ref_lm = self._preprocess(ref_img)
        gen_aligned, gen_lm = self._preprocess(gen_img)

        ref_feat = self._extractor.get_embedding(ref_aligned)
        gen_feat = self._extractor.get_embedding(gen_aligned)

        cosine  = float(np.dot(ref_feat, gen_feat))
        l2_dist = float(np.linalg.norm(ref_feat - gen_feat))

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            Image.fromarray(ref_aligned).save(
                os.path.join(save_dir, "ref_aligned_112.png")
            )
            Image.fromarray(gen_aligned).save(
                os.path.join(save_dir, "gen_aligned_112.png")
            )
            logger.info(f"Saved aligned crops -> {save_dir}")

        return {
            "l2_distance":       l2_dist,
            "cosine_similarity": cosine,
            "ref_landmarks":     ref_lm.tolist(),
            "gen_landmarks":     gen_lm.tolist(),
        }


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="ArcFace face similarity -- corrected standard pipeline."
    )
    parser.add_argument("--ref",          type=str, required=True,
                        help="Reference image path")
    parser.add_argument("--gen",          type=str, required=True,
                        help="Generated image path")
    parser.add_argument("--arcface_root", type=str, default=DEFAULT_ARCFACE_ROOT,
                        help="Path to arcface-pytorch repo root")
    parser.add_argument("--model_path",   type=str, default=DEFAULT_ARCFACE_MODEL_PATH,
                        help="Path to .pth weight file")
    parser.add_argument("--backbone",     type=str, default="mobilefacenet",
                        choices=["mobilefacenet", "ir_se_50", "ir_se_100",
                                 "ir_se_152", "resnet_50", "resnet_101", "resnet_152"])
    parser.add_argument("--cuda",         action="store_true")
    parser.add_argument("--detector",     type=str, default="mtcnn",
                        choices=["mtcnn", "retinaface"])
    parser.add_argument("--remove_bg",    action="store_true",
                        help="Enable background removal via mediapipe")
    parser.add_argument("--debug_dir",    type=str, default=None,
                        help="If set, save aligned 112x112 crops here for inspection")
    args = parser.parse_args()

    evaluator = FaceSimilarityEvaluator(
        arcface_root=args.arcface_root,
        model_path=args.model_path,
        backbone=args.backbone,
        use_cuda=args.cuda,
        enabled=True,
        detector=args.detector,
        remove_bg=args.remove_bg,
    )

    ref_img = Image.open(args.ref).convert("RGB")
    gen_img = Image.open(args.gen).convert("RGB")

    result = evaluator.compute_similarity_with_debug(
        ref_img, gen_img, save_dir=args.debug_dir
    )

    # 原调用方兼容写法（无需修改）
    distance = result["l2_distance"]
    similarity_score = max(0.0, 1.0 - distance)

    print("=" * 60)
    print("ArcFace Face Similarity -- Corrected Standard Pipeline")
    print("=" * 60)
    print(f"  Reference         : {args.ref}")
    print(f"  Generated         : {args.gen}")
    print(f"  Detector          : {args.detector}")
    print(f"  Backbone          : {args.backbone}")
    print(f"  Remove BG         : {args.remove_bg}")
    print("-" * 60)
    print(f"  L2 distance       : {result['l2_distance']:.6f}   down lower  = more similar")
    print(f"  Cosine similarity : {result['cosine_similarity']:+.6f}   up higher = more similar")
    print(f"  Similarity score  : {similarity_score:.6f}   (= max(0, 1-L2), original mapping)")
    print("=" * 60)
    if args.debug_dir:
        print(f"  Aligned crops saved -> {args.debug_dir}")

    # 参考阈值（mobilefacenet，仅供参考）
    cosine = result["cosine_similarity"]
    if cosine >= 0.50:
        label = "High similarity (likely same person)"
    elif cosine >= 0.30:
        label = "Medium similarity"
    elif cosine >= 0.10:
        label = "Low similarity"
    else:
        label = "Not similar (likely different persons)"
    print(f"  Interpretation    : {label}")
    print("=" * 60)