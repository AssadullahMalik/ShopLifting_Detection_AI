"""
Shared VideoMAE clip-classification interface.

This is the single place that turns a 16-frame clip into a shoplifting /
non-shoplifting probability. Both the Streamlit dashboard and the CLI
inference script use it, so model loading, preprocessing and inference stay
consistent and are never duplicated.

Design notes
------------
* Model + processor come from :mod:`src.pipeline.videomae_loader`, which
  loads the trained checkpoint strictly from local files (no downloads) and
  restores the fused ``q_bias``/``v_bias`` attention biases. That
  compatibility behaviour is preserved here and must not be bypassed.
* Preprocessing uses the SAVED ``VideoMAEImageProcessor`` (resize to
  shortest edge 224 + center crop 224 + ImageNet normalize), matching
  training, instead of a hand-rolled torchvision transform.
* BGR -> RGB conversion happens exactly once, inside this module.
* The class mapping is taken from the checkpoint config:
      index 0 -> Non-Shoplifting, index 1 -> Shoplifting.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Union

import numpy as np
import torch
from torchvision import transforms

from src.pipeline import paths
from src.pipeline.videomae_loader import load_videomae_model_and_processor

# ImageNet statistics used verbatim by the training pipeline
# (scripts/train_local.py + src/data/clip_dataset.py). The saved
# preprocessor_config.json carries the SAME mean/std, so only the resize/crop
# behaviour differs -- see _build_clip_transform.
_TRAIN_MEAN = [0.485, 0.456, 0.406]
_TRAIN_STD = [0.229, 0.224, 0.225]

# A clip frame may be an OpenCV BGR ndarray or an already-RGB PIL image.
FrameLike = Union[np.ndarray, "object"]


class VideoMAEClassifier:
    """Load the trained VideoMAE model once and classify 16-frame clips."""

    def __init__(
        self,
        model_dir: Path | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else paths.resolve_videomae_dir()
        self.model, self.processor = load_videomae_model_and_processor(self.model_dir)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # Contract values read straight from the trained checkpoint config.
        self.num_frames = int(self.model.config.num_frames)          # 16
        self.image_size = int(self.model.config.image_size)          # 224
        self.id2label = {int(k): str(v) for k, v in dict(self.model.config.id2label).items()}
        self.clip_length = self.num_frames

        # Preprocessing MUST match how the model was trained, not the default
        # HF processor. train_local.py + clip_dataset.py used a direct
        # Resize((224, 224)) aspect-squash + ToTensor + ImageNet normalize, so
        # we reproduce that exactly. The saved VideoMAEImageProcessor instead
        # does shortest-edge resize + CENTER CROP, which throws away ~44% of a
        # 16:9 CCTV frame's horizontal field and feeds the model
        # out-of-distribution input (verified: shifts logits and can flip the
        # class near the threshold). The processor is still kept loaded for
        # metadata, but clip preprocessing uses _clip_transform below.
        self._clip_transform = self._build_clip_transform()

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #
    def _build_clip_transform(self):
        """Training-equivalent per-frame transform (aspect-squash to 224)."""
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_TRAIN_MEAN, std=_TRAIN_STD),
        ])

    def _to_rgb_pil(self, frame: FrameLike):
        """Convert one frame to an RGB PIL image exactly once.

        Accepts an OpenCV BGR ``ndarray`` (H, W, 3) or a PIL image. Raises
        ``ValueError`` for malformed frames instead of silently misclassifying.
        """
        from PIL import Image

        if isinstance(frame, np.ndarray):
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError(
                    f"Expected a (H, W, 3) BGR array, got shape {frame.shape}"
                )
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                raise ValueError(f"Empty frame with shape {frame.shape}")
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return Image.fromarray(rgb)

        # Assume PIL.Image.Image (duck-typed to avoid a hard import here).
        if hasattr(frame, "convert"):
            return frame.convert("RGB")

        raise ValueError(
            f"Unsupported frame type: {type(frame).__name__}. "
            f"Expected a numpy BGR array or a PIL image."
        )

    def preprocess_clip(self, frames: Sequence[FrameLike]) -> torch.Tensor:
        """Validate a clip and produce processor pixel values.

        Args:
            frames: sequence of exactly ``self.num_frames`` frames (BGR ndarrays
                or PIL images).

        Returns:
            Float tensor of shape (1, num_frames, 3, H, W) on ``self.device``.

        Raises:
            ValueError: on wrong frame count or malformed frames.
        """
        n = len(frames)
        if n != self.num_frames:
            raise ValueError(
                f"VideoMAE requires exactly {self.num_frames} frames per clip, "
                f"got {n}."
            )
        if n == 0:
            raise ValueError("Empty clip: no frames provided.")

        pil_frames = [self._to_rgb_pil(f) for f in frames]
        # Match training exactly: per-frame PIL -> Resize((224,224)) -> ToTensor
        # -> Normalize, then stack into (1, num_frames, 3, 224, 224).
        tensors = [self._clip_transform(pf) for pf in pil_frames]
        pixel_values = torch.stack(tensors).unsqueeze(0)
        return pixel_values.to(self.device)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def classify_clip(self, frames: Sequence[FrameLike]) -> Dict:
        """Classify a 16-frame clip into Non-Shoplifting / Shoplifting.

        Returns a documented dict:
            class_probabilities   [p_non_shoplifting, p_shoplifting] (floats)
            predicted_class_id    int (0 or 1)
            predicted_label       str  (from the checkpoint's id2label)
            shoplifting_prob      float  == class_probabilities[1]
            normal_prob           float  == class_probabilities[0]
            logits                [float, float]

        Raises:
            ValueError: malformed clip / wrong frame count / dims.
            RuntimeError: non-finite model output or inference failure.
        """
        pixel_values = self.preprocess_clip(frames)
        try:
            with torch.no_grad():
                logits = self.model(pixel_values=pixel_values).logits
        except Exception as exc:  # noqa: BLE001 - surface inference failures clearly
            raise RuntimeError(f"VideoMAE inference failed: {type(exc).__name__}: {exc}") from exc

        if logits.shape[-1] != 2:
            raise RuntimeError(
                f"Expected 2 output classes, got logits shape {tuple(logits.shape)}"
            )
        probs = torch.softmax(logits, dim=1)[0]
        if not torch.isfinite(probs).all():
            raise RuntimeError("VideoMAE produced non-finite probabilities")

        p = [float(probs[0].item()), float(probs[1].item())]
        predicted_id = int(torch.argmax(probs).item())
        return {
            "class_probabilities": p,
            "predicted_class_id": predicted_id,
            "predicted_label": self.id2label.get(predicted_id, f"class_{predicted_id}"),
            "shoplifting_prob": p[1],
            "normal_prob": p[0],
            "logits": [float(x) for x in logits[0].tolist()],
        }
