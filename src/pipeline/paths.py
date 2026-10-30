"""
Centralized filesystem paths for trained model assets.

Resolves the project root from THIS module's location (not the current
working directory), so imports work whether the code is launched from the
repository root, the dashboard, or the scripts folder:

    <PROJECT_ROOT>/
        models/                         <- VideoMAE HF model directory
            config.json
            model.safetensors
            preprocessor_config.json
            shoplifting_yolov8n_best.pt <- YOLO checkpoint file

NOTE: The Hugging Face loaders (``from_pretrained`` / ``AutoImageProcessor``)
expect the *directory* ``VIDEOMAE_DIR``, not an individual weight file.
The YOLO loader expects the *file* ``YOLO_WEIGHTS``.
"""
from __future__ import annotations

from pathlib import Path

# .../src/pipeline/paths.py -> parents[2] == project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"

#: Directory holding config.json + model.safetensors + preprocessor_config.json
VIDEOMAE_DIR = MODELS_DIR

#: Single .pt checkpoint file
YOLO_WEIGHTS = MODELS_DIR / "shoplifting_yolov8n_best.pt"

# Weight file names the HF loaders accept, in order of preference.
_VIDEOMAE_WEIGHT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")


def resolve_videomae_dir() -> Path:
    """Validate and return the VideoMAE model directory for HF loaders.

    Raises:
        FileNotFoundError: If the directory or its config/weight files are
            missing, with sa message naming the exact path that was expected.
    """
    if not VIDEOMAE_DIR.is_dir():
        raise FileNotFoundError(
            f"VideoMAE model directory not found: {VIDEOMAE_DIR}\n"
            f"Expected Hugging Face files (config.json + model.safetensors) "
            f"directly inside {MODELS_DIR}. Pass the DIRECTORY to "
            f"from_pretrained(), not an individual weight file."
        )

    if not (VIDEOMAE_DIR / "config.json").is_file():
        raise FileNotFoundError(
            f"VideoMAE config.json not found in: {VIDEOMAE_DIR}"
        )

    weights = [p for p in _VIDEOMAE_WEIGHT_CANDIDATES if (VIDEOMAE_DIR / p).is_file()]
    if not weights:
        raise FileNotFoundError(
            f"No VideoMAE weights found in: {VIDEOMAE_DIR}\n"
            f"Expected one of: {', '.join(_VIDEOMAE_WEIGHT_CANDIDATES)}"
        )
    return VIDEOMAE_DIR


def resolve_yolo_weights() -> Path:
    """Validate and return the path to the trained YOLO .pt checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
    """
    if not YOLO_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"YOLO checkpoint not found: {YOLO_WEIGHTS}\n"
            f"Expected the trained weights file 'shoplifting_yolov8n_best.pt' "
            f"inside {MODELS_DIR}."
        )
    return YOLO_WEIGHTS


def missing_model_paths() -> list[str]:
    """Return human-readable descriptions of missing assets (empty = all OK)."""
    problems: list[str] = []
    if not MODELS_DIR.is_dir():
        problems.append(f"Models directory missing: {MODELS_DIR}")
        return problems
    try:
        resolve_videomae_dir()
    except FileNotFoundError as exc:
        problems.append(str(exc))
    try:
        resolve_yolo_weights()
    except FileNotFoundError as exc:
        problems.append(str(exc))
    return problems
