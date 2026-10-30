"""
Step 1 verification: load both trained checkpoints through the centralized
paths module and run a synthetic-inference smoke test.

This NEVER downloads a replacement model: Hugging Face loaders are called
with local_files_only=True, so a missing/broken checkpoint fails loudly.

Usage:
    python scripts/verify_models.py
"""
import sys
from pathlib import Path

# Allow running from any working directory (repo root, scripts/, dashboard/).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import paths  # noqa: E402


def verify_videomae() -> bool:
    import torch
    import transformers

    from src.pipeline.videomae_loader import load_videomae_model_and_processor

    model_dir = paths.resolve_videomae_dir()
    print(f"[VideoMAE] Model directory : {model_dir}")
    print(f"[VideoMAE] transformers    : {transformers.__version__}")

    try:
        model, processor = load_videomae_model_and_processor(model_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"[VideoMAE] FAIL: {type(exc).__name__}: {exc}")
        return False

    num_frames = int(model.config.num_frames)
    image_size = int(model.config.image_size)
    id2label = dict(model.config.id2label)
    num_classes = int(model.config.num_labels)

    print(f"[VideoMAE] Processor       : {type(processor).__name__}")
    print(f"[VideoMAE] num_frames      : {num_frames}")
    print(f"[VideoMAE] image_size      : {image_size}x{image_size}")
    print(f"[VideoMAE] id2label        : {id2label}")

    ok = True
    if num_frames != 16:
        print(f"[VideoMAE] WARNING: expected 16 frames per clip, config says {num_frames}")
        ok = False
    if num_classes != 2:
        print(f"[VideoMAE] WARNING: expected 2 classes, config says {num_classes}")
        ok = False

    # Synthetic 16-frame RGB clip -> (1, 16, 3, 224, 224)
    frames = [torch.full((3, image_size, image_size), 0.5, dtype=torch.float32)
              for _ in range(num_frames)]
    pixel_values = torch.stack(frames).unsqueeze(0)
    print(f"[VideoMAE] Input tensor    : {tuple(pixel_values.shape)}")

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits

    if tuple(logits.shape) != (1, num_classes):
        print(f"[VideoMAE] FAIL: logits shape {tuple(logits.shape)}, expected (1, {num_classes})")
        ok = False
    elif not torch.isfinite(logits).all():
        print("[VideoMAE] FAIL: logits contain NaN/Inf")
        ok = False
    else:
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(torch.argmax(probs))
        print(f"[VideoMAE] Synthetic inference OK: logits={logits.tolist()[0]}")
        print(f"[VideoMAE] Predicted class (synthetic): {id2label.get(pred_id)}")

    return ok


def verify_yolo() -> bool:
    import numpy as np
    from ultralytics import YOLO

    weights = paths.resolve_yolo_weights()
    print(f"\n[YOLO] Checkpoint          : {weights}")

    model = YOLO(str(weights), task="detect")
    names = model.names
    print(f"[YOLO] Class names         : {names}")
    print(f"[YOLO] Number of classes   : {len(names)}")

    ok = True
    expected = {0: "Shoplifting", 1: "normal"}
    normalized = {int(k): str(v).strip().lower() for k, v in names.items()}
    if normalized != {k: v.lower() for k, v in expected.items()}:
        print(f"[YOLO] WARNING: class names differ from expected {expected} "
              f"(read from checkpoint, continuing)")

    # Synthetic frame (gray 640x480 BGR). Zero detections is fine; we only
    # verify that inference runs end-to-end without a crash.
    frame = np.full((480, 640, 3), 127, dtype=np.uint8)
    results = model.predict(frame, verbose=False, device="cpu")
    boxes = results[0].boxes
    n = 0 if boxes is None else len(boxes)
    print(f"[YOLO] Synthetic inference OK: {n} detection(s) on blank frame "
          f"(0 is expected/acceptable)")
    return ok


def main() -> int:
    print("=" * 60)
    print("  Model Loading Verification (Step 1)")
    print("=" * 60)

    problems = paths.missing_model_paths()
    if problems:
        print("FAIL: required model files are missing:")
        for p in problems:
            print(f"  - {p}")
        return 1

    videomae_ok = verify_videomae()
    yolo_ok = verify_yolo()

    print("\n" + "=" * 60)
    print(f"  VideoMAE : {'PASS' if videomae_ok else 'FAIL'}")
    print(f"  YOLO     : {'PASS' if yolo_ok else 'FAIL'}")
    print("=" * 60)
    return 0 if (videomae_ok and yolo_ok) else 2


if __name__ == "__main__":
    sys.exit(main())
