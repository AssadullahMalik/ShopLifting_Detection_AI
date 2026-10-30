"""
Step 1 tests: centralized model paths + loading of the REAL trained
checkpoints (VideoMAE and YOLO).

Tests skip with an explicit reason when a required model file or dependency
is unavailable -- they never fake a pass and never download models.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import paths  # noqa: E402

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Centralized path resolution
# ---------------------------------------------------------------------------
class TestPaths:
    def test_project_root_resolved_from_module_location(self):
        # Root must contain models/ and requirements.txt regardless of CWD
        assert paths.PROJECT_ROOT == ROOT
        assert (paths.PROJECT_ROOT / "requirements.txt").is_file()
        assert paths.MODELS_DIR.is_dir()

    def test_videomae_dir_points_at_directory_not_file(self):
        # HF loaders need the DIRECTORY containing config.json + weights
        assert paths.VIDEOMAE_DIR.is_dir()
        assert (paths.VIDEOMAE_DIR / "config.json").is_file()

    def test_yolo_weights_is_a_file(self):
        assert paths.YOLO_WEIGHTS.is_file()
        assert paths.YOLO_WEIGHTS.suffix == ".pt"

    def test_resolve_helpers_return_existing_paths(self):
        assert paths.resolve_videomae_dir().is_dir()
        assert paths.resolve_yolo_weights().is_file()

    def test_missing_files_raise_actionable_errors(self, monkeypatch, tmp_path):
        # models/ exists but is empty -> both resolvers must complain with
        # the exact expected path in the message.
        models = tmp_path / "models"
        models.mkdir()
        monkeypatch.setattr(paths, "MODELS_DIR", models)
        monkeypatch.setattr(paths, "VIDEOMAE_DIR", models)
        monkeypatch.setattr(paths, "YOLO_WEIGHTS", models / "shoplifting_yolov8n_best.pt")

        with pytest.raises(FileNotFoundError, match="VideoMAE config.json not found"):
            paths.resolve_videomae_dir()
        with pytest.raises(FileNotFoundError, match="YOLO checkpoint not found"):
            paths.resolve_yolo_weights()
        assert len(paths.missing_model_paths()) == 2

    def test_missing_weights_inside_existing_dir(self, monkeypatch, tmp_path):
        # Directory + config exist but weights are absent -> explicit error
        vmae = tmp_path / "models"
        vmae.mkdir()
        (vmae / "config.json").write_text("{}")
        monkeypatch.setattr(paths, "MODELS_DIR", vmae)
        monkeypatch.setattr(paths, "VIDEOMAE_DIR", vmae)
        monkeypatch.setattr(paths, "YOLO_WEIGHTS", vmae / "nope.pt")

        with pytest.raises(FileNotFoundError, match="No VideoMAE weights found"):
            paths.resolve_videomae_dir()


# ---------------------------------------------------------------------------
# VideoMAE: real checkpoint loading (session-scoped, loaded once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def videomae():
    if not paths.VIDEOMAE_DIR.is_dir():
        pytest.skip(f"VideoMAE model directory missing: {paths.VIDEOMAE_DIR}")
    pytest.importorskip("transformers")
    from src.pipeline.videomae_loader import load_videomae_model_and_processor
    try:
        model, processor = load_videomae_model_and_processor()
    except Exception as exc:  # noqa: BLE001 - surface real load failures
        pytest.fail(f"Trained VideoMAE checkpoint failed to load: {exc}")
    return model, processor


class TestVideoMAELoading:
    def test_config_matches_expected_contract(self, videomae):
        model, _ = videomae
        assert model.config.num_frames == 16
        assert model.config.image_size == 224
        assert dict(model.config.id2label) == {0: "Non-Shoplifting", 1: "Shoplifting"}

    def test_trained_weights_loaded_not_random_init(self, videomae):
        """Checkpoint stores fused q_bias/v_bias (transformers 5.18 export);
        the loader must remap them onto the installed model's query.bias /
        value.bias, byte-for-byte, instead of leaving them re-initialized."""
        model, _ = videomae
        pytest.importorskip("safetensors")
        from safetensors.torch import load_file

        ckpt = load_file(str(paths.VIDEOMAE_DIR / "model.safetensors"))
        sd = model.state_dict()

        q_keys = [k for k in sd if k.endswith("attention.query.bias")]
        v_keys = [k for k in sd if k.endswith("attention.value.bias")]
        assert q_keys and v_keys
        # Restored values must equal the checkpoint's fused biases exactly.
        for qk in q_keys:
            orig = qk[: -len("query.bias")] + "q_bias"
            assert torch.equal(sd[qk], ckpt[orig]), f"{qk} not restored from {orig}"
        for vk in v_keys:
            orig = vk[: -len("value.bias")] + "v_bias"
            assert torch.equal(sd[vk], ckpt[orig]), f"{vk} not restored from {orig}"
        # And they must be real trained values, not zero-initialized.
        assert any(sd[k].abs().sum().item() > 0 for k in q_keys), (
            "All query.bias tensors are zero - trained attention biases were "
            "NOT restored from the checkpoint"
        )

    def test_inference_output_shape_on_16_frame_clip(self, videomae):
        model, _ = videomae
        clip = torch.rand(1, 16, 3, 224, 224)
        with torch.no_grad():
            logits = model(pixel_values=clip).logits
        assert logits.shape == (1, 2)
        assert torch.isfinite(logits).all()

    def test_rejects_wrong_frame_count(self, videomae):
        """Documenting the 16-frame requirement: a 15-frame clip must error,
        not silently produce logits."""
        model, _ = videomae
        bad_clip = torch.rand(1, 15, 3, 224, 224)
        with pytest.raises(Exception):
            with torch.no_grad():
                model(pixel_values=bad_clip)

    def test_processor_produces_expected_shape(self, videomae):
        """The shipped VideoMAEImageProcessor yields (1, 16, 3, 224, 224)."""
        _, processor = videomae
        np = pytest.importorskip("numpy")
        Image = pytest.importorskip("PIL.Image", reason="Pillow required")
        from PIL import Image as PILImage
        frames = [PILImage.fromarray(np.zeros((480, 640, 3), dtype=np.uint8))
                  for _ in range(16)]
        encoded = processor(images=frames, return_tensors="pt")
        assert tuple(encoded["pixel_values"].shape) == (1, 16, 3, 224, 224)


# ---------------------------------------------------------------------------
# YOLO: real checkpoint loading
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def yolo():
    if not paths.YOLO_WEIGHTS.is_file():
        pytest.skip(f"YOLO checkpoint missing: {paths.YOLO_WEIGHTS}")
    ultralytics = pytest.importorskip("ultralytics")
    try:
        model = ultralytics.YOLO(str(paths.YOLO_WEIGHTS), task="detect")
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Trained YOLO checkpoint failed to load: {exc}")
    return model


class TestYOLOLoading:
    def test_class_names_read_from_checkpoint(self, yolo):
        names = {int(k): str(v).strip().lower() for k, v in yolo.names.items()}
        assert names == {0: "shoplifting", 1: "normal"}, (
            f"Unexpected class names in checkpoint: {yolo.names}"
        )

    def test_synthetic_frame_inference(self, yolo):
        import numpy as np
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        results = yolo.predict(frame, verbose=False, device="cpu")
        assert len(results) == 1
        boxes = results[0].boxes
        n = 0 if boxes is None else len(boxes)
        assert n >= 0  # zero detections on a blank frame is acceptable

    def test_detector_default_path_resolves(self, yolo):
        """PersonDetector's module-level default must point at the real file."""
        from src.pipeline.detector import DEFAULT_MODEL_PATH, PersonDetector
        assert Path(DEFAULT_MODEL_PATH).is_file()
        detector = PersonDetector()  # uses centralized default
        assert detector.model is not None
