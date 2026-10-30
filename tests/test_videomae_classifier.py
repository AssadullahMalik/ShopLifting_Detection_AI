"""
Step 3 tests: shared VideoMAE clip-classification integration.

Two kinds of tests live here and are clearly labelled:

* REAL-WEIGHTS / REAL-PROCESSOR integration -> use the trained checkpoint in
  ``models/`` and synthetic clips (real forward passes, no mocks).
* SOURCE-WIRING checks -> assert the dashboard and CLI route VideoMAE through
  the shared classifier (no duplicate loader / manual transform) and preserve
  windowing, smoothing, dedup and YOLO-vs-VideoMAE separation.

No network is required: the loader is local-only. CPU-only is expected in this
environment; tests do not assume a GPU.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.pipeline import paths
from src.pipeline.classifier import VideoMAEClassifier

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "src" / "dashboard" / "video_detection_app.py"
CLI = ROOT / "scripts" / "inference.py"


def _bgr_frames(n, height=240, width=320, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
            for _ in range(n)]


@pytest.fixture(scope="session")
def classifier():
    # REAL trained checkpoint + processor, loaded once for the whole session.
    return VideoMAEClassifier()


# --------------------------------------------------------------------------- #
# Clip preprocessing + inference (REAL weights)
# --------------------------------------------------------------------------- #
class TestClassifierContract:
    def test_loads_trained_checkpoint_contract(self, classifier):
        assert classifier.num_frames == 16
        assert classifier.image_size == 224
        assert classifier.id2label == {0: "Non-Shoplifting", 1: "Shoplifting"}

    def test_model_is_local_and_eval_mode(self, classifier):
        assert classifier.model.training is False
        assert classifier.device.type in {"cpu", "cuda"}

    def test_preprocess_shape_is_16x3x224x224(self, classifier):
        pv = classifier.preprocess_clip(_bgr_frames(16))
        assert tuple(pv.shape) == (1, 16, 3, 224, 224)
        assert pv.device.type == classifier.device.type

    def test_classify_real_weights_two_finite_probs(self, classifier):
        res = classifier.classify_clip(_bgr_frames(16, seed=1))
        p = res["class_probabilities"]
        assert len(p) == 2
        assert all(np.isfinite(p))
        assert p[0] + p[1] == pytest.approx(1.0, abs=1e-5)
        assert len(res["logits"]) == 2

    def test_class_mapping_preserved(self, classifier):
        res = classifier.classify_clip(_bgr_frames(16, seed=2))
        assert res["predicted_class_id"] in (0, 1)
        expected = {0: "Non-Shoplifting", 1: "Shoplifting"}[res["predicted_class_id"]]
        assert res["predicted_label"] == expected
        # shoplifting is ALWAYS class index 1, normal is index 0
        assert res["shoplifting_prob"] == pytest.approx(res["class_probabilities"][1])
        assert res["normal_prob"] == pytest.approx(res["class_probabilities"][0])

    def test_pil_input_accepted(self, classifier):
        from PIL import Image
        pil = [Image.fromarray(f) for f in _bgr_frames(16, seed=3)]
        res = classifier.classify_clip(pil)
        assert res["class_probabilities"][0] + res["class_probabilities"][1] \
            == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("bad_n", [15, 17, 0, 1])
    def test_rejects_wrong_frame_count(self, classifier, bad_n):
        with pytest.raises(ValueError, match="exactly 16 frames"):
            classifier.classify_clip(_bgr_frames(bad_n))

    def test_rejects_malformed_frame_dimensions(self, classifier):
        frames = _bgr_frames(16)
        frames[5] = np.zeros((10, 10), dtype=np.uint8)  # 2D, not (H, W, 3)
        with pytest.raises(ValueError):
            classifier.classify_clip(frames)

    def test_rejects_empty_frame(self, classifier):
        frames = _bgr_frames(16)
        frames[0] = np.zeros((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            classifier.classify_clip(frames)

    def test_rejects_unsupported_frame_type(self, classifier):
        frames = _bgr_frames(16)
        frames[2] = "not-a-frame"
        with pytest.raises(ValueError):
            classifier.classify_clip(frames)

    def test_missing_files_raise_before_any_download(self, classifier, tmp_path, monkeypatch):
        # An empty models dir must fail via the path validator (no HF download),
        # proving loading never falls back to the network.
        empty = tmp_path / "models"
        empty.mkdir()
        monkeypatch.setattr(paths, "VIDEOMAE_DIR", empty)
        monkeypatch.setattr(paths, "MODELS_DIR", empty)
        with pytest.raises(FileNotFoundError, match="config.json"):
            VideoMAEClassifier()  # model_dir=None -> resolves through paths


# --------------------------------------------------------------------------- #
# CLI integration (real weights)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def cli_detector():
    from scripts.inference import ShopliftingDetector
    return ShopliftingDetector()  # REAL checkpoint via shared classifier


class TestCLIIntegration:
    def test_detector_uses_shared_classifier(self, cli_detector):
        assert isinstance(cli_detector.classifier, VideoMAEClassifier)
        assert cli_detector.clip_length == cli_detector.classifier.num_frames == 16

    def test_predict_clip_returns_legacy_dict(self, cli_detector):
        r = cli_detector.predict_clip(_bgr_frames(16, seed=4))
        assert set(r) == {"prediction", "confidence", "shoplifting_prob", "normal_prob"}
        assert r["prediction"] in ("shoplifting", "normal")
        assert 0.0 <= r["confidence"] <= 1.0
        # prediction is consistent with which class probability is larger
        if r["prediction"] == "shoplifting":
            assert r["shoplifting_prob"] >= r["normal_prob"]
        else:
            assert r["normal_prob"] >= r["shoplifting_prob"]

    def test_predict_clip_under_full_clip_returns_none(self, cli_detector):
        # Legacy sliding-window contract: an incomplete clip yields no result
        # (never a silent "normal" classification).
        assert cli_detector.predict_clip(_bgr_frames(15)) is None

    def test_predict_clip_propagates_invalid_frame(self, cli_detector):
        frames = _bgr_frames(16)
        frames[3] = np.zeros((8, 8), dtype=np.uint8)  # malformed: 2D, not HxWx3
        with pytest.raises(ValueError):
            cli_detector.predict_clip(frames)

    def test_cli_source_no_naive_load_or_manual_transform(self):
        src = CLI.read_text()
        assert "VideoMAEClassifier" in src
        assert "VideoMAEForVideoClassification.from_pretrained" not in src
        assert "transforms.Compose" not in src
        assert "self.transform" not in src


# --------------------------------------------------------------------------- #
# Dashboard integration (source wiring -- no streamlit runtime needed)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def dashboard_src():
    return DASHBOARD.read_text()


class TestDashboardIntegration:
    def test_dashboard_imports_shared_classifier(self, dashboard_src):
        assert "from src.pipeline.classifier import VideoMAEClassifier" in dashboard_src
        assert "VideoMAEClassifier()" in dashboard_src

    def test_dashboard_drops_naive_loader_and_manual_transform(self, dashboard_src):
        assert "VideoMAEForVideoClassification.from_pretrained" not in dashboard_src
        assert "transforms.Compose" not in dashboard_src
        assert "TRANSFORM" not in dashboard_src

    def test_dashboard_uses_videomae_class1_probability(self, dashboard_src):
        assert 'classify_clip' in dashboard_src
        assert '["shoplifting_prob"]' in dashboard_src

    def test_dashboard_preserves_windowing_smoothing_dedup(self, dashboard_src):
        for token in ("frame_stride", "window_stride", "recent_probs",
                      "np.mean(recent_probs)", "deduplicate_alerts",
                      "last_smoothed >= threshold"):
            assert token in dashboard_src, f"missing preserved behaviour: {token}"

    def test_dashboard_keeps_yolo_tracker_separate_from_videomae(self, dashboard_src):
        # YOLO/ByteTrack still driven independently of the clip probability.
        assert "tracker.update(detections)" in dashboard_src
        assert "_tracked_to_dicts" in dashboard_src
        assert "tracker_id" in dashboard_src


# --------------------------------------------------------------------------- #
# P1 regression: inference preprocessing must match the TRAINING transform
# (Resize((224,224)) squash), NOT the saved processor's center-crop.
# Uses the REAL processor/model to prove the two differ on 16:9 input.
# --------------------------------------------------------------------------- #
class TestPreprocessingMatchesTraining:
    def _training_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def test_preprocess_equals_training_transform_on_nonsquare(self, classifier):
        from PIL import Image
        # 16:9-ish non-square frames where crop vs squash really diverge.
        # Feed identical PIL input to both sides so this isolates the
        # resize/normalise semantics (the actual P1 concern), independent of
        # the BGR->RGB channel swap the classifier applies to ndarray input.
        frames = _bgr_frames(16, height=480, width=854, seed=11)
        pils = [Image.fromarray(f) for f in frames]
        got = classifier.preprocess_clip(pils)
        assert tuple(got.shape) == (1, 16, 3, 224, 224)

        tf = self._training_transform()
        expected = torch.stack([tf(p) for p in pils]).unsqueeze(0)
        # Byte-for-byte equal to the training pipeline's output.
        assert torch.allclose(got.cpu(), expected, atol=1e-6)

    def test_preprocess_no_longer_matches_center_crop_processor(self, classifier):
        from PIL import Image
        frames = _bgr_frames(16, height=480, width=854, seed=12)
        got = classifier.preprocess_clip(frames).cpu()
        proc = classifier.processor(
            images=[Image.fromarray(f).convert("RGB") for f in frames],
            return_tensors="pt",
        )["pixel_values"]
        # The squash transform must differ from the center-crop processor on
        # non-square input; if this ever matches, the P1 fix regressed.
        assert not torch.allclose(got, proc, atol=1e-3)

    def test_classify_clip_still_valid_after_fix(self, classifier):
        res = classifier.classify_clip(_bgr_frames(16, height=480, width=854, seed=13))
        p = res["class_probabilities"]
        assert len(p) == 2 and all(np.isfinite(p))
        assert p[0] + p[1] == pytest.approx(1.0, abs=1e-5)
        assert res["shoplifting_prob"] == pytest.approx(p[1])
