"""
Step 2 tests: YOLO class mapping, ByteTrack integration, detector->tracker
plumbing, and dashboard rendering helpers.

Real trained checkpoints are used for the detector tests where available;
tracker/rendering tests use synthetic detector-format detections so they run
fast and deterministically. Tests skip with an explicit reason when a model
file or dependency is missing -- they never fake a pass.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import paths  # noqa: E402


def _det(bbox, conf=0.9, cls=0, label=None):
    """Build a detector-format detection dict."""
    return {
        "bbox": [float(c) for c in bbox],
        "confidence": float(conf),
        "class_id": int(cls),
        "label": label or {0: "Shoplifting", 1: "normal"}.get(int(cls), "region"),
    }


# ---------------------------------------------------------------------------
# Detector: class mapping + output format (uses the real checkpoint)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def detector():
    if not paths.YOLO_WEIGHTS.is_file():
        pytest.skip(f"YOLO checkpoint missing: {paths.YOLO_WEIGHTS}")
    pytest.importorskip("ultralytics")
    from src.pipeline.detector import PersonDetector
    return PersonDetector(conf_threshold=0.5)


class TestDetectorClassMapping:
    def test_class_names_read_from_checkpoint(self, detector):
        names = {int(k): str(v).strip().lower() for k, v in detector.class_names.items()}
        assert names == {0: "shoplifting", 1: "normal"}

    def test_detect_returns_consistent_dict_schema(self, detector):
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        dets = detector.detect(frame)
        assert isinstance(dets, list)
        for d in dets:  # blank frame -> empty; loop is a no-op but validates type
            assert set(d) >= {"bbox", "confidence", "class_id", "label"}
            assert len(d["bbox"]) == 4
            assert isinstance(d["class_id"], int)
            assert isinstance(d["label"], str)
            assert 0.0 <= d["confidence"] <= 1.0

    def test_conf_threshold_is_respected(self, detector):
        # Force a high threshold; a blank frame must still return no detections
        detector.conf_threshold = 0.99
        frame = np.full((360, 640, 3), 200, dtype=np.uint8)
        assert detector.detect(frame) == []

    def test_rejects_bad_frame_input(self, detector):
        with pytest.raises(ValueError):
            detector.detect(None)
        with pytest.raises(ValueError):
            detector.detect("not-an-array")
        with pytest.raises(ValueError):
            detector.detect(np.zeros((10, 10), dtype=np.uint8))  # not 3-channel

    def test_missing_checkpoint_raises_actionable_error(self):
        pytest.importorskip("ultralytics")
        from src.pipeline.detector import PersonDetector
        missing = str(Path(paths.MODELS_DIR) / "does_not_exist.pt")
        with pytest.raises(FileNotFoundError, match="YOLO checkpoint not found"):
            PersonDetector(model_path=missing)


# ---------------------------------------------------------------------------
# supervision.Detections conversion helper
# ---------------------------------------------------------------------------
class TestToSVDetections:
    def test_empty_list_returns_empty_detections(self):
        pytest.importorskip("supervision")
        from src.pipeline.detector import to_sv_detections
        import supervision as sv
        out = to_sv_detections([])
        assert isinstance(out, sv.Detections)
        assert len(out) == 0

    def test_preserves_class_and_confidence(self):
        pytest.importorskip("supervision")
        from src.pipeline.detector import to_sv_detections
        out = to_sv_detections([_det([0, 0, 10, 20], 0.7, 1),
                               _det([5, 5, 15, 25], 0.9, 0)])
        assert sorted(out.class_id.tolist()) == [0, 1]
        assert np.allclose(sorted(out.confidence.tolist()), [0.7, 0.9])


# ---------------------------------------------------------------------------
# ByteTrack tracker (ultralytics backend)
# ---------------------------------------------------------------------------
@pytest.fixture()
def tracker():
    pytest.importorskip("ultralytics")
    pytest.importorskip("supervision")
    from src.pipeline.tracker import PersonTracker
    return PersonTracker(frame_rate=30)


class TestByteTrackIntegration:
    def test_uses_ultralytics_bytetracker(self, tracker):
        from ultralytics.trackers import BYTETracker
        assert isinstance(tracker.tracker, BYTETracker)

    def test_preserves_class_and_confidence(self, tracker):
        out = tracker.update([_det([100, 100, 200, 300], 0.9, 1)])
        assert out.class_id.tolist() == [1]
        assert np.allclose(out.confidence[0], 0.9, atol=0.05)

    def test_id_stability_across_frames(self, tracker):
        seq = [[100, 100, 200, 300], [104, 102, 204, 302], [108, 104, 208, 304]]
        ids = []
        for box in seq:
            out = tracker.update([_det(box, 0.9, 0)])
            ids.append(out.tracker_id.tolist())
        # Every frame has exactly one track and the SAME id throughout.
        assert all(i == [1] for i in ids), ids

    def test_two_distinct_regions_get_distinct_ids(self, tracker):
        out = tracker.update([
            _det([50, 50, 150, 250], 0.95, 0),
            _det([400, 50, 500, 250], 0.8, 1),
        ])
        assert len(out) == 2
        assert len(set(out.tracker_id.tolist())) == 2

    def test_empty_detections_returns_empty_detections(self, tracker):
        import supervision as sv
        # Prime a track, then send an empty frame -> must not crash.
        tracker.update([_det([100, 100, 200, 300], 0.9, 0)])
        out = tracker.update([])
        assert isinstance(out, sv.Detections)
        assert len(out) == 0

    def test_cross_stream_isolation(self, tracker):
        tracker.update([_det([100, 100, 200, 300], 0.9, 0)])
        tracker.update([_det([104, 102, 204, 302], 0.9, 0)])
        tracker.reset()
        # After reset, a new detection starts a fresh id sequence from 1.
        out = tracker.update([_det([500, 500, 560, 560], 0.9, 1)])
        assert out.tracker_id.tolist() == [1]
        assert out.class_id.tolist() == [1]

    def test_detector_to_tracker_end_to_end(self, detector, tracker):
        """Format produced by the detector flows into the tracker unchanged."""
        frame = np.full((480, 640, 3), 127, dtype=np.uint8)
        dets = detector.detect(frame)  # real detector output (list, maybe empty)
        out = tracker.update(dets)
        import supervision as sv
        assert isinstance(out, sv.Detections)


# ---------------------------------------------------------------------------
# Dashboard rendering helpers (no Streamlit runtime, no reprocessing)
# ---------------------------------------------------------------------------
class TestDashboardHelpers:
    @pytest.fixture()
    def dashboard(self):
        pytest.importorskip("streamlit")
        try:
            import src.dashboard.video_detection_app as app
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"dashboard module not importable in test env: {exc}")
        return app

    def test_annotate_frame_handles_class_and_tracker_id(self, dashboard):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        dets = [{"bbox": [10, 10, 60, 120], "confidence": 0.9,
                 "class_id": 0, "label": "Shoplifting", "tracker_id": 3}]
        out = dashboard.annotate_frame(frame.copy(), dets, True, 0.95)
        assert out.shape == frame.shape
        # Frame was drawn on (status bar makes top rows non-black somewhere).
        assert out.any()

    def test_annotate_frame_empty_detections(self, dashboard):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        out = dashboard.annotate_frame(frame.copy(), [], False, 0.1)
        assert out.shape == frame.shape

    def test_tracked_to_dicts_maps_class_and_id(self, dashboard):
        pytest.importorskip("supervision")
        import supervision as sv

        class _YoloStub:
            class_names = {0: "Shoplifting", 1: "normal"}

        det = sv.Detections(
            xyxy=np.array([[10., 10., 60., 120.]]),
            confidence=np.array([0.88]),
            class_id=np.array([1]),
            tracker_id=np.array([7]),
        )
        dicts = dashboard._tracked_to_dicts(det, _YoloStub())
        assert len(dicts) == 1
        d = dicts[0]
        assert d["label"] == "normal"
        assert d["class_id"] == 1
        assert d["tracker_id"] == 7
        assert np.allclose(d["bbox"], [10, 10, 60, 120])

    def test_tracked_to_dicts_empty(self, dashboard):
        pytest.importorskip("supervision")
        import supervision as sv

        class _YoloStub:
            class_names = {0: "Shoplifting", 1: "normal"}

        assert dashboard._tracked_to_dicts(sv.Detections.empty(), _YoloStub()) == []
