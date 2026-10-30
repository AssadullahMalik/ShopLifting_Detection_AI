"""
ByteTrack person-behavior tracking for the detection pipeline.

supervision >= 0.30 removed its ByteTrack implementation, so this module
uses the ByteTrack that ships with the installed ultralytics package
(ultralytics.trackers.BYTETracker), driven through supervision.Detections
which supervision 0.31.1 still provides.

Each PersonTracker instance owns its own BYTETracker state, so IDs never
leak between unrelated video streams: create one tracker per stream (or
call reset() between streams).
"""
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Support both `import src.pipeline.tracker` and running this file directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import supervision as sv

from src.pipeline.detector import to_sv_detections

_TRACKER_ARG_DEFAULTS = {
    "track_high_thresh": 0.5,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.6,
    # Generous buffer: the dashboard runs YOLO every N frames, so tracks must
    # coast through frames without detector updates instead of expiring.
    "track_buffer": 120,
    "match_thresh": 0.8,
    "fuse_score": True,
}


class PersonTracker:
    """ByteTrack wrapper preserving class IDs, confidences and stable IDs.

    Despite the historical name, detections come from the trained
    shoplifting/normal BEHAVIOR-region checkpoint (class 0 = Shoplifting,
    class 1 = normal); the tracker follows those regions over time.
    """

    def __init__(
        self,
        frame_rate: int = 30,
        tracker_args: Optional[dict] = None,
    ) -> None:
        from ultralytics.trackers import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace

        merged = dict(_TRACKER_ARG_DEFAULTS)
        if tracker_args:
            merged.update(tracker_args)
        self._args = IterableSimpleNamespace(**merged)
        self._BYTETracker = BYTETracker
        self.tracker = BYTETracker(self._args)
        self.frame_rate = frame_rate

    def reset(self) -> None:
        """Clear all track state. Call when starting a new video stream."""
        self.tracker = self._BYTETracker(self._args)

    def update(self, detections: List[dict]) -> sv.Detections:
        """Advance the tracker one frame with detector-format detections.

        Args:
            detections: list of dicts from ``PersonDetector.detect()``
                (keys: bbox, confidence, class_id, label). An empty list is
                valid and lets existing tracks coast for a while.

        Returns:
            ``sv.Detections`` with ``tracker_id``, ``class_id`` and
            ``confidence`` preserved. Never raises on empty input.
        """
        det_sv = to_sv_detections(detections)
        if len(det_sv) == 0:
            self._update_from_sv(det_sv)  # coast existing tracks
            return sv.Detections.empty()

        tracked = self._update_from_sv(det_sv)
        if tracked is None:
            # Tracker failure must not kill the pipeline: fall back to raw.
            return det_sv
        return tracked

    def _update_from_sv(self, det_sv: sv.Detections):
        """Run BYTETracker on an sv.Detections and re-wrap into sv.Detections.

        Returns None if the underlying tracker raised.
        """
        try:
            boxes = self._to_ultralytics_boxes(det_sv)
            out = self.tracker.update(boxes)  # (N, 8): x1,y1,x2,y2,id,score,cls,idx
        except Exception:
            return None

        if out is None or len(out) == 0:
            return sv.Detections.empty()

        out = np.asarray(out, dtype=float)
        xyxy = out[:, 0:4]
        tracker_id = out[:, 4].astype(int)
        score = out[:, 5]
        cls = out[:, 6].astype(int)

        return sv.Detections(
            xyxy=xyxy,
            confidence=score.astype(float),
            class_id=cls,
            tracker_id=tracker_id,
        )

    @staticmethod
    def _to_ultralytics_boxes(det_sv: sv.Detections):
        """Build an ultralytics Boxes object from sv.Detections.

        Ultralytics 8.4 Boxes row layout is [x1, y1, x2, y2, conf, cls]
        (conf = data[:, -2], cls = data[:, -1]); the detection index is added
        internally by parse_bboxes, so it must NOT be included here.
        """
        from ultralytics.engine.results import Boxes

        n = len(det_sv)
        xyxy = det_sv.xyxy.reshape(n, 4) if n else np.zeros((0, 4))
        cls = (
            det_sv.class_id.astype(float).reshape(n, 1)
            if n and det_sv.class_id is not None
            else np.zeros((n, 1))
        )
        conf = (
            det_sv.confidence.astype(float).reshape(n, 1)
            if n and det_sv.confidence is not None
            else np.ones((n, 1))
        )
        # conf BEFORE cls to match data[:, -2]=conf, data[:, -1]=cls.
        data = np.hstack([xyxy, conf, cls])
        return Boxes(data, orig_shape=(0, 0))


if __name__ == "__main__":
    import cv2

    from src.pipeline.detector import PersonDetector

    detector = PersonDetector(conf_threshold=0.6)
    tracker = PersonTracker()

    # Annotators
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(0)

    print("Testing tracker... press 'q' to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        tracked = tracker.update(detections)

        # Annotate frame
        if len(tracked) > 0:
            labels = [
                f"{detector.class_names.get(int(c), 'obj')} {s:.0%} ID:{tid}"
                for c, s, tid in zip(
                    tracked.class_id, tracked.confidence, tracked.tracker_id
                )
            ]
            annotated = box_annotator.annotate(scene=frame.copy(), detections=tracked)
            annotated = label_annotator.annotate(
                scene=annotated, detections=tracked, labels=labels
            )
        else:
            annotated = frame

        cv2.imshow("Tracking Test", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking test completed!")
