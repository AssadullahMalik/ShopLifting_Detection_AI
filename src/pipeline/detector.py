import sys
from pathlib import Path
from typing import List

import numpy as np

# Support both `import src.pipeline.detector` and running this file directly.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ultralytics import YOLO

from src.pipeline import paths

# Trained checkpoint lives in <project root>/models (kept in sync with paths.py)
DEFAULT_MODEL_PATH = str(paths.YOLO_WEIGHTS)

# Fallback ONLY if the checkpoint carries no names metadata. The trained
# checkpoint embeds {0: 'Shoplifting', 1: 'normal'} and is the source of truth.
# NOTE: this is a shoplifting/normal BEHAVIOR-REGION detector. It does NOT
# detect people, bags, or any COCO object.
_FALLBACK_CLASS_NAMES = {0: "Shoplifting", 1: "normal"}


class PersonDetector:
    """YOLO detector for the trained shoplifting-behavior checkpoint.

    Class mapping is read from the checkpoint itself (class 0 = Shoplifting,
    class 1 = normal behavior regions). The class name ``PersonDetector`` is
    kept for backward compatibility with existing imports; the model does not
    detect persons.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.5,
    ) -> None:
        p = Path(model_path)
        if not p.is_file() and not p.is_absolute():
            # Allow bare filenames / Hub model names only when relative
            self.model = YOLO(model_path)
        else:
            if not p.is_file():
                raise FileNotFoundError(
                    f"YOLO checkpoint not found: {p}\n"
                    f"Expected the trained weights inside {paths.MODELS_DIR}."
                )
            self.model = YOLO(str(p))
        self.conf_threshold = conf_threshold
        # Read labels from the checkpoint; fall back only if metadata is absent.
        raw_names = self.model.names or {}
        self.class_names = {
            int(k): str(v) for k, v in (raw_names or _FALLBACK_CLASS_NAMES).items()
        }

    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detect behavior regions (Shoplifting / normal) in a BGR frame.

        Returns a list of dicts with keys:
            bbox       [x1, y1, x2, y2] floats
            confidence float >= conf_threshold
            class_id   int (0 = Shoplifting, 1 = normal, per checkpoint)
            label      str class name read from the checkpoint
        Returns [] for empty/invalid frames or when nothing passes the
        confidence threshold. Raises on genuinely corrupt input.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError(
                f"Expected a (H, W, 3) BGR numpy array, got "
                f"{'None' if frame is None else type(frame).__name__}"
            )
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return []

        results = self.model(frame, verbose=False)[0]

        detections: List[dict] = []
        boxes = results.boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            detections.append(
                {
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": conf,
                    "class_id": cls_id,
                    "label": self.class_names.get(cls_id, f"class_{cls_id}"),
                }
            )
        return detections


def to_sv_detections(detections: List[dict]):
    """Convert detector output to a supervision.Detections object.

    Preserves bbox, confidence and class_id. Returns an empty Detections for
    an empty list so downstream trackers/annotators never see bare lists.
    """
    import supervision as sv

    if not detections:
        return sv.Detections.empty()
    xyxy = np.asarray([d["bbox"] for d in detections], dtype=float).reshape(-1, 4)
    conf = np.asarray([d["confidence"] for d in detections], dtype=float)
    cls = np.asarray([d.get("class_id", 0) for d in detections], dtype=int)
    return sv.Detections(xyxy=xyxy, confidence=conf, class_id=cls)


if __name__ == "__main__":
    import cv2

    detector = PersonDetector(conf_threshold=0.6)

    # Test webcam
    cap = cv2.VideoCapture(0)

    print("Testing Behavior Detector... press q to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        # Draw boxes
        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
            color = (0, 0, 255) if det["class_id"] == 0 else (0, 200, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{det['label']} {det['confidence']:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        cv2.imshow("Behavior Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Behavior Detection Test Complete")
