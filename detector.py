"""
detector.py
-----------
YOLOv8-based object detection module.
Wraps Ultralytics YOLO to produce bounding boxes in the format
expected by the SORT tracker: [x1, y1, x2, y2, confidence, class_id].
"""

import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """
    Wraps YOLOv8 for frame-level object detection.

    Args:
        model_path     : Path to YOLO weights file (e.g. 'yolov8n.pt').
                         If the file does not exist locally Ultralytics will
                         download it automatically on the first run.
        conf_threshold : Minimum confidence to keep a detection (0.0–1.0).
        iou_threshold  : NMS IOU threshold passed to YOLO (0.0–1.0).
        target_classes : Optional list of class *names* to keep
                         (e.g. ['person', 'car']).  None = keep all classes.
        device         : Inference device – 'cpu', 'cuda', 'mps', etc.
                         None = Ultralytics auto-selects.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.40,
        target_classes: list | None = None,
        device: str | None = None,
    ):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # Load YOLOv8 model (downloads automatically if not present)
        self.model = YOLO(model_path)
        if device:
            self.model.to(device)

        # COCO class names dictionary  {0: 'person', 1: 'bicycle', ...}
        self.class_names: dict[int, str] = self.model.names

        # Build the set of class IDs we care about (None = all classes)
        self.target_class_ids: set[int] | None = None
        if target_classes is not None:
            self.target_class_ids = {
                cid
                for cid, name in self.class_names.items()
                if name in target_classes
            }
            if not self.target_class_ids:
                raise ValueError(
                    f"None of {target_classes!r} were found in the model's "
                    f"class list: {list(self.class_names.values())}"
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Run detection on a single BGR frame (as returned by OpenCV).

        Returns
        -------
        np.ndarray of shape (N, 6) where each row is:
            [x1, y1, x2, y2, confidence, class_id]
        Returns an empty array with shape (0, 6) when nothing is detected.
        """
        # Run inference (verbose=False suppresses per-frame console output)
        results = self.model(
        frame,
        conf=self.conf_threshold,
        iou=self.iou_threshold,
        imgsz=480, 
        agnostic_nms=True,
        half=True, # ← Detects far more objects
        verbose=False,
        )[0]

        detections = []

        if results.boxes is None or len(results.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            # Filter by target classes if specified
            if (
                self.target_class_ids is not None
                and class_id not in self.target_class_ids
            ):
                continue

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            detections.append([x1, y1, x2, y2, confidence, class_id])

        if not detections:
            return np.empty((0, 6), dtype=np.float32)

        return np.array(detections, dtype=np.float32)

    def get_class_name(self, class_id: int) -> str:
        """Return the human-readable name for a COCO class ID."""
        return self.class_names.get(int(class_id), f"cls_{class_id}")
