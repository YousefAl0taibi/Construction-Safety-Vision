from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
from ultralytics import YOLO


class PPEDetector:
    """Small shared wrapper used by both the CLI and HTTP API."""

    def __init__(self, model_path: str | Path, confidence: float = 0.35, device: str = "cpu"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model = YOLO(path)
        self.confidence = confidence
        self.device = device
        self._inference_lock = Lock()
        self._warmed_up = False

    def warmup(self) -> None:
        """Run one tiny inference to validate the model and configured device."""
        if self._warmed_up:
            return
        with self._inference_lock:
            if self._warmed_up:
                return
            sample = np.zeros((32, 32, 3), dtype=np.uint8)
            self.model.predict(
                sample, conf=self.confidence, device=self.device, verbose=False
            )
            self._warmed_up = True

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], float]:
        with self._inference_lock:
            result = self.model.predict(
                image, conf=self.confidence, device=self.device, verbose=False
            )[0]
            self._warmed_up = True
        detections: list[dict[str, object]] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            detections.append(
                {
                    "classId": class_id,
                    "className": result.names[class_id],
                    "confidence": round(float(box.conf.item()), 4),
                    "box": [round(float(value), 1) for value in box.xyxy[0].tolist()],
                }
            )
        annotated = result.plot(line_width=2, font_size=12)
        return annotated, detections, float(result.speed["inference"])
