from __future__ import annotations

from pathlib import Path

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

    def predict(self, image: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], float]:
        result = self.model.predict(
            image, conf=self.confidence, device=self.device, verbose=False
        )[0]
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
        return result.plot(), detections, float(result.speed["inference"])
