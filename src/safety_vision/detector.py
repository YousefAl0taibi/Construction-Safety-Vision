from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .types import Detection


class OnnxYoloDetector:
    """YOLO ONNX inference with automatic CUDA-to-CPU fallback."""

    def __init__(
        self,
        model_path: str | Path,
        labels: list[str],
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        provider: str = "auto",
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}. Train/export it with notebooks/train_colab.ipynb."
            )
        self.labels = labels
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        available = ort.get_available_providers()
        if provider == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDA requested, but onnxruntime-gpu/CUDAExecutionProvider is unavailable")
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider in {"auto", "cuda"} and "CUDAExecutionProvider" in available
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, image: np.ndarray) -> list[Detection]:
        tensor, scale, pad_x, pad_y = self._preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]
        return self._postprocess(output, image.shape[:2], scale, pad_x, pad_y)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.input_size / width, self.input_size / height)
        resized = cv2.resize(image, (round(width * scale), round(height * scale)))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x = (self.input_size - resized.shape[1]) // 2
        pad_y = (self.input_size - resized.shape[0]) // 2
        canvas[pad_y : pad_y + resized.shape[0], pad_x : pad_x + resized.shape[1]] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
        return np.ascontiguousarray(tensor), scale, pad_x, pad_y

    def _postprocess(
        self,
        raw: np.ndarray,
        image_shape: tuple[int, int],
        scale: float,
        pad_x: int,
        pad_y: int,
    ) -> list[Detection]:
        predictions = np.squeeze(raw)
        if predictions.ndim != 2:
            raise ValueError(f"Unsupported ONNX output shape: {raw.shape}")
        # Standard Ultralytics export: [4 + classes, anchors].
        if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] >= 5:
            predictions = predictions.T

        boxes: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in predictions:
            if len(row) == 6 and row[4] <= 1 and float(row[5]).is_integer():
                x1, y1, x2, y2, score, class_id = row
                x, y, width, height = x1, y1, x2 - x1, y2 - y1
            else:
                class_scores = row[4 : 4 + len(self.labels)]
                class_id = int(np.argmax(class_scores))
                score = float(class_scores[class_id])
                if score < self.confidence_threshold:
                    continue
                cx, cy, width, height = row[:4]
                x, y = cx - width / 2, cy - height / 2
            if float(score) < self.confidence_threshold:
                continue
            boxes.append([float(x), float(y), float(width), float(height)])
            scores.append(float(score))
            class_ids.append(int(class_id))

        selected = cv2.dnn.NMSBoxes(
            boxes, scores, self.confidence_threshold, self.iou_threshold
        )
        height, width = image_shape
        detections: list[Detection] = []
        for index in np.array(selected).reshape(-1):
            x, y, box_width, box_height = boxes[int(index)]
            x1 = max(0.0, (x - pad_x) / scale)
            y1 = max(0.0, (y - pad_y) / scale)
            x2 = min(float(width), (x + box_width - pad_x) / scale)
            y2 = min(float(height), (y + box_height - pad_y) / scale)
            class_id = class_ids[int(index)]
            label = self.labels[class_id] if class_id < len(self.labels) else str(class_id)
            detections.append(
                Detection(x1, y1, x2, y2, scores[int(index)], class_id, label)
            )
        return detections


def load_labels(path: str | Path) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
