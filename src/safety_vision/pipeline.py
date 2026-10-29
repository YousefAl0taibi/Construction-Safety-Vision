from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .detector import OnnxYoloDetector
from .rules import SafetyRuleEngine
from .tracker import IoUTracker
from .types import Detection, SafetyEvent


COLORS = {
    "person": (255, 180, 0),
    "helmet": (0, 200, 0),
    "safety_vest": (0, 255, 255),
    "head": (0, 80, 255),
    "excavator": (255, 0, 180),
    "forklift": (255, 0, 180),
}


class SafetyPipeline:
    def __init__(
        self,
        detector: OnnxYoloDetector,
        tracker: IoUTracker | None = None,
        rules: SafetyRuleEngine | None = None,
        events_dir: str | Path = "events",
    ):
        self.detector = detector
        self.tracker = tracker or IoUTracker()
        self.rules = rules or SafetyRuleEngine()
        self.events_dir = Path(events_dir)
        self.events_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        camera_id: str = "camera-1",
        danger_zones: list[list[tuple[int, int]]] | None = None,
    ) -> tuple[np.ndarray, list[SafetyEvent]]:
        detections = self.tracker.update(self.detector.detect(frame))
        events = self.rules.evaluate(detections, frame_index, camera_id, danger_zones)
        annotated = draw_detections(frame.copy(), detections, danger_zones or [])
        for event in events:
            filename = f"{camera_id}_{event.event_type}_{event.track_id}_{frame_index}.jpg"
            path = self.events_dir / filename
            cv2.imwrite(str(path), annotated)
            event.evidence_path = str(path)
            with (self.events_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event.as_dict(), ensure_ascii=False) + "\n")
        return annotated, events

    def process_video(
        self,
        source: str | int,
        output_path: str | Path,
        camera_id: str = "camera-1",
        display: bool = False,
    ) -> int:
        capture = cv2.VideoCapture(source)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        frame_index = 0
        event_count = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, events = self.process_frame(frame, frame_index, camera_id)
            writer.write(annotated)
            event_count += len(events)
            if display:
                cv2.imshow("Construction Safety Vision", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
        capture.release()
        writer.release()
        cv2.destroyAllWindows()
        return event_count


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    danger_zones: list[list[tuple[int, int]]],
) -> np.ndarray:
    for zone in danger_zones:
        cv2.polylines(frame, [np.array(zone, dtype=np.int32)], True, (0, 0, 255), 2)
    for detection in detections:
        color = COLORS.get(detection.label, (220, 220, 220))
        x1, y1, x2, y2 = map(int, detection.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        track = f" #{detection.track_id}" if detection.track_id else ""
        text = f"{detection.label}{track} {detection.confidence:.2f}"
        cv2.putText(frame, text, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return frame

