from __future__ import annotations

from typing import Literal

try:
    from .geometry import center_in_top
    from .tracker import SimpleTracker
except ImportError:  # Allows: python src/detect.py
    from geometry import center_in_top
    from tracker import SimpleTracker


HelmetStatus = Literal["OK", "NO_HELMET", "UNKNOWN"]


def tracked_helmet_statuses(
    detections: list[dict[str, object]], tracker: SimpleTracker
) -> list[dict[str, object]]:
    """Associate helmet evidence with tracked people using a small spatial rule."""
    people = [d for d in detections if str(d["className"]).lower() == "person"]
    head_detections = [
        d
        for d in detections
        if str(d["className"]).lower() in {"helmet", "no_helmet"}
    ]
    person_boxes = [d["box"] for d in people]

    statuses = []
    used_head_detections: set[int] = set()
    for track_id, person_box in tracker.update(person_boxes).items():
        candidates = [
            (index, detection)
            for index, detection in enumerate(head_detections)
            if index not in used_head_detections
            and center_in_top(detection["box"], person_box, 0.4)
        ]
        if candidates:
            detection_index, detection = max(
                candidates, key=lambda item: float(item[1].get("confidence", 0.0))
            )
            used_head_detections.add(detection_index)
            helmet_status: HelmetStatus = (
                "OK"
                if str(detection["className"]).lower() == "helmet"
                else "NO_HELMET"
            )
        else:
            helmet_status = "UNKNOWN"
        statuses.append(
            {"trackId": track_id, "box": person_box, "helmetStatus": helmet_status}
        )
    return statuses
