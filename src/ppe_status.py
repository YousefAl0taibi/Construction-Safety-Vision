from __future__ import annotations

try:
    from .geometry import center_in_top
    from .tracker import SimpleTracker
except ImportError:  # Allows: python src/detect.py
    from geometry import center_in_top
    from tracker import SimpleTracker


def tracked_helmet_statuses(
    detections: list[dict[str, object]], tracker: SimpleTracker
) -> list[dict[str, object]]:
    """Associate helmets with tracked people using a small spatial rule."""
    people = [d for d in detections if str(d["className"]).lower() == "person"]
    helmets = [d for d in detections if str(d["className"]).lower() == "helmet"]
    person_boxes = [d["box"] for d in people]
    helmet_boxes = [d["box"] for d in helmets]

    statuses = []
    for track_id, person_box in tracker.update(person_boxes).items():
        wearing = any(center_in_top(helmet_box, person_box, 0.4) for helmet_box in helmet_boxes)
        statuses.append(
            {"trackId": track_id, "box": person_box, "wearingHelmet": wearing}
        )
    return statuses
