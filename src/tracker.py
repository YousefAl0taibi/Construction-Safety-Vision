from __future__ import annotations

try:
    from .geometry import Box, iou
except ImportError:  # Allows: python src/detect.py
    from geometry import Box, iou


class SimpleTracker:
    """Assign stable IDs between nearby frames using greedy IoU matching."""

    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self.tracks: dict[int, Box] = {}
        self.next_id = 1

    def update(self, boxes: list[Box]) -> dict[int, Box]:
        assigned: dict[int, Box] = {}
        used: set[int] = set()
        for box in boxes:
            best_id, best_score = None, 0.0
            for track_id, previous in self.tracks.items():
                if track_id in used:
                    continue
                score = iou(box, previous)
                if score > best_score:
                    best_id, best_score = track_id, score
            if best_id is None or best_score < self.threshold:
                best_id = self.next_id
                self.next_id += 1
            assigned[best_id] = box
            used.add(best_id)
        self.tracks = assigned
        return assigned
