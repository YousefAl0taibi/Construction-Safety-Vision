from __future__ import annotations

try:
    from .geometry import Box, iou
except ImportError:  # Allows: python src/detect.py
    from geometry import Box, iou


class SimpleTracker:
    """Assign stable IDs between nearby frames using greedy IoU matching."""

    def __init__(self, threshold: float = 0.3, max_missed: int = 30):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if max_missed < 0:
            raise ValueError("max_missed must be non-negative")

        self.threshold = threshold
        self.max_missed = max_missed
        self.tracks: dict[int, Box] = {}
        self.missed: dict[int, int] = {}
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
        next_tracks = dict(assigned)
        next_missed = {track_id: 0 for track_id in assigned}

        # Keep unmatched tracks briefly so one detector miss does not
        # immediately assign a new ID when the person reappears.
        for track_id, previous_box in self.tracks.items():
            if track_id in assigned:
                continue
            missed_frames = self.missed.get(track_id, 0) + 1
            if missed_frames <= self.max_missed:
                next_tracks[track_id] = previous_box
                next_missed[track_id] = missed_frames

        self.tracks = next_tracks
        self.missed = next_missed
        return assigned
