from __future__ import annotations

from dataclasses import dataclass

from .geometry import iou
from .types import Detection


@dataclass
class _Track:
    detection: Detection
    missed: int = 0


class IoUTracker:
    """Small dependency-free tracker suitable for an MVP and deterministic tests."""

    def __init__(self, match_threshold: float = 0.3, max_missed: int = 20):
        self.match_threshold = match_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[Detection]) -> list[Detection]:
        candidates = [d for d in detections if d.label == "person"]
        unmatched_tracks = set(self._tracks)
        for detection in candidates:
            matches = [
                (track_id, iou(detection, track.detection))
                for track_id, track in self._tracks.items()
                if track_id in unmatched_tracks
            ]
            track_id, score = max(matches, key=lambda pair: pair[1], default=(None, 0.0))
            if track_id is None or score < self.match_threshold:
                track_id = self._next_id
                self._next_id += 1
            detection.track_id = track_id
            self._tracks[track_id] = _Track(detection)
            unmatched_tracks.discard(track_id)

        for track_id in unmatched_tracks:
            self._tracks[track_id].missed += 1
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.missed <= self.max_missed
        }
        return detections

