from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from time import monotonic

from .geometry import center_in_region, point_in_polygon
from .types import Detection, SafetyEvent


@dataclass(slots=True)
class RuleConfig:
    window_size: int = 20
    minimum_positive_frames: int = 15
    cooldown_seconds: float = 10.0


class SafetyRuleEngine:
    def __init__(self, config: RuleConfig | None = None):
        self.config = config or RuleConfig()
        self._history: dict[tuple[int, str], deque[bool]] = defaultdict(
            lambda: deque(maxlen=self.config.window_size)
        )
        self._last_event_at: dict[tuple[int, str], float] = {}

    def evaluate(
        self,
        detections: list[Detection],
        frame_index: int,
        camera_id: str = "camera-1",
        danger_zones: list[list[tuple[int, int]]] | None = None,
    ) -> list[SafetyEvent]:
        people = [d for d in detections if d.label == "person" and d.track_id is not None]
        helmets = [d for d in detections if d.label == "helmet"]
        vests = [d for d in detections if d.label == "safety_vest"]
        machines = [d for d in detections if d.label in {"excavator", "forklift"}]
        events: list[SafetyEvent] = []

        for person in people:
            assert person.track_id is not None
            has_helmet = any(center_in_region(item, person, top_ratio=0.4) for item in helmets)
            has_vest = any(center_in_region(item, person, top_ratio=0.75) for item in vests)
            checks = {
                "MISSING_HELMET": not has_helmet,
                "MISSING_SAFETY_VEST": not has_vest,
            }
            if danger_zones:
                checks["DANGER_ZONE_INTRUSION"] = any(
                    point_in_polygon(person.center, zone) for zone in danger_zones
                ) and bool(machines)

            for event_type, positive in checks.items():
                event = self._record(person, event_type, positive, frame_index, camera_id)
                if event:
                    events.append(event)
        return events

    def _record(
        self,
        person: Detection,
        event_type: str,
        positive: bool,
        frame_index: int,
        camera_id: str,
    ) -> SafetyEvent | None:
        assert person.track_id is not None
        key = person.track_id, event_type
        history = self._history[key]
        history.append(positive)
        if len(history) < self.config.window_size:
            return None
        if sum(history) < self.config.minimum_positive_frames:
            return None
        now = monotonic()
        if now - self._last_event_at.get(key, float("-inf")) < self.config.cooldown_seconds:
            return None
        self._last_event_at[key] = now
        return SafetyEvent(
            event_type=event_type,
            track_id=person.track_id,
            confidence=person.confidence,
            camera_id=camera_id,
            frame_index=frame_index,
            metadata={"personBox": list(map(round, person.box))},
        )

