from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    label: str
    track_id: int | None = None

    @property
    def box(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(slots=True)
class SafetyEvent:
    event_type: str
    track_id: int
    confidence: float
    camera_id: str
    frame_index: int
    evidence_path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    occurred_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "eventType": self.event_type,
            "trackId": self.track_id,
            "confidence": round(self.confidence, 4),
            "cameraId": self.camera_id,
            "frameIndex": self.frame_index,
            "evidencePath": self.evidence_path,
            "metadata": self.metadata,
            "occurredAt": self.occurred_at,
        }

