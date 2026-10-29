from __future__ import annotations

Box = list[float]


def iou(a: Box, b: Box) -> float:
    """Return intersection over union for two [x1, y1, x2, y2] boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def center_in_top(inner: Box, outer: Box, ratio: float) -> bool:
    """Return whether the inner-box centre lies in the top part of outer."""
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be between 0 and 1")
    cx = (inner[0] + inner[2]) / 2
    cy = (inner[1] + inner[3]) / 2
    limit = outer[1] + (outer[3] - outer[1]) * ratio
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= limit
