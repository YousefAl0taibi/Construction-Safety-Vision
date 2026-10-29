from __future__ import annotations

from collections.abc import Sequence

from .types import Detection


def iou(a: Detection, b: Detection) -> float:
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def point_in_polygon(point: tuple[float, float], polygon: Sequence[tuple[int, int]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            edge_x = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < edge_x:
                inside = not inside
        j = i
    return inside


def center_in_region(item: Detection, region: Detection, top_ratio: float = 1.0) -> bool:
    x, y = item.center
    region_bottom = region.y1 + (region.y2 - region.y1) * top_ratio
    return region.x1 <= x <= region.x2 and region.y1 <= y <= region_bottom

