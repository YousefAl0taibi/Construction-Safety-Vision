from pathlib import Path

import pytest

from src.detect import default_output
from src.geometry import center_in_top, iou
from src.ppe_status import tracked_helmet_statuses
from src.tracker import SimpleTracker


def test_iou_for_overlapping_boxes():
    assert iou([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)


def test_center_in_top():
    person = [0, 0, 100, 200]
    assert center_in_top([30, 10, 70, 50], person, 0.4)
    assert not center_in_top([30, 120, 70, 170], person, 0.4)


def test_tracker_keeps_id_for_overlapping_person():
    tracker = SimpleTracker(threshold=0.3)
    first_id = next(iter(tracker.update([[0, 0, 100, 200]])))
    second_id = next(iter(tracker.update([[5, 0, 105, 200]])))
    assert second_id == first_id


def test_helmet_status_is_associated_with_person_head_area():
    detections = [
        {"className": "Person", "box": [0, 0, 100, 200]},
        {"className": "helmet", "box": [30, 10, 70, 50]},
    ]
    statuses = tracked_helmet_statuses(detections, SimpleTracker())
    assert statuses[0]["wearingHelmet"] is True


def test_default_output_matches_source_kind():
    assert default_output(Path("site.jpg")) == Path("outputs/result.jpg")
    assert default_output(Path("site.mp4")) == Path("outputs/result.mp4")
