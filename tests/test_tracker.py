from safety_vision.tracker import IoUTracker
from safety_vision.types import Detection


def person(x1: float) -> Detection:
    return Detection(x1, 10, x1 + 100, 210, 0.9, 0, "person")


def test_tracker_keeps_identity_for_overlapping_box():
    tracker = IoUTracker()
    first = tracker.update([person(10)])[0]
    second = tracker.update([person(15)])[0]
    assert first.track_id == second.track_id


def test_tracker_creates_new_identity_for_distant_box():
    tracker = IoUTracker()
    first = tracker.update([person(10)])[0]
    second = tracker.update([person(500)])[0]
    assert first.track_id != second.track_id

