from pathlib import Path

import pytest
import numpy as np

from src import detect
from src.detect import default_output, ensure_distinct_paths
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


def test_tracker_keeps_id_across_short_detection_gap():
    tracker = SimpleTracker()
    first_id = next(iter(tracker.update([[0, 0, 100, 200]])))

    assert tracker.update([]) == {}
    second_id = next(iter(tracker.update([[5, 0, 105, 200]])))

    assert second_id == first_id


def test_tracker_expires_id_after_max_missed_frames():
    tracker = SimpleTracker(threshold=0.3, max_missed=1)
    first_id = next(iter(tracker.update([[0, 0, 100, 200]])))

    tracker.update([])
    tracker.update([])
    second_id = next(iter(tracker.update([[5, 0, 105, 200]])))

    assert second_id != first_id


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold": 1.1}, "threshold"),
        ({"max_missed": -1}, "max_missed"),
    ],
)
def test_tracker_rejects_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SimpleTracker(**kwargs)


def test_helmet_status_is_associated_with_person_head_area():
    detections = [
        {"className": "Person", "box": [0, 0, 100, 200]},
        {"className": "helmet", "box": [30, 10, 70, 50]},
    ]
    statuses = tracked_helmet_statuses(detections, SimpleTracker())
    assert statuses[0]["helmetStatus"] == "OK"


def test_explicit_no_helmet_and_missing_evidence_have_distinct_states():
    person = {"className": "Person", "box": [0, 0, 100, 200]}
    no_helmet = {
        "className": "no_helmet",
        "confidence": 0.8,
        "box": [30, 10, 70, 50],
    }

    missing = tracked_helmet_statuses([person], SimpleTracker())
    explicit = tracked_helmet_statuses([person, no_helmet], SimpleTracker())

    assert missing[0]["helmetStatus"] == "UNKNOWN"
    assert explicit[0]["helmetStatus"] == "NO_HELMET"


def test_one_helmet_detection_is_not_reused_for_two_people():
    detections = [
        {"className": "Person", "box": [0, 0, 100, 200]},
        {"className": "Person", "box": [20, 0, 120, 200]},
        {"className": "helmet", "confidence": 0.9, "box": [40, 10, 70, 50]},
    ]

    statuses = tracked_helmet_statuses(detections, SimpleTracker())

    assert [status["helmetStatus"] for status in statuses] == ["OK", "UNKNOWN"]


def test_default_output_matches_source_kind():
    assert default_output(Path("site.jpg")) == Path("outputs/result.jpg")
    assert default_output(Path("site.mp4")) == Path("outputs/result.mp4")


def test_source_cannot_also_be_the_output(tmp_path):
    source = tmp_path / "site.mp4"
    with pytest.raises(SystemExit, match="must be different"):
        ensure_distinct_paths(str(source), source)

    ensure_distinct_paths("0", source)


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)
        self.released = False

    def isOpened(self):
        return True

    def get(self, _property):
        return 25

    def read(self):
        return next(self.frames, (False, None))

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self):
        self.released = False

    def isOpened(self):
        return True

    def write(self, _frame):
        return None

    def release(self):
        self.released = True


def install_fake_video_io(monkeypatch, frames):
    capture = FakeCapture(frames)
    writer = FakeWriter()
    monkeypatch.setattr(detect.cv2, "VideoCapture", lambda _source: capture)
    monkeypatch.setattr(detect.cv2, "VideoWriter", lambda *_args: writer)
    monkeypatch.setattr(detect.cv2, "destroyAllWindows", lambda: None)
    return capture, writer


def test_video_with_no_frames_fails_and_releases_resources(monkeypatch, tmp_path):
    capture, writer = install_fake_video_io(monkeypatch, [])

    with pytest.raises(SystemExit, match="produced no frames"):
        detect.detect_video(object(), "site.mp4", tmp_path / "result.mp4", False)

    assert capture.released
    assert writer.released


def test_video_releases_resources_when_inference_fails(monkeypatch, tmp_path):
    frame = np.zeros((25, 25, 3), dtype=np.uint8)
    capture, writer = install_fake_video_io(monkeypatch, [(True, frame)])

    class FailingDetector:
        def predict(self, _frame):
            raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        detect.detect_video(FailingDetector(), "site.mp4", tmp_path / "result.mp4", False)

    assert capture.released
    assert writer.released
