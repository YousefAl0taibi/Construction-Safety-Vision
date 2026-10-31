import argparse
from pathlib import Path

import pytest

from scripts.download_saudi_video import (
    is_supported_mp4,
    load_video_source,
    manual_download_instructions,
)
from scripts.run_saudi_video_pilot import validate_args


def test_video_manifest_selects_one_approved_documented_source():
    row = load_video_source(Path("data/saudi/video_sources.csv"), None)

    assert row["id"] == "dvids-923711-airfield-repair"
    assert row["review_status"] == "approved"
    assert row["rights"] == "Public Domain"
    assert row["source_page"].startswith("https://www.dvidshub.net/video/")
    assert row["asset_url"].startswith("https://www.dvidshub.net/download/videofile/")
    assert float(row["clip_duration_seconds"]) == 30.0


def test_mp4_validation_rejects_html_and_truncated_headers():
    valid = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 1024

    assert is_supported_mp4(valid, "video/mp4")
    assert not is_supported_mp4(b"<html>forbidden</html>", "video/mp4")
    assert not is_supported_mp4(valid, "text/html")
    assert not is_supported_mp4(b"\x00\x00\x00\x18ftyp", "video/mp4")


def test_video_pilot_requires_downloaded_source(tmp_path):
    args = argparse.Namespace(
        model=tmp_path / "best.pt",
        source=tmp_path / "clip.mp4",
        confidence=0.35,
        start_seconds=0.0,
        duration_seconds=30.0,
        gif_fps=1.0,
        gif_width=720,
        gif_start_seconds=18.0,
        gif_duration_seconds=10.0,
        threads=4,
    )
    args.model.write_bytes(b"model")

    with pytest.raises(SystemExit, match="Run scripts/download_saudi_video.py first"):
        validate_args(args)


def test_manual_download_message_names_page_resolution_and_destination(tmp_path):
    row = load_video_source(Path("data/saudi/video_sources.csv"), None)
    destination = tmp_path / row["filename"]

    message = manual_download_instructions(row, destination)

    assert row["source_page"] in message
    assert "1280x720" in message
    assert str(destination) in message
