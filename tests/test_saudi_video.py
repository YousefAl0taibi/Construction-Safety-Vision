import argparse
import csv
from pathlib import Path

import pytest

from scripts.run_saudi_video_pilot import validate_args


def test_video_manifest_selects_one_approved_documented_source():
    with Path("data/saudi/video_sources.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == "dvids-923711-airfield-repair"
    assert row["review_status"] == "approved"
    assert row["rights"] == "Public Domain"
    assert row["source_page"].startswith("https://www.dvidshub.net/video/")
    assert row["asset_url"].startswith("https://www.dvidshub.net/download/videofile/")
    assert float(row["clip_duration_seconds"]) == 30.0


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

    with pytest.raises(SystemExit, match="Download the documented clip manually"):
        validate_args(args)
