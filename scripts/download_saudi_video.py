"""Download the license-documented Saudi video selected for the motion demo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from urllib.error import HTTPError

try:
    from .download_saudi_sources import request_bytes
except ImportError:  # Allows: python scripts/download_saudi_video.py
    from download_saudi_sources import request_bytes


MAX_VIDEO_BYTES = 80 * 1024 * 1024


def manual_download_instructions(row: dict[str, str], destination: Path) -> str:
    return (
        "DVIDS requires a free signed-in account for video downloads. "
        f"Open {row['source_page']}, sign in, choose the 1280x720 MP4, and save it as "
        f"{destination}. Then run: python -m scripts.run_saudi_video_pilot --device cpu"
    )


def is_supported_mp4(payload: bytes, content_type: str) -> bool:
    """Accept a complete-looking ISO Base Media/MP4 response, not an HTML error page."""
    return (
        len(payload) >= 12
        and payload[4:8] == b"ftyp"
        and content_type in {"video/mp4", "application/mp4", "application/octet-stream"}
    )


def load_video_source(manifest: Path, source_id: str | None) -> dict[str, str]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    approved = [row for row in rows if row["review_status"] == "approved"]
    if source_id:
        approved = [row for row in approved if row["id"] == source_id]
    if len(approved) != 1:
        selection = source_id or "the default approved source"
        raise ValueError(f"expected one row for {selection}, found {len(approved)}")
    return approved[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download one approved, license-documented Saudi video."
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/saudi/video_sources.csv")
    )
    parser.add_argument("--source-id")
    parser.add_argument(
        "--output", type=Path, default=Path("data/saudi/videos/raw")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/saudi/video-download-report.json")
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        row = load_video_source(args.manifest, args.source_id)
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Could not load video manifest {args.manifest}: {error}") from error

    destination = args.output / row["filename"]
    print(f"Source page: {row['source_page']}")
    print(f"Destination: {destination}")
    if args.dry_run:
        return

    args.output.mkdir(parents=True, exist_ok=True)
    payload: bytes
    final_url = row["asset_url"]
    status = "existing"
    if destination.exists() and not args.overwrite:
        payload = destination.read_bytes()
        if not is_supported_mp4(payload, "video/mp4"):
            raise SystemExit(
                f"Existing file is not a valid MP4: {destination}; rerun with --overwrite"
            )
    else:
        try:
            payload, content_type, final_url = request_bytes(
                row["asset_url"], max_bytes=MAX_VIDEO_BYTES
            )
        except HTTPError as error:
            if error.code in {401, 403}:
                raise SystemExit(manual_download_instructions(row, destination)) from error
            raise SystemExit(f"Could not download {row['id']}: {error}") from error
        except Exception as error:
            raise SystemExit(f"Could not download {row['id']}: {error}") from error
        if not is_supported_mp4(payload, content_type):
            raise SystemExit(f"Unexpected video response: {content_type}")
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        status = "downloaded"

    report = {
        "id": row["id"],
        "status": status,
        "file": destination.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "sourcePage": row["source_page"],
        "assetUrl": final_url,
        "rights": row["rights"],
        "rightsUrl": row["rights_url"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{status.title()}: {destination} ({len(payload)} bytes)")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
