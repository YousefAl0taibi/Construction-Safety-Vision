"""Run the detector over the locally downloaded Saudi pilot images."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2

from src.predictor import PPEDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create annotated images and a JSON summary for the Saudi-domain pilot."
    )
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/saudi/sources.csv")
    )
    parser.add_argument(
        "--source", type=Path, default=Path("data/saudi/images/raw")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/saudi-pilot"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument(
        "--statuses",
        nargs="+",
        default=["approved"],
        help="Manifest review_status values to process (default: approved).",
    )
    return parser.parse_args()


def load_manifest_entries(manifest: Path, statuses: set[str]) -> list[dict[str, str]]:
    """Load traceable source rows selected by review status."""
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["review_status"] in statuses]
    filenames = [row["filename"] for row in selected]
    if len(filenames) != len(set(filenames)):
        raise ValueError("manifest filenames must be unique")
    if not selected:
        raise ValueError(f"manifest has no rows with status: {', '.join(sorted(statuses))}")
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.source.is_dir():
        raise SystemExit(f"Image directory not found: {args.source}")
    try:
        entries = load_manifest_entries(args.manifest, set(args.statuses))
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"Could not load manifest {args.manifest}: {error}") from error

    unsupported = [
        row["filename"]
        for row in entries
        if Path(row["filename"]).suffix.lower() not in IMAGE_EXTENSIONS
    ]
    if unsupported:
        raise SystemExit(f"Unsupported image extension in manifest: {', '.join(unsupported)}")
    missing = [
        row["filename"]
        for row in entries
        if not (args.source / row["filename"]).is_file()
    ]
    if missing:
        raise SystemExit(
            "Approved image files are missing; run the downloader first: "
            + ", ".join(missing)
        )

    annotated_dir = args.output / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    detector = PPEDetector(args.model, args.confidence, args.device)
    detector.warmup()

    class_counts: Counter[str] = Counter()
    inference_times: list[float] = []
    records: list[dict[str, object]] = []
    started = time.perf_counter()

    for index, entry in enumerate(entries, start=1):
        source_path = args.source / entry["filename"]
        record: dict[str, object] = {
            "id": entry["id"],
            "sceneGroup": entry["scene_group"],
            "reviewStatus": entry["review_status"],
            "sourcePage": entry["source_page"],
            "source": source_path.as_posix(),
        }
        image = cv2.imread(str(source_path))
        if image is None:
            record.update({"status": "failed", "error": "image could not be decoded"})
            records.append(record)
            print(f"[{index}/{len(entries)}] failed: {source_path.name}")
            continue

        try:
            annotated, detections, inference_ms = detector.predict(image)
            destination = annotated_dir / f"{source_path.stem}-annotated.jpg"
            if not cv2.imwrite(str(destination), annotated):
                raise OSError(f"could not write {destination}")
        except Exception as error:
            record.update({"status": "failed", "error": str(error)})
            records.append(record)
            print(f"[{index}/{len(entries)}] failed: {source_path.name}: {error}")
            continue

        inference_times.append(inference_ms)
        class_counts.update(str(item["className"]) for item in detections)
        height, width = image.shape[:2]
        record.update(
            {
                "status": "ok",
                "annotated": destination.as_posix(),
                "imageSize": {"width": width, "height": height},
                "inferenceMs": round(inference_ms, 1),
                "detectionCount": len(detections),
                "detections": detections,
            }
        )
        records.append(record)
        print(
            f"[{index}/{len(entries)}] {source_path.name}: "
            f"{len(detections)} detection(s), {inference_ms:.1f} ms"
        )

    successful = sum(record["status"] == "ok" for record in records)
    failed = len(records) - successful
    report: dict[str, object] = {
        "schemaVersion": 1,
        "purpose": "qualitative Saudi-domain pilot; not an accuracy benchmark",
        "manifest": args.manifest.as_posix(),
        "statuses": args.statuses,
        "source": args.source.as_posix(),
        "output": args.output.as_posix(),
        "model": args.model.as_posix(),
        "device": args.device,
        "confidence": args.confidence,
        "imageCount": len(entries),
        "sceneGroupCount": len({entry["scene_group"] for entry in entries}),
        "successfulImages": successful,
        "failedImages": failed,
        "detectionCount": sum(class_counts.values()),
        "classCounts": dict(sorted(class_counts.items())),
        "meanInferenceMs": (
            round(statistics.fmean(inference_times), 1) if inference_times else None
        ),
        "elapsedSeconds": round(time.perf_counter() - started, 2),
        "images": records,
    }

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Summary: {summary_path}")
    if failed:
        raise SystemExit(f"{failed} image(s) failed; see {summary_path}")
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
