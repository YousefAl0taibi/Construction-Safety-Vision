"""Run a short Saudi video clip through detection, tracking, and GIF preview."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image
import torch
import ultralytics

from src.detect import annotate_tracking
from src.predictor import PPEDetector
from src.tracker import SimpleTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate video detection and simple tracking on a documented Saudi clip."
    )
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/saudi/videos/raw/dvids-923711-airfield-repair.mp4"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/saudi-video/annotated.mp4")
    )
    parser.add_argument(
        "--gif", type=Path, default=Path("docs/images/saudi/saudi-video-demo.gif")
    )
    parser.add_argument(
        "--summary", type=Path, default=Path("results/saudi-video-benchmark.json")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument("--gif-fps", type=float, default=1.0)
    parser.add_argument("--gif-width", type=int, default=720)
    parser.add_argument("--gif-start-seconds", type=float, default=18.0)
    parser.add_argument("--gif-duration-seconds", type=float, default=10.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model}")
    if not args.source.is_file():
        raise SystemExit(
            f"Video not found: {args.source}. Download the documented clip manually; "
            "see data/saudi/README.md."
        )
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.start_seconds < 0 or args.duration_seconds <= 0:
        raise SystemExit("start must be non-negative and duration must be positive")
    if args.gif_fps <= 0 or args.gif_width < 160:
        raise SystemExit("--gif-fps must be positive and --gif-width at least 160")
    if (
        args.gif_start_seconds < 0
        or args.gif_duration_seconds <= 0
        or args.gif_start_seconds + args.gif_duration_seconds > args.duration_seconds
    ):
        raise SystemExit("GIF window must be positive and fit inside the processed clip")
    if args.threads < 1:
        raise SystemExit("--threads must be at least 1")


def resize_for_gif(frame, width: int) -> Image.Image:
    height = max(1, round(frame.shape[0] * width / frame.shape[1]))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def run(args: argparse.Namespace) -> dict[str, object]:
    validate_args(args)
    capture = cv2.VideoCapture(str(args.source))
    if not capture.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video: {args.source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds * 1000.0)
    frame_limit = max(1, round(args.duration_seconds * source_fps))
    gif_interval = max(1, round(source_fps / args.gif_fps))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise SystemExit(f"Could not open video writer: {args.output}")

    torch.set_num_threads(args.threads)
    detector = PPEDetector(args.model, args.confidence, args.device)
    detector.warmup()
    torch.set_num_threads(args.threads)
    tracker = SimpleTracker(max_missed=max(1, round(source_fps)))
    inference_times: list[float] = []
    gif_frames: list[Image.Image] = []
    status_counts: Counter[str] = Counter()
    track_ids: set[int] = set()
    detection_count = 0
    started = time.perf_counter()
    frame_count = 0
    try:
        while frame_count < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            annotated, detections, inference_ms = detector.predict(frame)
            statuses = annotate_tracking(annotated, detections, tracker)
            detection_count += len(detections)
            inference_times.append(inference_ms)
            status_counts.update(str(status["helmetStatus"]) for status in statuses)
            track_ids.update(int(status["trackId"]) for status in statuses)
            writer.write(annotated)
            relative_seconds = frame_count / source_fps
            in_gif_window = (
                args.gif_start_seconds
                <= relative_seconds
                < args.gif_start_seconds + args.gif_duration_seconds
            )
            if in_gif_window and frame_count % gif_interval == 0:
                gif_frames.append(resize_for_gif(annotated, args.gif_width))
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started
    if frame_count == 0 or not gif_frames:
        raise SystemExit(f"Video produced no frames: {args.source}")

    args.gif.parent.mkdir(parents=True, exist_ok=True)
    gif_duration_ms = max(1, round(1000 / args.gif_fps))
    gif_frames[0].save(
        args.gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=gif_duration_ms,
        loop=0,
        optimize=True,
    )

    summary: dict[str, object] = {
        "schemaVersion": 1,
        "workload": "decoded, inferred, annotated, tracked, and encoded video clip",
        "source": args.source.as_posix(),
        "sourcePage": (
            "https://www.dvidshub.net/video/923711/"
            "2nd-distribution-support-battalion-conduct-airfield-damage-repair-during-native-fury"
        ),
        "model": args.model.as_posix(),
        "device": args.device,
        "confidence": args.confidence,
        "modelInitialisationExcluded": True,
        "torchThreads": torch.get_num_threads(),
        "startSeconds": args.start_seconds,
        "requestedDurationSeconds": args.duration_seconds,
        "processedDurationSeconds": round(frame_count / source_fps, 2),
        "sourceFps": round(source_fps, 3),
        "resolution": {"width": width, "height": height},
        "frames": frame_count,
        "gifFrames": len(gif_frames),
        "gifSampleFps": args.gif_fps,
        "gifStartSeconds": args.gif_start_seconds,
        "gifDurationSeconds": args.gif_duration_seconds,
        "gifCreationExcludedFromTiming": True,
        "detections": detection_count,
        "uniquePersonTrackIds": len(track_ids),
        "helmetStatusFrameCounts": dict(sorted(status_counts.items())),
        "meanModelInferenceMs": round(statistics.fmean(inference_times), 1),
        "endToEndProcessingFps": round(frame_count / elapsed, 2),
        "elapsedSeconds": round(elapsed, 2),
        "videoOutput": args.output.as_posix(),
        "gifOutput": args.gif.as_posix(),
        "environment": {
            "hardware": args.hardware or platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "ultralytics": ultralytics.__version__,
        },
        "notes": (
            "Throughput includes decode, inference, tracking, annotation, and MP4 encoding; "
            "GIF creation is measured after this timer. Track IDs are heuristic, not identities."
        ),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
