from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import torch
import ultralytics

from src.predictor import PPEDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark model inference and the complete in-memory image pipeline."
    )
    parser.add_argument("--model", type=Path, default=Path("models/best.pt"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.threads < 1:
        raise SystemExit("warmup must be >= 0; iterations and threads must be >= 1")

    torch.set_num_threads(args.threads)

    image = cv2.imread(str(args.source))
    if image is None:
        raise SystemExit(f"Could not read image: {args.source}")

    detector = PPEDetector(args.model, args.confidence, args.device)
    detector.warmup()  # Initialise the backend before fixing the benchmark thread count.
    torch.set_num_threads(args.threads)
    for _ in range(args.warmup):
        detector.predict(image)

    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    detection_counts: list[int] = []
    for _ in range(args.iterations):
        started = time.perf_counter()
        _, detections, model_ms = detector.predict(image)
        end_to_end_ms.append((time.perf_counter() - started) * 1000)
        inference_ms.append(model_ms)
        detection_counts.append(len(detections))

    mean_inference_ms = statistics.fmean(inference_ms)
    mean_end_to_end_ms = statistics.fmean(end_to_end_ms)
    height, width = image.shape[:2]
    report = {
        "schemaVersion": 1,
        "workload": "repeated single-image inference; end-to-end excludes disk I/O",
        "source": args.source.as_posix(),
        "imageSize": {"width": width, "height": height},
        "model": args.model.as_posix(),
        "device": args.device,
        "confidence": args.confidence,
        "modelInitialisationExcluded": True,
        "warmupIterations": args.warmup,
        "measuredIterations": args.iterations,
        "torchThreads": torch.get_num_threads(),
        "meanDetectionsPerImage": round(statistics.fmean(detection_counts), 1),
        "modelInferenceMsMean": round(mean_inference_ms, 1),
        "modelInferenceFps": round(1000 / mean_inference_ms, 2),
        "endToEndMsMean": round(mean_end_to_end_ms, 1),
        "endToEndFps": round(1000 / mean_end_to_end_ms, 2),
        "environment": {
            "hardware": args.hardware or platform.processor() or platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
    }

    rendered = json.dumps(report, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
