from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from safety_vision.detector import OnnxYoloDetector, load_labels


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible end-to-end ONNX latency benchmark")
    parser.add_argument("--model", default="models/safety_detector.onnx")
    parser.add_argument("--labels", default="configs/labels.txt")
    parser.add_argument("--image")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=200)
    parser.add_argument("--output", default="reports/benchmark.json")
    args = parser.parse_args()

    detector = OnnxYoloDetector(args.model, load_labels(args.labels), provider=args.device)
    image = cv2.imread(args.image) if args.image else np.zeros((720, 1280, 3), dtype=np.uint8)
    if image is None:
        raise SystemExit(f"Could not read benchmark image: {args.image}")
    for _ in range(args.warmup):
        detector.detect(image)
    timings = []
    for _ in range(args.runs):
        started = time.perf_counter_ns()
        detector.detect(image)
        timings.append((time.perf_counter_ns() - started) / 1_000_000)

    median_ms = statistics.median(timings)
    report = {
        "model": str(Path(args.model).name),
        "provider": detector.provider,
        "onnxruntime": ort.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "inputImage": {"width": image.shape[1], "height": image.shape[0]},
        "modelInput": detector.input_size,
        "warmupRuns": args.warmup,
        "measuredRuns": args.runs,
        "latencyMs": {
            "median": round(median_ms, 3),
            "p95": round(percentile(timings, 0.95), 3),
            "mean": round(statistics.mean(timings), 3),
        },
        "fpsFromMedianLatency": round(1000 / median_ms, 2),
        "scope": "decode excluded; preprocess + ONNX inference + NMS included; batch=1",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

