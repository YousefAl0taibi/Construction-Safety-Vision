"""Run a trained YOLO PPE model on an image, video, or webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from .ppe_status import tracked_helmet_statuses
    from .predictor import PPEDetector
    from .tracker import SimpleTracker
except ImportError:  # Allows: python src/detect.py
    from ppe_status import tracked_helmet_statuses
    from predictor import PPEDetector
    from tracker import SimpleTracker


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construction PPE detector")
    parser.add_argument("--model", default="models/best.pt", help="Path to trained .pt model")
    parser.add_argument("--source", required=True, help="Image, video path, or webcam index")
    parser.add_argument("--output", default="outputs/result.mp4", help="Output image or video")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 1, ...")
    parser.add_argument("--show", action="store_true", help="Show frames while processing video")
    return parser.parse_args()


def detect_image(detector: PPEDetector, source: Path, output: Path) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"Could not read image: {source}")
    annotated, detections, inference_ms = detector.predict(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), annotated)
    print(f"Detections: {len(detections)}")
    print(f"Inference: {inference_ms:.1f} ms")
    print(f"Saved: {output}")


def detect_video(
    detector: PPEDetector,
    source: str,
    output: Path,
    show: bool,
) -> None:
    capture_source: int | str = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open video source: {source}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (width, height)
    )

    frame_count = 0
    tracker = SimpleTracker()
    inference_ms: list[float] = []
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        annotated, detections, frame_inference_ms = detector.predict(frame)
        for status in tracked_helmet_statuses(detections, tracker):
            x1, y1, _, _ = (int(value) for value in status["box"])
            wearing = bool(status["wearingHelmet"])
            label = f'#{status["trackId"]} OK' if wearing else f'#{status["trackId"]} NO HELMET'
            color = (0, 180, 0) if wearing else (0, 0, 255)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        writer.write(annotated)
        inference_ms.append(frame_inference_ms)
        frame_count += 1
        if show:
            cv2.imshow("Construction PPE Detector", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    elapsed = time.perf_counter() - started
    capture.release()
    writer.release()
    cv2.destroyAllWindows()
    average_inference = sum(inference_ms) / len(inference_ms) if inference_ms else 0.0
    print(f"Frames: {frame_count}")
    print(f"Average model inference: {average_inference:.1f} ms")
    print(f"End-to-end processing FPS: {frame_count / elapsed:.1f}" if elapsed else "FPS: n/a")
    print(f"Saved: {output}")


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"Model not found: {model_path}. Run notebooks/train_colab.ipynb first."
        )
    detector = PPEDetector(model_path, args.confidence, args.device)
    source_path = Path(args.source)
    output = Path(args.output)
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        detect_image(detector, source_path, output)
    else:
        detect_video(detector, args.source, output, args.show)


if __name__ == "__main__":
    main()
