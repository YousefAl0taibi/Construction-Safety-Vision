"""Run a trained YOLO PPE model on an image, video, or webcam."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


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


def detect_image(model: YOLO, source: Path, output: Path, confidence: float, device: str) -> None:
    image = cv2.imread(str(source))
    if image is None:
        raise SystemExit(f"Could not read image: {source}")
    result = model.predict(image, conf=confidence, device=device, verbose=False)[0]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result.plot())
    print(f"Detections: {len(result.boxes)}")
    print(f"Inference: {result.speed['inference']:.1f} ms")
    print(f"Saved: {output}")


def detect_video(
    model: YOLO,
    source: str,
    output: Path,
    confidence: float,
    device: str,
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
    inference_ms: list[float] = []
    started = time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        result = model.predict(frame, conf=confidence, device=device, verbose=False)[0]
        annotated = result.plot()
        writer.write(annotated)
        inference_ms.append(result.speed["inference"])
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
    model = YOLO(model_path)
    source_path = Path(args.source)
    output = Path(args.output)
    if source_path.suffix.lower() in IMAGE_EXTENSIONS:
        detect_image(model, source_path, output, args.confidence, args.device)
    else:
        detect_video(model, args.source, output, args.confidence, args.device, args.show)


if __name__ == "__main__":
    main()
