from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import build_components
from .pipeline import SafetyPipeline


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Construction PPE detection and safety events")
    root.add_argument("--config", default="configs/inference.yaml")
    subcommands = root.add_subparsers(dest="command", required=True)
    image = subcommands.add_parser("image")
    image.add_argument("--source", required=True)
    image.add_argument("--output", default="outputs/result.jpg")
    video = subcommands.add_parser("video")
    video.add_argument("--source", required=True, help="Video path or webcam index")
    video.add_argument("--output", default="outputs/result.mp4")
    video.add_argument("--camera-id", default="camera-1")
    video.add_argument("--display", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    detector, rules = build_components(args.config)
    pipeline = SafetyPipeline(detector, rules=rules)
    print(f"Inference provider: {detector.provider}")
    if args.command == "image":
        frame = cv2.imread(args.source)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.source}")
        result, events = pipeline.process_frame(frame, 0)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.output, result)
        print(f"Saved {args.output}; events={len(events)}")
    else:
        source = int(args.source) if args.source.isdigit() else args.source
        count = pipeline.process_video(source, args.output, args.camera_id, args.display)
        print(f"Saved {args.output}; events={count}")


if __name__ == "__main__":
    main()

