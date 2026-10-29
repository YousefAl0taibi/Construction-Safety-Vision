from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    model = YOLO(args.weights)
    exported = model.export(format="onnx", imgsz=args.imgsz, opset=args.opset, simplify=True)
    print(f"Exported: {Path(exported).resolve()}")


if __name__ == "__main__":
    main()

