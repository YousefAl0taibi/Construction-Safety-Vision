from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from safety_vision.config import build_components
from safety_vision.pipeline import SafetyPipeline

app = FastAPI(title="Construction Safety Vision API", version="0.1.0")


@lru_cache
def get_pipeline() -> SafetyPipeline:
    config_path = os.getenv("SAFETY_VISION_CONFIG", "configs/inference.yaml")
    detector, rules = build_components(config_path)
    return SafetyPipeline(detector, rules=rules)


@app.get("/health")
def health() -> dict[str, str]:
    try:
        provider = get_pipeline().detector.provider
    except FileNotFoundError:
        return {"status": "degraded", "reason": "model_not_found"}
    return {"status": "ok", "provider": provider}


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as incoming:
        incoming.write(await file.read())
        input_path = Path(incoming.name)
    output_path = input_path.with_name(input_path.stem + "_result.jpg")
    try:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise HTTPException(415, "The uploaded file is not a readable image")
        result, _ = get_pipeline().process_frame(frame, 0, "api")
        cv2.imwrite(str(output_path), result)
        return FileResponse(output_path, media_type="image/jpeg", filename="detection.jpg")
    finally:
        input_path.unlink(missing_ok=True)

