from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from .predictor import PPEDetector

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/best.pt"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.35"))
DEVICE = os.getenv("DEVICE", "cpu")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Construction PPE Detector", version="1.0.0")


@lru_cache(maxsize=1)
def get_detector() -> PPEDetector:
    return PPEDetector(MODEL_PATH, CONFIDENCE, DEVICE)


@app.get("/health")
def health() -> JSONResponse:
    ready = MODEL_PATH.exists()
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "model_missing",
            "model": str(MODEL_PATH),
            "device": DEVICE,
        },
    )


async def decode_upload(file: UploadFile) -> np.ndarray:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Image exceeds the configured upload limit")
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "The uploaded image could not be decoded")
    return image


def run_prediction(image: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]], float]:
    try:
        return get_detector().predict(image)
    except FileNotFoundError as error:
        raise HTTPException(503, str(error)) from error


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, object]:
    image = await decode_upload(file)
    _, detections, inference_ms = run_prediction(image)
    return {
        "detections": detections,
        "count": len(detections),
        "inferenceMs": round(inference_ms, 1),
    }


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...)) -> Response:
    image = await decode_upload(file)
    annotated, detections, inference_ms = run_prediction(image)
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(500, "Could not encode prediction result")
    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={
            "X-Detection-Count": str(len(detections)),
            "X-Inference-Ms": f"{inference_ms:.1f}",
        },
    )
