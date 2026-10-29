import cv2
import numpy as np
from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def test_health_returns_503_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.pt")
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "model_missing"


def test_predict_rejects_non_image_upload():
    response = client.post("/predict", files={"file": ("notes.txt", b"x", "text/plain")})
    assert response.status_code == 415


def test_predict_returns_machine_readable_detections(monkeypatch):
    class FakeDetector:
        def predict(self, image):
            detections = [
                {
                    "classId": 0,
                    "className": "helmet",
                    "confidence": 0.91,
                    "box": [10.0, 20.0, 30.0, 40.0],
                }
            ]
            return image, detections, 12.34

    monkeypatch.setattr(api, "get_detector", lambda: FakeDetector())
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    response = client.post(
        "/predict", files={"file": ("image.jpg", encoded.tobytes(), "image/jpeg")}
    )
    assert response.status_code == 200
    assert response.json() == {
        "detections": [
            {
                "classId": 0,
                "className": "helmet",
                "confidence": 0.91,
                "box": [10.0, 20.0, 30.0, 40.0],
            }
        ],
        "count": 1,
        "inferenceMs": 12.3,
    }
