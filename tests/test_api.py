import cv2
import numpy as np
from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def encoded_test_image():
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_api_version_matches_first_release():
    assert api.app.version == "0.1.0"


def test_health_returns_503_when_model_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "MODEL_PATH", tmp_path / "missing.pt")
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "model_missing"


def test_health_returns_503_when_model_cannot_load(monkeypatch, tmp_path):
    model = tmp_path / "broken.pt"
    model.write_bytes(b"not a model")
    monkeypatch.setattr(api, "MODEL_PATH", model)
    monkeypatch.setattr(api, "get_detector", lambda: (_ for _ in ()).throw(RuntimeError("bad model")))
    response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "model_error"


def test_health_warms_up_the_model(monkeypatch, tmp_path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"placeholder")

    class FakeDetector:
        def __init__(self):
            self.warmup_calls = 0

        def warmup(self):
            self.warmup_calls += 1

    detector = FakeDetector()
    monkeypatch.setattr(api, "MODEL_PATH", model)
    monkeypatch.setattr(api, "get_detector", lambda: detector)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert detector.warmup_calls == 1


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
    response = client.post(
        "/predict", files={"file": ("image.jpg", encoded_test_image(), "image/jpeg")}
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


def test_predict_returns_503_when_model_inference_fails(monkeypatch):
    class FailingDetector:
        def predict(self, _image):
            raise RuntimeError("device unavailable")

    monkeypatch.setattr(api, "get_detector", lambda: FailingDetector())

    response = client.post(
        "/predict", files={"file": ("image.jpg", encoded_test_image(), "image/jpeg")}
    )

    assert response.status_code == 503
    assert "Model inference unavailable" in response.json()["detail"]
