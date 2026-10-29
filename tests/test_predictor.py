import numpy as np

from src import predictor


def test_warmup_runs_only_once_and_uses_configured_device(monkeypatch, tmp_path):
    model_path = tmp_path / "best.pt"
    model_path.write_bytes(b"placeholder")

    class FakeModel:
        def __init__(self):
            self.calls = []

        def predict(self, image, **kwargs):
            self.calls.append((image, kwargs))
            return []

    model = FakeModel()
    monkeypatch.setattr(predictor, "YOLO", lambda _path: model)
    detector = predictor.PPEDetector(model_path, confidence=0.4, device="7")

    detector.warmup()
    detector.warmup()

    assert len(model.calls) == 1
    image, kwargs = model.calls[0]
    assert image.shape == (32, 32, 3)
    assert image.dtype == np.uint8
    assert kwargs == {"conf": 0.4, "device": "7", "verbose": False}
