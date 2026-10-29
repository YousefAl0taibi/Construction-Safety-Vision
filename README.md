# Construction PPE Detector

A junior-friendly computer-vision project that trains a YOLO model on Google Colab and uses OpenCV
to detect construction workers and personal protective equipment in images and videos.

## What the project demonstrates

- Transfer learning with a pretrained YOLO11n model.
- GPU training in Google Colab.
- A proper train/validation/test workflow.
- Evaluation using Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
- Object detection on images, videos, and a webcam.
- Exporting a trained model to ONNX.
- Serving image predictions through a small FastAPI service.
- Packaging the API in a non-root Docker container.

The scope is intentionally small enough to explain clearly in a junior-level interview.

## Dataset

The notebook uses the
[Ultralytics Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/).
It contains 1,416 images:

| Split | Images |
|---|---:|
| Train | 1,132 |
| Validation | 143 |
| Test | 141 |

Its 11 classes cover people, worn PPE, and some missing-PPE labels. The dataset and trained
Ultralytics weights are AGPL-3.0; review the licence before using them in a commercial product.

## Train in Google Colab

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb) in Colab and select:

```text
Runtime → Change runtime type → GPU
```

Then run all cells. The notebook will:

1. install Ultralytics;
2. download the dataset automatically;
3. fine-tune YOLO11n for up to 40 epochs;
4. evaluate the held-out test split;
5. print Precision, Recall, mAP50, and mAP50-95;
6. export `best.pt` and `best.onnx`;
7. download all results as `construction-ppe-results.zip`.

After the first training run, copy the real values here:

| Metric | Result |
|---|---:|
| Precision | Pending training |
| Recall | Pending training |
| mAP@0.5 | Pending training |
| mAP@0.5:0.95 | Pending training |

No performance number is claimed before the notebook produces it.

## Run locally

Create an environment and install the small runtime dependency set:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy the trained checkpoint to:

```text
models/best.pt
```

Detect objects in an image:

```bash
python src/detect.py \
  --model models/best.pt \
  --source samples/site.jpg \
  --output outputs/site-result.jpg
```

Process a video with an NVIDIA GPU:

```bash
python src/detect.py \
  --model models/best.pt \
  --source samples/site.mp4 \
  --output outputs/site-result.mp4 \
  --device 0
```

Use webcam `0` and show the output window:

```bash
python src/detect.py --source 0 --output outputs/webcam.mp4 --show
```

The script reports the number of frames, average model inference time, and end-to-end processing
FPS for videos. These numbers depend on the machine used.

## Deployable demo API

This is a small, deployable demonstration of the trained model—not a production platform. The API
deliberately exposes only three focused endpoints:

```text
GET  /health              service and model readiness
POST /predict             JSON detections for application integration
POST /predict/annotated   annotated JPEG for visual inspection
```

Start it locally:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Try it:

```bash
curl -F "file=@samples/site.jpg" \
  http://localhost:8000/predict
```

Example response:

```json
{
  "detections": [
    {
      "classId": 0,
      "className": "helmet",
      "confidence": 0.91,
      "box": [120.0, 80.0, 210.0, 190.0]
    }
  ],
  "count": 1,
  "inferenceMs": 18.4
}
```

Request the visual result separately:

```bash
curl -F "file=@samples/site.jpg" \
  http://localhost:8000/predict/annotated \
  --output outputs/api-result.jpg
```

Configuration is supplied through environment variables rather than hardcoded deployment values:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Trained checkpoint location |
| `CONFIDENCE` | `0.35` | Detection confidence threshold |
| `DEVICE` | `cpu` | `cpu` or an NVIDIA device such as `0` |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum uploaded image size |

The prediction endpoints accept JPEG, PNG, and WebP images and reject oversized uploads. The
health endpoint returns HTTP `503` when the model is missing, so Docker does not report an unusable
container as healthy.

## Docker

The model is mounted at runtime so it is not baked into the public image:

```bash
docker build -t construction-ppe-detector .
docker run --rm -p 8000:8000 \
  -v "${PWD}/models:/app/models:ro" \
  construction-ppe-detector
```

The container runs as a non-root user and includes a health check. The default image is CPU-based;
GPU container deployment is intentionally left outside this junior project.

## Scope boundary

Included because it demonstrates that the model can leave the notebook:

- one inference contract, with JSON and annotated-image representations;
- input validation and environment-based configuration;
- a health endpoint, three API tests, and a Docker image.

Explicitly out of scope:

- authentication, user accounts, or a database;
- queues, distributed workers, Kubernetes, or autoscaling;
- monitoring dashboards, alerting, or an SLA;
- automatic retraining or model registries.

Those are product/platform concerns and would distract from the computer-vision learning goal.

## Tests

Three small API tests cover readiness, upload validation, and the JSON response contract without
loading a real model:

```bash
pip install -r requirements-dev.txt
pytest -q
```

## Project structure

```text
Construction-Safety-Vision/
├── notebooks/
│   └── train_colab.ipynb
├── src/
│   ├── api.py
│   ├── detect.py
│   └── predictor.py
├── tests/
│   └── test_api.py
├── models/
│   └── README.md
├── samples/
│   └── README.md
├── results/
│   └── README.md
├── outputs/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Limitations

- Accuracy can decrease for distant workers, poor lighting, or unusual camera angles.
- The public dataset may not represent a specific real construction site.
- Predictions are not a replacement for a qualified safety officer.
- A confidence threshold must be validated for the target environment.
