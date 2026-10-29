# Construction Safety Vision

A GPU-ready computer-vision system that detects PPE, tracks workers across video frames, and
produces auditable safety events instead of treating every frame as an independent alert.

## What makes it more than an object-detection demo

- Custom YOLO training notebook for Google Colab GPU.
- ONNX deployment with automatic CUDA/CPU provider selection.
- OpenCV image, video, and webcam processing.
- Persistent worker IDs through a deterministic IoU tracker.
- Temporal confirmation: a violation must persist across a configurable frame window.
- PPE-to-person association for helmets and safety vests.
- Dangerous-zone intrusion rules when heavy machinery is present.
- Evidence images and JSONL event records.
- FastAPI image endpoint and a testable Python rule engine.

## Current performance status

The first-party checkpoint has not been trained yet, so the repository deliberately publishes no
fabricated performance claims. The Colab notebook now produces the required evidence in one run:

| Metric | Held-out test result |
|---|---:|
| Precision | Pending first Colab run |
| Recall | Pending first Colab run |
| mAP@0.50 | Pending first Colab run |
| mAP@0.50:0.95 | Pending first Colab run |
| GPU median / p95 latency | Pending first Colab run |
| GPU FPS (batch 1) | Pending first Colab run |

Numbers are promoted into this table only with committed `reports/metrics.json` and
`reports/benchmark.json` evidence.

## Architecture

```text
Camera / image / video
        │
        ▼
 YOLO ONNX detector ──► person, helmet, vest, machinery
        │
        ▼
 Worker tracking ──► stable track_id
        │
        ▼
 Temporal safety rules ──► confirmed violations only
        │
        ├──► annotated video
        ├──► evidence JPEG
        └──► events/events.jsonl
```

## Quick start

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -e .
```

For NVIDIA GPU inference, replace CPU ONNX Runtime with the GPU package:

```bash
pip uninstall -y onnxruntime
pip install -r requirements-gpu.txt
```

Train and export the model using `notebooks/train_colab.ipynb`, then copy the exported file to:

```text
models/safety_detector.onnx
```

Run inference:

```bash
safety-vision image --source samples/site.jpg --output outputs/result.jpg
safety-vision video --source samples/site.mp4 --output outputs/result.mp4
safety-vision video --source 0 --output outputs/webcam.mp4 --display
```

The CLI prints the active provider. GPU inference must show:

```text
Inference provider: CUDAExecutionProvider
```

If it shows `CPUExecutionProvider`, check the NVIDIA driver, CUDA compatibility, and the installed
`onnxruntime-gpu` package. The code deliberately does not claim GPU use merely because a GPU exists.

## API

```bash
uvicorn app.api:app --reload
curl -F "file=@samples/site.jpg" http://localhost:8000/detect/image --output result.jpg
```

`GET /health` returns the actual ONNX execution provider. Before the model is installed it returns
a degraded status rather than crashing the service.

## Dataset and training in Colab

The reproducible baseline uses
[Atharion PPE v1.0-public](https://github.com/atharion-team/ppe-dataset-v1/releases/tag/v1.0):
1,130 images and 5,355 boxes across `Person`, `Hardhat`, and `Safety Vest`. Its scene-group split is
791 train, 170 validation, and 169 untouched test images. The notebook downloads the release and
verifies SHA-256 `f5f6f1966c4bdfff4151ee68fa0da21f446e1b788d15280e5f70028c69ec88c5`.

1. Open `notebooks/train_colab.ipynb` directly from this repository in Colab.
2. Select a GPU runtime and run all cells.
3. Training uses validation for model selection and evaluates the test split once afterward.
4. Colab writes Precision, Recall, mAP50, mAP50-95, per-class AP, curves, and confusion matrix.
5. It exports ONNX, verifies CUDA inference, and records median/p95 latency and FPS.
6. All release artifacts are copied to `MyDrive/construction-safety/release-v0.1.0`.

The dataset contains mixed AGPL-3.0 and CC BY 4.0 sources. Preserve its `NOTICE.md` and
`attribution.csv`. The resulting Ultralytics checkpoint and ONNX export are released under
AGPL-3.0 unless another applicable licence has been obtained. Do not commit the dataset or private
site footage.

## Benchmarking

After training, benchmark on the actual deployment machine:

```bash
python scripts/benchmark.py \
  --model models/safety_detector.onnx \
  --image samples/site.jpg \
  --device cuda \
  --warmup 20 \
  --runs 200
```

The report includes hardware/runtime context and measures preprocessing, ONNX inference, and NMS
at batch size 1. Video decoding is intentionally excluded.

## Model release gate

Create release `v0.1.0` only when all of these exist:

- `safety_detector.onnx` and `best.pt`;
- `metrics.json` from the held-out 169-image test split;
- Precision/Recall and PR/F1 curves plus confusion matrix;
- `benchmark.json` from a named deployment device;
- successful ONNX smoke test and CI run.

## Safety-event semantics

Helmet detections must have their center inside the top 40% of a tracked person's box. Vest
detections must be inside the upper 75%. By default, an issue must appear in at least 15 of the last
20 frames before it becomes an event. Each worker/event pair then has a cooldown to prevent alert
flooding. These are explainable heuristics and should be calibrated against the target camera angle.

This project is a decision-support prototype. It must not be represented as a certified occupational
safety system, and human review remains required.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
ruff check .
```

The tests cover geometric zones, identity continuity, temporal confirmation, PPE association, and
the machinery requirement for dangerous-zone events.

## Roadmap

- Replace the baseline IoU tracker with ByteTrack for crowded or occluded scenes.
- Store events in PostgreSQL and add a review dashboard.
- Add camera-specific polygons through a UI.
- Add model/dataset version fields to every event.
- Quantize ONNX/TensorRT models and publish latency benchmarks.
- Add an evaluation set for the complete event pipeline, not only detector mAP.
