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

## Training in Colab

1. Create a YOLO-format dataset with the classes in `configs/dataset.yaml`.
2. Preserve a genuine test split that is not used for model selection.
3. Upload the ZIP to `MyDrive/construction-safety/dataset.zip`.
4. Open `notebooks/train_colab.ipynb` in Colab and select a GPU runtime.
5. Run all cells. Checkpoints and plots are copied to Google Drive.
6. Download `safety_detector.onnx` into `models/`.

The dataset should look like:

```text
dataset/
├── dataset.yaml
├── images/{train,val,test}/
└── labels/{train,val,test}/
```

Do not commit licensed datasets, private site footage, or large weights. Publish model artifacts in
a versioned GitHub Release and document the dataset license and provenance.

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

