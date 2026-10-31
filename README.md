# Construction PPE Detector

A junior-friendly computer-vision project that trains a YOLO model on Google Colab and uses OpenCV
to detect construction workers and personal protective equipment in images and videos.

## Demo

![Construction PPE detection demo](docs/images/demo.gif)

This before/after GIF was produced by the trained model at confidence `0.35`. The included
[`samples/construction-site-synthetic.png`](samples/construction-site-synthetic.png) input is
AI-generated so the project can be tried immediately without publishing private site footage. It is
not part of the dataset or test split and is not presented as evidence of model accuracy.

## What the project demonstrates

- Transfer learning with a pretrained YOLO11n model.
- GPU training in Google Colab.
- A proper train/validation/test workflow.
- Evaluation using Precision, Recall, mAP@0.5, and mAP@0.5:0.95.
- Object detection on images, videos, and a webcam.
- Simple IoU-based person tracking with helmet-status labels in video.
- Exporting a trained model to ONNX.
- Serving image predictions through a small FastAPI service.
- Packaging the API in a non-root Docker container.
- Curating a license-documented Saudi-domain pilot without mixing it into the original benchmark.

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

### Saudi-domain pilot

[`data/saudi/sources.csv`](data/saudi/sources.csv) records 21 worksite and safety-image sources with
reported Saudi capture locations. After visual review, 15 are approved for the pilot, five are
excluded, and one KAUST source is pending review. The original pages and reuse terms are recorded
before download. Run the manifest-driven downloader with:

```text
python scripts/download_saudi_sources.py --dry-run
python scripts/download_saudi_sources.py
```

Then run the model on the approved images only:

```text
python -m scripts.run_saudi_pilot --device cpu
```

The raw images and full output remain outside Git history. Because most frames come from a few
related galleries, this pilot is for qualitative domain inspection—not a representative Saudi
benchmark and not evidence of accuracy. See [`data/saudi/README.md`](data/saudi/README.md) for the
review, attribution, privacy, and leakage rules. Random Google Images are intentionally excluded
unless their original page supplies explicit reusable rights.

#### Qualitative examples

Both examples below were captured at Prince Sultan Air Base in Saudi Arabia and run through
`best.pt` at confidence `0.35`. In the first image, the model finds four people and four vests but
only two helmets. This is useful evidence that small PPE remains difficult; it is not an accuracy
measurement.

![Predictions for workers on a lift at Prince Sultan Air Base](docs/images/saudi/dvids-7926006-prediction.jpg)

In the second image, the heavily occluded worker inside the manhole receives no detection at all.
Keeping this failure case visible makes the domain limitation clearer than showing only successful
outputs.

**Model output: 0 detections at confidence `0.35`.**

![Missed worker in a confined-space inspection](docs/images/saudi/dvids-7930293-prediction.jpg)

The two source photographs are marked Public Domain on DVIDS; creator, source, modification, and
reuse notes are recorded in [`data/saudi/ATTRIBUTION.md`](data/saudi/ATTRIBUTION.md).

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
6. count training instances per class;
7. measure fixed confidence operating points at `0.20`, `0.35`, and `0.50`;
8. render six held-out prediction examples;
9. export `best.pt` and `best.onnx`;
10. download the models, metrics, plots, class counts, threshold study, and examples as
   `construction-ppe-results.zip`.

To evaluate the exact checkpoint already used by this repository without retraining, change
`USE_EXISTING_CHECKPOINT` to `True`, run all cells, and upload `models/best.pt` when Colab asks. The
same test metrics, confidence-threshold study, training-label counts, examples, and exports will be
generated against that checkpoint.

The baseline was trained for up to 40 epochs with YOLO11n and evaluated on the dataset's held-out
141-image test split:

| Metric | Result |
|---|---:|
| Precision | 0.6584 |
| Recall | 0.5312 |
| mAP@0.5 | 0.5572 |
| mAP@0.5:0.95 | 0.2787 |

These values come from [`results/metrics.json`](results/metrics.json), generated by the training
notebook rather than copied from another model. They establish an honest baseline; the error counts
below show that recall and foreground/background separation need more work.

![Test-set confusion matrix](results/confusion_matrix.png)

### Error analysis

Inter-class confusion is rare: only 8 errors occur between two foreground classes. Most errors are
missed ground-truth objects (351) or predictions with no matched object (225 background false
positives). The model misses 38 `Person` instances, while `no_boots` has 0 correct matches and all 23
ground-truth instances are assigned to background. This points to detection recall and localisation
as larger weaknesses than class separation. The matrix alone does not prove that small object size
is the cause; that would require stratifying results by bounding-box area and inspecting occlusion
and lighting. See [`results/README.md`](results/README.md) for the count definitions.

## Run locally

Create an environment and install the runtime dependencies.

PowerShell on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Bash on Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Download the trained checkpoint from the
[GitHub Releases page](https://github.com/YousefAl0taibi/Construction-Safety-Vision/releases)
and place it at:

```text
models/best.pt
```

Detect objects in an image:

```text
python src/detect.py --model models/best.pt --source samples/construction-site-synthetic.png --output outputs/site-result.jpg
```

To process your own video, first place an authorised clip at `samples/site.mp4`, then run:

```text
python src/detect.py --model models/best.pt --source samples/site.mp4 --output outputs/site-result.mp4 --device 0
```

Use webcam `0` and show the output window:

```text
python src/detect.py --source 0 --output outputs/webcam.mp4 --show
```

For videos and webcam input, the script gives each visible person a temporary tracking ID and adds
one of three heuristic labels: `OK` when helmet evidence is associated with the person's head area,
`NO HELMET` when explicit `no_helmet` evidence is associated with it, or `UNKNOWN` when the model
does not provide clear evidence either way. This is intentionally a simple IoU and spatial-rule
demo, not a safety-certified tracking system. A track is retained internally for up to 30 missed
frames so a brief detector gap does not immediately renumber the person; retained tracks are not
reported as current detections. The script also reports the number of frames, average model
inference time, and end-to-end processing FPS. These numbers depend on the machine used.

### Reproducible Saudi video check

[`data/saudi/video_sources.csv`](data/saudi/video_sources.csv) records a 66-second DVIDS B-roll clip
of airfield repair at Prince Sultan Air Base. The source page marks it Public Domain. The raw MP4
remains ignored; the pipeline processes the first 30 seconds and creates an annotated MP4, a small
README GIF, and a separate video-throughput JSON report:

![Saudi airfield-repair video detection and tracking demo](docs/images/saudi/saudi-video-demo.gif)

The committed run processed 720 frames (30 seconds at 1280×720 and 24 source FPS) on an AMD Ryzen 5
3500U CPU. Mean model inference was `181.4 ms` per frame and the complete decode → inference →
tracking → annotation → MP4-encoding pipeline achieved `4.60 FPS`. It is therefore an offline demo,
not real-time processing of this 24 FPS source. The model produced 1,614 detections and 20 temporary
person track IDs. Those IDs may include fragmented tracks and must not be interpreted as 20 unique
people. Person-frame helmet states were 280 `OK`, 671 `UNKNOWN`, and no explicit `NO_HELMET`;
these are per-frame heuristic outputs, not safety events or accuracy measurements. Full metadata is
in [`results/saudi-video-benchmark.json`](results/saudi-video-benchmark.json).

```text
python scripts/download_saudi_video.py
python -m scripts.run_saudi_video_pilot --device cpu
```

Video downloads on DVIDS require a free signed-in account. If the downloader reports `403`, open
the recorded source page, sign in, choose the 1280×720 MP4, and save it as
`data/saudi/videos/raw/dvids-923711-airfield-repair.mp4`; then run the second command above.

The raw and annotated MP4 files remain outside Git history. For a concise visual preview, the 1 FPS
GIF shows seconds 18–28 where people and tracking labels are visible. The FPS figure still comes
from all 720 processed frames, not from this selected display window.

### Reproducible CPU benchmark

The committed benchmark repeats the included 1536×1024 demonstration image after five warm-up
runs and measures 100 iterations on an AMD Ryzen 5 3500U CPU using four Torch threads:

| Measurement | Mean latency | Throughput |
|---|---:|---:|
| Model inference | 235.7 ms | 4.24 FPS |
| End-to-end in-memory image pipeline | 263.5 ms | 3.79 FPS |

The end-to-end number includes preprocessing, inference, conversion of detections, and annotation;
it excludes disk I/O and video encoding. For a comparable input size, encoded-video throughput is
expected to be lower and is not directly measured here. Recorded run metadata is stored in
[`results/benchmark.json`](results/benchmark.json), and the benchmark can be repeated with:

```text
python -m scripts.benchmark --model models/best.pt --source samples/construction-site-synthetic.png --device cpu --confidence 0.35 --output results/benchmark.json
```

## Deployable demo API

This is a small, deployable demonstration of the trained model—not a production platform. The API
deliberately exposes only three focused endpoints:

```text
GET  /health              service and model readiness
POST /predict             JSON detections for application integration
POST /predict/annotated   annotated JPEG for visual inspection
```

Start it locally:

```text
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Try it:

PowerShell on Windows:

```powershell
curl.exe -F "file=@samples/construction-site-synthetic.png" http://localhost:8000/predict
```

Bash on Linux or macOS:

```bash
curl -F "file=@samples/construction-site-synthetic.png" http://localhost:8000/predict
```

Illustrative response shape (not the complete response for the sample above):

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

PowerShell on Windows:

```powershell
curl.exe -F "file=@samples/construction-site-synthetic.png" http://localhost:8000/predict/annotated --output outputs/api-result.jpg
```

Bash on Linux or macOS:

```bash
curl -F "file=@samples/construction-site-synthetic.png" http://localhost:8000/predict/annotated --output outputs/api-result.jpg
```

Configuration is supplied through environment variables rather than hardcoded deployment values:

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/best.pt` | Trained checkpoint location |
| `CONFIDENCE` | `0.35` | Detection confidence threshold |
| `DEVICE` | `cpu` | `cpu` or an NVIDIA device such as `0` |
| `MAX_UPLOAD_BYTES` | `10485760` | Maximum uploaded image size |

The prediction endpoints accept JPEG, PNG, and WebP images and reject oversized uploads. The
health endpoint loads the model once and returns HTTP `503` when it is missing or cannot be loaded,
so Docker does not report an unusable container as healthy.

## Docker

The model is mounted at runtime so it is not baked into the public image:

The following commands are each a single line and work in both PowerShell and Bash:

```text
docker build -t construction-ppe-detector .
docker run --rm -p 8000:8000 -v "${PWD}/models:/app/models:ro" construction-ppe-detector
```

The container runs as a non-root user and includes a health check. The default image is CPU-based;
GPU container deployment is intentionally left outside this junior project.

## Scope boundary

Included because it demonstrates that the model can leave the notebook:

- one inference contract, with JSON and annotated-image representations;
- input validation and environment-based configuration;
- a model-loading health endpoint, focused API and tracking tests, and a Docker image.

Explicitly out of scope:

- authentication, user accounts, or a database;
- queues, distributed workers, Kubernetes, or autoscaling;
- monitoring dashboards, alerting, or an SLA;
- automatic retraining or model registries.

Those are product/platform concerns and would distract from the computer-vision learning goal.

## Tests

Focused tests cover model warmup, readiness, upload validation, the JSON response contract,
geometry, tracking, helmet association, output-path safety, video resource cleanup, and validation
of the Saudi image/video manifests and downloader helpers:

```text
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Key project structure (abridged)

```text
Construction-Safety-Vision/
├── notebooks/
│   └── train_colab.ipynb
├── src/
│   ├── api.py
│   ├── detect.py
│   ├── geometry.py
│   ├── ppe_status.py
│   ├── tracker.py
│   └── predictor.py
├── tests/
│   ├── test_api.py
│   ├── test_predictor.py
│   ├── test_saudi_sources.py
│   ├── test_saudi_video.py
│   └── test_tracking.py
├── data/saudi/
│   ├── ATTRIBUTION.md
│   ├── README.md
│   ├── sources.csv
│   └── video_sources.csv
├── models/
│   └── README.md
├── samples/
│   ├── construction-site-synthetic.png
│   └── README.md
├── docs/images/
│   ├── demo.gif
│   ├── demo-output-synthetic.jpg
│   └── saudi/
│       ├── dvids-7926006-prediction.jpg
│       ├── dvids-7930293-prediction.jpg
│       └── saudi-video-demo.gif
├── results/
│   ├── benchmark.json
│   ├── confusion_matrix.png
│   ├── metrics.json
│   ├── model_checksums.txt
│   ├── saudi-video-benchmark.json
│   └── README.md
├── scripts/
│   ├── benchmark.py
│   ├── download_saudi_sources.py
│   ├── download_saudi_video.py
│   ├── run_saudi_pilot.py
│   └── run_saudi_video_pilot.py
├── outputs/
├── Dockerfile
├── LICENSE
├── requirements-dev.txt
├── requirements.txt
└── README.md
```

## Limitations

- Accuracy can decrease for distant workers, poor lighting, or unusual camera angles.
- The public dataset may not represent a specific real construction site.
- Predictions are not a replacement for a qualified safety officer.
- A confidence threshold must be validated for the target environment.

## License

This project is released under the [GNU AGPL-3.0](LICENSE). It uses Ultralytics software and models,
as well as the AGPL-3.0 Construction-PPE dataset. Under Ultralytics' current licensing guidance,
using its software or trained models requires either complying with AGPL-3.0 for the complete
project, as this repository does, or obtaining an Ultralytics Enterprise license when those
obligations cannot be met—for example, for a proprietary closed-source deployment. Review the
dataset terms separately before commercial use.
