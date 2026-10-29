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

Create an environment and install the two dependencies:

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

## Project structure

```text
Construction-Safety-Vision/
├── notebooks/
│   └── train_colab.ipynb
├── src/
│   └── detect.py
├── models/
│   └── README.md
├── samples/
│   └── README.md
├── results/
│   └── README.md
├── outputs/
├── requirements.txt
└── README.md
```

## Limitations

- Accuracy can decrease for distant workers, poor lighting, or unusual camera angles.
- The public dataset may not represent a specific real construction site.
- Predictions are not a replacement for a qualified safety officer.
- A confidence threshold must be validated for the target environment.
