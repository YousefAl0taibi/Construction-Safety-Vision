# Construction Safety Vision — Model Card

## Status

No first-party checkpoint has been published yet. Release `v0.1.0` is created only after the Colab
training notebook produces a held-out test report and an ONNX model that passes a smoke test.

## Intended use

Screening construction imagery for people, hardhats, and safety vests. It is a decision-support
prototype, not a certified occupational-safety system. A human must review alerts.

## Dataset

Atharion PPE Detection Dataset v1.0-public: 1,130 images, 5,355 boxes, and three classes. Its
scene-group split contains 791 train, 170 validation, and 169 held-out test images. SHA-256 of the
published ZIP: `f5f6f1966c4bdfff4151ee68fa0da21f446e1b788d15280e5f70028c69ec88c5`.

The dataset combines AGPL-3.0 and CC BY 4.0 sources. Keep its `NOTICE.md` and `attribution.csv` with
any redistributed copy. The trained Ultralytics checkpoint and its ONNX export are published under
AGPL-3.0 unless covered by another applicable licence.

## Known limitations

- Staged-photo to CCTV domain gap.
- Few small objects; distant PPE performance is not established.
- Positive-only taxonomy; compliance is derived geometrically and lacks event-level ground truth.
- Performance must be recalibrated for each camera angle and lighting condition.

## Metrics

Pending the first reproducible Colab run. The release must include Precision, Recall, mAP50,
mAP50-95, per-class AP, and a deployment-device latency report.

