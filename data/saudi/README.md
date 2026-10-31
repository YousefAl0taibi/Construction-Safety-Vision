# Saudi-domain pilot

This directory defines a small, license-documented pilot for checking the detector on scenes with
reported capture locations in Saudi Arabia. Raw images are downloaded locally and remain ignored
by Git.

The manifest contains 21 documented sources from five scene groups. Visual review marked 15 as
`approved`, five as `excluded`, and one KAUST source as `candidate` pending download and review:

- construction of housing units at Prince Sultan Air Base;
- confined-space inspections at Prince Sultan Air Base;
- a construction-site safety drill in Saudi Arabia;
- a worksite safety installation described by DVIDS as being in Dammam; and
- scaffolding workers at KAUST.

Most sources come from a small number of galleries. They are correlated views, not 21 independent
sites, and must not be presented as a representative Saudi test set. They are suitable for
qualitative domain inspection and for finding obvious failure modes. A quantitative Saudi benchmark
still needs more independently collected and authorised sites.

## Download

Review [`sources.csv`](sources.csv) and [`ATTRIBUTION.md`](ATTRIBUTION.md), then run:

```text
python scripts/download_saudi_sources.py --dry-run
python scripts/download_saudi_sources.py
```

Images are written to `data/saudi/images/raw/`, which is intentionally ignored. For each successful
file, the downloader writes its direct asset URL, byte size, and SHA-256 digest to
`data/saudi/download-report.json`.
By default it downloads rows marked `candidate` or `approved`; rows marked `excluded` remain in the
manifest for an auditable record of the curation decision.

The command downloads the documented public image files: 1000-pixel previews for DVIDS and the
original Commons asset. It does not scrape Google Images. Search-engine results are only discovery
leads; a file is added to this manifest only after its original page and reuse terms have been
identified.

## Run the pilot

After placing `best.pt` in `models/`, run the detector over the 15 approved images:

```text
python -m scripts.run_saudi_pilot --device cpu
```

Annotated images are written to `outputs/saudi-pilot/annotated/`, and
`outputs/saudi-pilot/summary.json` records every detection plus aggregate class counts. This is a
qualitative domain check, not a precision, recall, or mAP measurement because the images do not have
reviewed ground-truth annotations.

## Review before annotation

1. Confirm that the source page still displays the same rights statement.
2. Remove near-duplicates and images without usable workers or PPE.
3. Do not perform face recognition or infer identity, nationality, or sensitive attributes.
4. Split by `scene_group`, never by individual image, to avoid leakage between similar frames.
5. Keep a final real-image test split separate from any fine-tuning data.
6. Annotate only the classes that are visibly supported; uncertain PPE should remain `UNKNOWN`.

The location field records the publisher's reported capture location, not the nationality of anyone
shown. One excluded DVIDS row preserves a caption/metadata location conflict explicitly rather than
silently resolving it.

## Licensed video check

[`video_sources.csv`](video_sources.csv) records one approved DVIDS B-roll video of airfield repair
at Prince Sultan Air Base. The source page marks the work Public Domain. DVIDS requires an
interactive sign-in for this download: open the recorded source page, choose its 1280×720 MP4, and
save it as `data/saudi/videos/raw/dvids-923711-airfield-repair.mp4`. Then run the 30-second check:

```text
python -m scripts.run_saudi_video_pilot --device cpu
```

The original and annotated MP4 files remain ignored. The derived GIF is intended
for `docs/images/saudi/`, while the machine-readable performance summary is intended for `results/`.
Review the generated frames before committing either file; a successful command is not by itself
evidence that the clip demonstrates useful tracking.
