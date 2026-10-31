# Attribution and reuse notes

The files listed in `sources.csv` remain under the terms stated by their original publishers. This
project does not relicense third-party photographs under the repository's AGPL license.

## DVIDS / U.S. government visual information

The listed DVIDS asset pages mark the selected photographs as **Public Domain**. DVIDS also states
that privacy/publicity rights are not waived, protected marks may remain, and use must not imply U.S.
Department of War endorsement. Retain the creator, source-page URL, and rights URL recorded in the
manifest.

Required project disclaimer from the current DVIDS notice:

> The appearance of U.S. Department of War (DoW) visual information does not imply or constitute
> DoW endorsement.

- Rights notice: <https://www.dvidshub.net/about/copyright>
- Source organisation: U.S. Army Corps of Engineers / U.S. Air Force via DVIDS

### Prediction examples tracked in this repository

The following derivatives preserve the original 1000-pixel preview dimensions and add only model
prediction boxes and labels:

| Repository file | Original source | Creator | Modification |
|---|---|---|---|
| `docs/images/saudi/dvids-7926006-prediction.jpg` | [CHU construction image 7926006](https://www.dvidshub.net/image/7926006/378th-aew-command-staff-tours-chu-construction) | Tech. Sgt. Alexander Frank | Prediction overlay added at confidence 0.35 |
| `docs/images/saudi/dvids-7930293-prediction.jpg` | [Confined-space image 7930293](https://www.dvidshub.net/image/7930293/confined-space-program-team-performs-preventive-manhole-inspections) | Tech. Sgt. Alexander Frank | Model returned no boxes; JPEG was re-encoded by the prediction pipeline |

Both source pages mark the photographs Public Domain. The DVIDS limitations and non-endorsement
statement above still apply to their use here.

### Video used for the motion demo

| Published derivative | Original source | Creator | Modification |
|---|---|---|---|
| `docs/images/saudi/saudi-video-demo.gif` | [Airfield damage repair during Native Fury](https://www.dvidshub.net/video/923711/2nd-distribution-support-battalion-conduct-airfield-damage-repair-during-native-fury) | Cpl. Meshaq Hylton | First 30 seconds processed with prediction, tracking, and status overlays; seconds 18–28 resized to 720 pixels wide and sampled at 1 FPS for the published preview |

The DVIDS page reports Prince Sultan Air Base, Saudi Arabia, marks the video Public Domain, and
identifies the original as a 66-second B-roll work dated 12 May 2024. The raw video and annotated
MP4 remain outside Git history; `video_sources.csv` records the exact download variant and clip
selection. The derivative is a model demonstration and does not imply DVIDS or Department of War
endorsement.

## KAUST scaffolding workers

- Title: `KAUST scaffolding workers.jpg`
- Creator: Wikimedia Commons user `Arbitrarily0`
- Source: <https://commons.wikimedia.org/wiki/File:KAUST_scaffolding_workers.jpg>
- License: Creative Commons Attribution-ShareAlike 3.0 Unported
- License text: <https://creativecommons.org/licenses/by-sa/3.0/>

Credit the creator, link the source and license, and indicate modifications. If the photograph or a
derivative is redistributed, preserve its CC BY-SA terms separately from the repository software
license.

These notes document the selected sources; they are not legal advice. Recheck the original page
before publishing a derived dataset or using the images commercially.
