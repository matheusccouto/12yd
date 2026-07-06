---
license: mit
tags:
  - penalty
  - football
  - lightgbm
  - shootout
---

# 12yd — Penalty Shootout Side Prediction

The current model card is [`docs/model-card-v4.md`](../docs/model-card-v4.md)
(v4 retrain, 437 training rows across 8 national-team + club tournaments,
LOTO CV aggregate SE 0.022). The v3 model card is preserved at
[`docs/model-card-v3.md`](../docs/model-card-v3.md) for historical
reference (v3 retrain, 151 pre-2026 training rows, 28-row 2026 holdout).

The files in this `model/` directory are a v3 snapshot:

- `lightgbm.pkl` — the v3 LightGBM artifact (151 pre-2026 training rows).
- `metrics.json` — the v3 held-out metrics report (the v3 numbers, not
  the v4 numbers in the v4 model card).
- `README.md` — this file.

The v4 work (commit `1df436c`, Issue #51) retrained on 437 rows but did
not commit the v4 artifact to this directory; the v4 retrain output
lives in the local `output/` slice (gitignored) and on Hugging Face
(`couto/12yd/model/`, pushed manually after the slice pipeline re-runs).
The v3 files are kept in the repo as a historical reference for the
state of the deployment at the time of the v3 release — they are not
the live artifact.
