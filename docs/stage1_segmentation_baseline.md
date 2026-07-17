# Stage 1 baseline: classical threshold + region-grow vs NasalSeg labels

Staged build plan item 1 (`docs/architecture_and_roadmap.md` §8): *"Reproduce
NasalSeg labels with threshold + region-grow; measure Dice vs ground truth."*
This records the first measurement.

## Method

`scripts/evaluate_nasalseg_dice.py` builds a binary airway mask from HU alone
(`sinus_cfd.pipeline.build_hu_threshold_mask`: threshold → morphological
closing → gap-bridging region-grow → drop components under 200 voxels) and
compares it against expert NasalSeg labels via the Dice coefficient, sweeping
the upper HU cutoff from -350 to -600 as the roadmap's sensitivity check
calls for.

```powershell
py -3.12 scripts/download_nasalseg.py
py -3.12 scripts/evaluate_nasalseg_dice.py --nasalseg-root data --n-cases 20
```

(NasalSeg's zip extracts `images/` and `labels/` directly under `data/`, not
nested under `data/NasalSeg/` — pass `--nasalseg-root data`.)

## Results (20 of 130 cases, evenly sampled)

Ground truth = labels 1–3 (left/right nasal cavity + nasopharynx), the
project's `DEFAULT_AIRWAY_LABELS`:

| hu_max | mean Dice | stdev | min | max |
|-------:|----------:|------:|----:|----:|
| -350 | 0.254 | 0.050 | 0.155 | 0.343 |
| -400 | 0.253 | 0.052 | 0.147 | 0.343 |
| -450 | 0.252 | 0.053 | 0.138 | 0.345 |
| -500 | 0.249 | 0.055 | 0.130 | 0.346 |
| -550 | 0.245 | 0.057 | 0.124 | 0.347 |
| -600 | 0.242 | 0.058 | 0.117 | 0.347 |

With sinuses folded into ground truth (labels 1–5, `--include-sinuses`):

| hu_max | mean Dice | stdev |
|-------:|----------:|------:|
| -350 | 0.401 | 0.046 |
| -400 | 0.398 | 0.049 |
| -600 | 0.371 | 0.055 |

## Reading the numbers

1. **Threshold barely matters** (0.242–0.254 across a 250 HU range). The
   error isn't coming from where the cutoff sits on the air/tissue boundary —
   it's structural.
2. **Including sinuses in the ground truth raises Dice by ~0.15** at every
   threshold. That isolates the dominant error: naive HU threshold plus
   largest-connected-components keeps *all* large air spaces in the crop,
   including the maxillary sinuses, which NasalSeg's `DEFAULT_AIRWAY_LABELS`
   deliberately excludes (sinus lumen is a separate structure from the
   nasal airway proper). The current classical mask is a superset of the
   intended target, not a noisy version of it.
3. **Confirms the roadmap's own caveat** (§2, "threshold alone is not
   enough"): distinguishing the nasal airway from paranasal sinus air needs
   the nostril-seeded region growing already implemented in
   `nasal_airway_ct.py` (interior air is grown from a naris seed, not just
   kept as "largest components"), or a learned segmentation (nnU-Net,
   already scaffolded — `docs/nnunet_nasal.md`), rather than a threshold
   sweep alone.

## Next step

Re-run this same Dice measurement against `nasal_airway_ct.extract_ct_nasal_airway`
(nostril-seeded growth, already excludes sinuses by construction) to check
whether seeding — not thresholding — closes most of the gap, before
committing to training nnU-Net.
