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

## Follow-up: does seeding or a body mask close the gap?

Tested two candidate fixes on a 3-case spot check (P001, P065, P130), both
against the same labels-1-3 ground truth:

| Approach | mean Dice |
|---|---:|
| plain threshold + largest components (baseline above) | 0.262 |
| + `tissues.segment_body` mask (exclude exterior scanner-FOV air) | 0.067 |
| `nasal_airway_ct.extract_ct_nasal_airway` (nostril-seeded growth) | 0.064 |

Both candidates made things **worse**, and for related reasons — both were
built for whole-head CT (Visible Human), not NasalSeg's pre-cropped FOV:

- `segment_body` builds the body silhouette by thresholding tissue and
  filling holes. Hole-filling only works when the air cavity is fully
  enclosed by tissue *within the array*. NasalSeg crops the FOV tight
  around the nose, so real nasal-cavity air is often touching the crop
  boundary (~25% of boundary voxels are air-range HU, checked on P001) —
  indistinguishable from true exterior air by this method. The mask ends up
  excluding real airway, not just scanner background.
- `extract_ct_nasal_airway` restricts its "vestibule + septum-split" domain
  to a small ROI anchored at the detected naris (built to find nostril
  ports for whole-head boundary conditions, not to reproduce the full
  nasal-cavity + nasopharynx extent). Its `passage_lumen` came out at ~10k
  voxels vs ~80k in the ground truth — it's solving a different, narrower
  problem than this Dice test asks.

**Conclusion:** for NasalSeg-style pre-cropped volumes, the crop has
already done the "remove scanner background" step, so the plain
threshold + largest-components baseline is the better of the three
classical options tested. The ~0.25 Dice ceiling isn't an exterior-air
problem or a threshold-tuning problem — it's specifically that nothing
here distinguishes nasal-cavity air from paranasal-sinus air within the
same connected blob.

## Next step

Two real paths forward, not yet tried:

1. A seed-and-flood approach scoped to the crop itself (not the whole-head
   ROI logic above) — seed from the most-anterior air voxels (no naris/
   vestibule restriction) and flood through air, then split off components
   whose connection to the seed passes through a narrow bottleneck
   consistent with a sinus ostium (a caliber threshold along the connecting
   path), rather than keeping every large component.
2. nnU-Net on NasalSeg (already scaffolded, `docs/nnunet_nasal.md`) — likely
   the more reliable route, since distinguishing nasal cavity from sinus air
   by geometry alone is exactly the kind of ambiguous case learned
   segmentation handles better than hand-tuned heuristics.

## nnU-Net results (pending)

Path 2 is underway: `nnUNetTrainer_250epochs`, fold 0 of `3d_fullres`, on
Colab (see `docs/nnunet_colab_training.md`). Once training finishes,
`scripts/compare_nnunet_vs_classical.py` scores it against the same ground
truth and classical baseline as above, on nnU-Net's own held-out fold-0
validation cases — fill in from that script's printed summary:

| Approach | mean Dice (labels 1-3) |
|---|---:|
| classical threshold + largest components (hu_max=-350) | 0.254 |
| nnU-Net (`nnUNetTrainer_250epochs`, fold 0) | *pending* |

nnU-Net per-structure Dice (mean) — this is the number that actually tests
whether the model learned to separate nasal cavity from sinus, where every
classical approach above failed to:

| Structure | mean Dice |
|---|---:|
| left_nasal_cavity | *pending* |
| right_nasal_cavity | *pending* |
| nasopharynx | *pending* |
| left_maxillary_sinus | *pending* |
| right_maxillary_sinus | *pending* |

Once trained, the model is also wired into the working pipeline as
`process_case(..., mask_source="nnunet")` (`src/sinus_cfd/nnunet_infer.py`),
which — unlike `mask_source="labels"` — needs no expert labels at all. That's
what makes it usable on a real new-patient CT rather than only on NasalSeg's
130 pre-labeled cases: the prediction supplies both the airway mask and the
label map boundary-condition port placement needs.
