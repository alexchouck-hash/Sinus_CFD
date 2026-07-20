# The NasalSeg airway-connectivity limitation (and how to get correct bilateral flow)

## The problem

CFD needs a **single connected fluid domain** from the inlets (nostrils) to the
outlet (nasopharynx/trachea). Real inspiration is
nostril → nasal cavity → **choana** → nasopharynx → pharynx, on **both** sides.

NasalSeg labels the nasopharynx (label 3) as a **compact region that does not
span both choanae** — so one nasal cavity (usually the right, label 2) is left
**disconnected** from the nasopharynx. Quantified across the whole dataset:

- **0 of 130** NasalSeg cases have both nasal cavities connected to the
  nasopharynx via labels {1,2,3} (even after morphological closing).
- The same disconnect appears in the **nnU-Net predictions** — the model is
  trained on these labels, so **retraining does not fix it**.
- For P001 specifically, the nasopharynx label sits almost entirely left of the
  septum (182 of 25,632 voxels on the right), while the right cavity is far
  right; the nearest label gap is ~25 voxels **across the septum** (the wrong
  place to bridge).

## Why the obvious fixes don't work on NasalSeg crops

| Attempt | Result |
|---|---|
| Connect labels {1,2,3} with morphological closing | Stays 2 components (gap is real, not a thin choana) |
| HU-air bridge at −400 (the pipeline default) | Connects, but routes the right side **through the maxillary sinus** — the "air from the sinuses" artifact |
| HU-air region-grow at −200 (reveals the true thin choanal air) | Connects bilaterally **but leaks to exterior** — the tight NasalSeg crop's airway is continuous with the room air through the nostrils, and excluding boundary-connected air removes the airway too |
| Targeted choanal bridge (air near both R-cavity and nasopharynx) | No such air exists — the mislabeled nasopharynx is >25 voxels away across the septum |
| Hide sinus segments in the viewer | Removes the **right-side flow entirely**, because its only outlet path runs through the sinus |

The root cause is upstream (the labels), so it can't be cleanly patched
downstream.

## What actually works: whole-head HU airway extraction

The **Visible Human** case *does* have a connected bilateral airway — a single
nares→trachea lumen (55 mL, one connected component) — because it was segmented
by `process_whole_head.py` **from the CT air lumen on a whole head**, not from
sparse crop labels. A whole-head scan provides the body context needed to
region-grow the airway and control leakage (the airway is bounded by tissue, not
flush with the crop boundary).

So the method the roadmap calls for — HU threshold + region-grow airway
extraction — is correct; it just needs a **whole-head** (or at least
skin-bounded) volume, which NasalSeg's tight crops don't provide.

## Recommendation to get physiologically-correct bilateral nasal flow

1. **Best:** a living-subject **whole-head CT** → `process_whole_head` HU airway
   extraction (connected geometry) → the validated CFD pipeline (physiological
   flow). This gets both correct geometry *and* correct physics. (Visible Human
   gives the geometry but its cadaver CFD is weak.)
2. **Reliable stopgap:** manually correct one case's nasopharynx/choana in
   3D Slicer (paint the missing right choana + nasopharynx), then run the
   pipeline. Turns any NasalSeg case into a connected domain.
3. **Not viable:** relabelling via nnU-Net — it inherits the connectivity gap
   from its training labels.

## Current state of the P001 airflow demo

P001's CFD used the −400 HU-bridged domain, so it **is** bilateral, but the
right side routes near/through the maxillary sinus (the segmentation limitation
above). The viewer shows the full bilateral streamlines with this caveat rather
than hiding half the nose. The **resistance** number is unaffected (sinuses are
dead-ends with ~zero net flow, and it validated against literature) — this is a
flow-visualization fidelity issue, not a physics-validity one.
