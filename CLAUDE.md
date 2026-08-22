# Sinus_CFD — project intent and working agreement

Research prototype. **Not a medical device.** Nothing here is validated for clinical use.

`AGENTS.md` is the command/convention reference. `docs/handoff.md` is the honest current
state. **This file is the destination** — what the project is ultimately for, so that any
session can tell whether a change moves toward it or merely sideways.

---

## The product, end to end

A surgeon loads a head CT and gets back: where the air actually flows, where mucus cannot
drain, what a proposed operation would change, and a navigable path for the instrument that
performs it.

Five capabilities, in dependency order. Each one is worthless if the one before it is wrong.

### 1. Import CT and auto-segment

Ingest DICOM/NRRD and produce a **bounded, watertight fluid domain** — a closed surface with
labelled inlet (nares), outlet (nasopharynx/trachea), and wall patches. CFD needs a sealed
volume, not a voxel mask; an unsealed domain leaks and the solve is meaningless.

Segmentation is a means, not the goal. If a step can be skipped without degrading the mesh
or the physics, skip it.

The hard part is **not Dice** — it is topology and mesh quality:

- One 26-connected lumen from both nares to the outlet, with **no trans-septal shortcut**
  and **no path through a sinus**.
- Sinus air excluded from the flow domain but retained, named, for drainage analysis (§3).
- Surfaces smoothed and decimated so `snappyHexMesh` produces no orphan cells, no
  sliver cells, and acceptable non-orthogonality/skewness. Voxel staircase artefacts at
  1 mm become mesh pathologies. Determining the right smoothing/remeshing pipeline
  (taubin vs. laplacian vs. marching-cubes + decimation, and how much is too much before
  a 1–2 mm channel closes) is **open work**.
- Thin structures are the constraint: the nasal valve and ostia are 1–3 mm. Smoothing that
  flatters the mesh but closes an ostium has destroyed the thing we are measuring.

### 2. CFD airflow, nares → nasopharynx, **under 20 minutes per scan**

Quasi-steady inspiration at a physiological set-point, per-side flow split, resistance,
wall shear, and conditioning. The 20-minute budget is a **hard product constraint**, not an
aspiration, and it governs architecture: mesh size, solver choice, steady vs. transient,
and how much of the run is parallelised. If a full transient solve cannot fit, the honest
answer is a validated steady approximation with stated limits — not a slower pipeline.

Report magnitudes only when calibrated. The in-repo potential-flow preview is a
visualisation, not a measurement, and must stay labelled as such.

### 3. Mucus transport and sinus drainage

Airflow is not the whole clinical picture. Mucociliary clearance runs from each sinus,
through its ostium, into the nasal cavity and back to the nasopharynx. This requires:

- **Ostium detection and calibre** per sinus — maxillary, frontal, sphenoid, ethmoid.
  Calibre in mm is the drainage metric; an obstructed ostium is the disease.
- **Frontal drainage pathway** — the frontal recess, which is anatomically the most
  variable and surgically the most dangerous.
- **Middle meatus flow** — the common drainage channel for the maxillary, frontal and
  anterior ethmoid cells.
- Per-sinus patency reported as a number a surgeon can act on, not a picture.

### 4. Instrument paths for surgical navigation

Curved-seeker trajectories, exported to overlay on the CT in a navigation system:

| Target | Instrument | Constraint |
|---|---|---|
| Frontal ostium | frontal sinus seeker | **2 mm diameter**, curved |
| Maxillary ostium | maxillary seeker | its own curvature |
| Sphenoid ostium | sphenoid seeker | its own curvature |
| Eustachian tube orifice | ET seeker | ~4 in shaft, **45° bend**, **18.5 mm** working tip past the bend |

A path is only useful if the **instrument** fits, not merely the centreline: clearance must
be checked against the tool's diameter and its fixed curvature, and the path must be
reachable from a nostril without levering against structures the tool must not touch.
This is a geometric feasibility problem, not just a shortest-path problem.

### 5. Virtual surgery — before and after

Simulate the intervention and re-run the physics:

- **Airway:** septoplasty, concha/turbinate reduction, nasal-valve intervention
  (alar batten graft), and show the change in flow split, resistance and conditioning.
- **Drainage:** maxillary antrostomy, frontal drill-out, balloon sinus dilation
  (**6 mm balloon, ostium stretched to 3–4 mm**), and show the change in sinus clearance.

The deliverable is a **ranked, comparative** answer — complexity versus predicted benefit —
not a single simulation.

---

## What follows from this

Consequences worth stating, because they have already caused wrong turns in this repo:

- **Topology and patency beat Dice.** A segmentation scoring 0.885 against labels whose
  cavities do not reach the nasopharynx is useless for every one of the five goals. Measure
  connectivity, ostium calibre and mesh validity — not overlap alone.
- **The scan must contain the anatomy.** Several cases here have no patent nasal airway at
  all (soft tissue where air should be) or clip the maxillary sinuses at the FOV. Screen
  for it before segmenting, and prefer sub-millimetre sinus protocols; a brain-framed head
  CT will not carry §3 or §4.
- **Goals 3 and 4 need the sinuses kept, not discarded.** §2 wants them out of the flow
  domain; §3 and §4 want them named, with their ostia measured. Both, from one pass.
- **Fail loudly.** A silently wrong domain produces a plausible number, and a plausible
  wrong number is worse than an error. Prefer a HARD failure with a machine-readable reason.
- **Verify numerically before adopting.** Reproduce any claim — including one from another
  agent, and including one of your own from earlier in a session — against the actual arrays
  before building on it.
- **Automatic by default.** Manual labelling is a last resort, and only ever to generate
  complete volumes, never slice-by-slice review.

---

## Not goals

- A clinical or regulatory-grade device.
- Beating published Dice on a public dataset for its own sake.
- Replacing the surgeon's judgement. The output is decision support with stated uncertainty.
