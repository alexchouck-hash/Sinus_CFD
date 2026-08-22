# Sinus_CFD — Agent Task Queue

Orchestrated build toward the platform goal in [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md).
Up to ~5 agents run concurrently. **Owner** = `grok` (audit/research/draft, read-only,
staged in `grok_inbox/`), `subagent` (Claude build/integration under review), or
`integrator` (orchestrator: architecture + `src/` integration + verify-then-adopt).
Nothing is adopted until its claims/output are reproduced numerically.

Status: `todo` · `running` · `review` (output produced, integrator verifying) · `done` · `blocked`.

---

## Wave 1 — Stage A: Harden + validate the foundation  *(active)*

| ID | Task | Owner | Status | Acceptance criteria |
|----|------|-------|--------|---------------------|
| **A1** | CQ500CT100 (0.5 mm living) end-to-end: geometry → connectivity check → real CFD → bilateral vestibule→pharynx flow | integrator | **running** | Airway is one connected component nostrils→outlet; outlet on-midline (spacing-fixed); CFD solves + mass-conserves; bilateral streamlines render in viewer |
| **A2** | Automated **connectivity/topology QA** harness (`scripts/qa_connectivity.py`) auditing any case | subagent | **todo** | For a `--case`: reports both-nostrils→outlet connectivity, single-component airway, no sinus-shortcut path, mass conservation, outlet midline-offset; exit non-zero on failure (CI-gate ready); runs on P001 + VH + CQ500 |
| **A3** | **Regression test suite** (`tests/`, pytest) locking in the audit fixes + core math | subagent | **todo** | Tests for spacing-aware mm↔index, orientation-aware trim (synthetic flipped case), wall-gradient interior-only (ratio ≈ O(1)), bilateral seeding L/R balance; all green; documented `pytest` invocation |
| **A4** | Verify + fix pending audit claims **5** (flux scaling on Dirichlet inlet) and **6** (OpenFOAM import silent size-mismatch / BC side-effect) | grok (verify+draft) → integrator | **todo** | Each claim reproduced numerically (CONFIRMED/REFUTED in `grok_inbox/`); confirmed ones patched + re-verified; refuted ones recorded |
| **A5** | **Multi-case CFD validation** report: run pipeline on ≥3 connected cases, compare nasal resistance to published CFD + rhinomanometry norms | subagent | **todo** | `docs/validation_multicase.md` with per-case R at rest + at 150 Pa, vs Borojeni-class CFD norms and rhinomanometry 0.10–0.40 range; honest scoping (no "validated" without conditions) |
| **A6** | **Research brief**: correct nasal-CFD BCs & turbulence (valve), public validation benchmarks, and state-of-the-art segmentation of turbinates/septum/sinus **ostia** & eustachian tubes | grok (research) | **todo** | `grok_inbox/research/` brief with cited, real sources; concrete recommendations feeding Stages B–D; flags what is buildable now vs. model-dependent |

Concurrency note: A2, A3, A5 (subagents) + A4, A6 (grok) can run alongside A1
(integrator). A5 depends partly on A1/A2 outputs → start after A2's QA exists.

---

## Wave 2 — Stage B: Clinical analysis  *(queued)*

- **B1** Segment clinically-relevant structures (turbinates, septum, maxillary/ethmoid/frontal/sphenoid sinuses, ostia) — extend nnU-Net labels or add a model; validate Dice + topology.
- **B2** Per-side airflow summary + **in-sinus pressure/flow** readout surfaced in the viewer (left vs. right; per-sinus).
- **B3** Obstruction metrics: minimal cross-sectional area, nasal-valve angle, septal-deviation index, turbinate hypertrophy — per side, vs. norms.
- **B4** Disease detection: polyps / soft-tissue masses, CRS mucosal opacification, ostial constriction, drainage-pathway patency.

## Wave 3 — Stage C: Treatment planning  *(queued)*

- **C1** Findings → candidate treatments mapping (septoplasty; turbinate reduction inferior/middle, microdebrider/RF; nasal-valve stent graft/RF/bioresorbable; balloon dilation; antrostomy; frontal drill-out).
- **C2** Score each candidate by **complexity vs. benefit**; simulate top options via virtual surgery + CFD pre/post (ΔR, Δflow-split, Δconditioning).
- **C3** Generate the **disease-state + treatment-plan report** (surgeon-facing).

## Wave 4 — Stage D: Surgical navigation  *(queued)*

- **D1** Detect + 3D-highlight frontal & sphenoid **ostia** and **eustachian-tube** openings on the anatomy/patient render.
- **D2** Compute delivery **pathways** to those landmarks (geodesic through the lumen); export landmarks + pathways for navigation / balloon dilation.

---

## Backlog / cross-cutting

- CI: wire A2 (connectivity QA) + A3 (pytest) as a gate so regressions can't ship.
- DICOM upload + de-identification path for real patient scans.
- Evidence-based dead-code cleanup (verify call sites; "delisted from viewer" ≠ dead).
- Viewer: fold per-side flow, sinus pressure, disease flags, and navigation exports into `app/viewer.py`.

---

## Change log

- 2026-08-22 — **Dead-end sinus strip: DONE** (handoff item 3; decision **K9 -> K9a**).
  `through_path_passage` kept only voxels within 4 mm of *the single shortest* naris->outlet
  geodesic; a nasal cavity is a broad volume and the second nostril's route is longer than
  "shortest", so the contralateral cavity always read as a detour and the strip disabled
  itself on every real head. Replaced by `auto_airway.dead_end_sinus_strip`: sinus = a
  chamber whose local radius exceeds `SINUS_SEED_RATIO` x its widest-path (maximin) radius
  to any opening, extent recovered by watershed on -EDT with per-body markers, and any
  basin touching an opening or the merge zone rejected as passage. Openings at BOTH ends
  are what keep the nasal valve: cavity behind a tight valve still has a wide route to the
  posterior opening. Measured sinus: VH Female **15.1 mL** (8.9 + 7.0 bilateral maxillary),
  VH Male 27.5 mL, CQ500CT105 **3.3 mL** vs **4.1 mL** measured independently in-FOV,
  THCA none (it has no resolved sinus air). **A12 closed as a side effect**: `passage_lumen`
  is now distinct from `airway_mask` on `--no-legacy` (VH 34.9 vs 55.3 mL), so
  `qa_connectivity` check 3 is no longer vacuous. 47 pytest green (+7 in
  `tests/test_sinus_strip.py`).
  Three things measured the hard way and worth not re-learning: the literal neck-cut of K9
  **cannot work** (231 candidate necks on VH isolated 0 pockets; VH's maxillary is fused to
  the cavity through **19** partial-volume perforations, so there is no neck); the strip must
  run on the **whole-head airway**, not the nasal-box flood domain, whose faces are artificial
  cuts that read as wide openings (CQ500: 6358 fake "opening" voxels vs 240 at the real
  nares); and the merge zone is a passage **marker/rejection test**, never a bottleneck
  source (seeding widest-path from it cost VH one maxillary).
  **Still open:** ~1.2 mL of nasopharynx reads as sinus on CQ500CT105 because its merge zone
  came from the `air p75` fallback rather than a bone landmark - bounded by handoff item 2.

- 2026-08-20 — **Naris detector on living CT: DONE** (handoff §9 item 1). Root cause was
  not the detector's clustering but a silent contract failure: on THCA there is no resolved
  air anterior of y=43, `whole_head` paints a synthetic geodesic tube through soft tissue,
  the air-HU preference filter discards every tube voxel and relocates the "naris" shell
  ~24 mm posterior, the collapsed blob yields **the same voxel twice**, and the old repair
  nudged x by 2 voxels — leaving both seeds on one side of the septum (L=1172 R=**14**).
  Fix: `validate_naris_pair` (all thresholds in **mm**, spacing-aware) + fallback to the
  whole-head prior from `<case>_nares.json` (which was always passed and always correct,
  but only consulted when the detector returned `None`) + `naris_detection_HARD_fail`
  instead of inventing a seed. Also made the x-peak confidence bar mm-based
  (`MIN_NARIS_PEAK_SEPARATION_MM = 15.0`, byte-identical at 1 mm isotropic) and added a
  WARN when the HU filter relocates the shell. Measured: THCA **L=1644 R=1664** (balance
  0.012 → 0.988); Visible Human Female **19148/17983** and Male **9563/6449**
  byte-identical; CQ500CT100 now HARD-fails with a reason (previously returned L=0 R=0
  while reporting `competing_geodesic_flood` success). 39 pytest green incl. 13 new in
  `tests/test_naris_validation.py`. **Not fixed** (still open): the HU filter itself still
  relocates the shell — it only reports now; the choanal-landmark bone fallback
  (`WARN: bone merge empty`) still fires on THCA / VH Male / CQ500; septum ridge remains
  broken (VH Male 0, THCA 9339 looks too fat). Those are handoff items 2 and 4.
- 2026-07-21 — Queue created. Wave 1 (Stage A) opened; A1 running (CQ500 0.5 mm on
  spacing-fixed code). Audit fixes 1–3 adopted (`grok_inbox/2026-07-21_verification.md`).
- 2026-07-21 (progress) — **A2 done+verified** (`scripts/qa_connectivity.py`; P001/THCA
  pass hard checks). **A3 done+verified** (`tests/`, 14 pytest green; no new bug in the
  fixed code). **A4 done**: claims 5 & 6 CONFIRMED — claim-5 (flux scaling on Dirichlet
  inlet) fix from Grok was reproduced and found *insufficient* (39 m/s still off), so the
  approximate solver's *magnitude* is now treated as an uncalibrated preview (viz-only
  normalization pending); claim-6a (OpenFOAM import fail-hard on size mismatch) → adopt;
  claim-6b (import mutating BC JSON) → defer. **A6 done** (research brief in
  `grok_inbox/research/`). A1 CQ500 0.5 mm re-running clean to completion.
- 2026-07-21 — **New backlog A7 (HIGH):** `whole_head.py` airway **seed placement**
  assumes Visible-Human orientation; CQ500 (`y_anterior_is_low=False`) under-extracted in
  `process_whole_head`. Make extraction seed placement orientation-robust (Fix 3 already
  made the *trim* orientation-aware; the *extraction* is the remaining gap). A fresh
  comprehensive Grok code audit is running to surface further issues.
- 2026-07-21 — Comprehensive audit landed (`grok_inbox/2026-07-21_audit_comprehensive.md`,
  ~20 findings). Audit-derived tasks added below. Credit-conservation pause, then resumed.
- 2026-07-22 — **Auto airspace isolation on CQ500 abandoned (5 attempts, all ~0 mL).**
  Root cause is topological, not a bug: the nasal airway is 26-connected to ambient
  through the nares, and the tight crop connects the sinuses to the boundary too, so all
  151 mL of air is one component — no threshold/hull/sealed-component method can draw the
  nostril plane. Zero-shot nnU-Net also mislocated labels (domain shift). **Pivot to
  guided manual labeling in 3D Slicer** (`docs/slicer_labeling.md` + starter
  `data/nnunet_infer/CQ500CT100_starter.nii.gz` = all-air to carve). One hand-labeled
  case → fine-tune the model so later scans auto-label.
- 2026-07-21 — **CQ500 extraction failure CONFIRMED** on the clean run (233 voxels again) —
  empirically validates audit finding #1. **A7 deployed** (Claude subagent: orientation-
  robust + mm-based airway seeds in `whole_head.py`, must not regress VH/THCA).
  **A8 deployed** (Claude subagent: fix `assess_dicom_incoming.py` internal-air metric —
  currently drops open-naris airways). A5 held until A7 makes CQ500 available.

## Audit-derived tasks (Stage A hardening)

| ID | Task | Owner | Status | Source |
|----|------|-------|--------|--------|
| **A7** | Orientation-robust + mm-based airway seed placement in `whole_head.py`; unify the two A-P heuristics | subagent | **running** | audit #1,#2 |
| **A8** | Fix `assess_dicom_incoming.py` internal-air metric (body-interior air, not face-touching exclusion) | subagent | **running** | audit #3 |
| **A9** | Spacing-weighted (anisotropic) Laplace stencil in `flow_field.solve_pressure_potential` | grok-draft→integrator | todo | audit #4 |
| **A10** | Hard success gate in `vestibule_to_pharynx_streamlines.py` (non-zero exit if lines don't reach vestibule/outlet) | integrator | todo | audit #5 |
| **A11** | Adopt claim-6a (OpenFOAM import fail-hard on U/mesh size mismatch); decide claim-6b (BC-mutation side-effect) | integrator | todo | A4 |
| **A12** | Persist `y_anterior_is_low`/`superior_is_high_z` in BC JSON; fix qa_connectivity check-3 (passage_lumen is NOT sinus-free — verified byte-identical to airway_mask) | integrator | todo | audit med, verified |
| **A13** | DICOM stacker picks arbitrary series `[0]`; pick thinnest/most-slices (like assess_dicom_incoming) | integrator | todo | audit med |
| **A5** | Multi-case CFD validation vs literature | subagent | held (needs A7→CQ500) | queue |
