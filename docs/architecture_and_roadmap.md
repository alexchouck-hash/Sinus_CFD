# Sinus_CFD — Architecture & Roadmap

Patient-specific nasal airflow simulation and virtual surgery planning from head CT.
This document lays out the intended pipeline, tool choices, the metrics that matter
clinically, and a staged build plan.

| Companion doc | Role |
|---------------|------|
| [`architecture.md`](architecture.md) | **What the repo implements today** (modules, data flow, viewer) |
| [`product_roadmap.md`](product_roadmap.md) | Product milestones (near / medium / long term) |
| **This file** | Field methodology, CFD/mesh guidance, clinical metrics, staged engineering plan |
| [`AGENTS.md`](../AGENTS.md) | Agent-oriented map of scripts and conventions |

> **Scope / status caveat.** This is research and planning software. The moment it
> informs an actual surgical decision or drives a navigated instrument in a patient,
> it becomes a regulated medical device (Software as a Medical Device) with a serious
> validation and regulatory burden. Keep it framed as research until that is addressed.

---

## 1. Where this sits in the field

This project rebuilds the "virtual surgery planning" pipeline that rhinology and
biomedical-engineering labs have developed over ~15 years (e.g. Marquette's Airway
Biomechanics Lab). That is good news: there is an existing literature to validate
against and to borrow methods from, rather than inventing from scratch.

The end-to-end chain is:

```
CT volume → airway segmentation → surface + volume mesh → CFD (airflow)
          → geometric + flow metrics → constriction detection
          → virtual surgery edit → re-mesh → re-CFD → compare
          → (parallel) pathology flags + navigation path planning
```

---

## 2. Segmentation: extracting the air

Air ≈ **−1000 HU** on the calibrated Hounsfield scale (air = −1000, water = 0,
soft tissue −100…+100, bone ≥ +400). Finding air is trivial; producing a *usable
airway domain* is not.

- **Upper threshold matters most.** Published nasal work uses upper cutoffs of
  −300 to −600 HU depending on scanner and slice thickness. Lower → tighter lumen
  (thin passages lost); higher → bleed into mucosa / partial-volume voxels. A common
  compromise is **−400 to −500 HU**, tuned per scan.
- **Threshold alone is not enough.** Raw thresholding grabs *all* air (room air,
  mouth, ear canals, bowel gas). Required cleanup:
  1. Threshold → binary air mask.
  2. **Connected-component analysis**, seeded at the nostrils, to isolate the
     nasal/sinus tree.
  3. Morphological closing to seal pinhole leaks.
  4. Explicit domain termination: close the mouth, cut the outlet at the
     nasopharynx / trachea.
  This is what the repo already does via `--mask-source labels|hu|labels_and_hu`
  and port detection.
- **Partial-volume warning.** The wall position shifts a fraction of a voxel with
  threshold choice, and CFD resistance scales ~ radius⁴, so the threshold directly
  changes pressure-drop numbers. Run a **sensitivity check** (segment at −400 and
  −500, compare) rather than trusting one value.
- **Tooling.** Don't hand-roll everything. **3D Slicer** Segment Editor
  (threshold + region-grow + local threshold + island removal) produces clean
  ground-truth masks fast. Automate (classical pipeline, then **nnU-Net**) once the
  manual workflow is understood.
- **CBCT caveat.** Cone-beam CT (common in ENT/dental) is not reliably HU-calibrated;
  a fixed threshold will not transfer. Tune per scanner.

## 3. Data sources

- **NasalSeg** — 130 head CTs with expert labels (L/R nasal cavity, nasopharynx,
  L/R maxillary sinus). Open access on Zenodo. Best starting point: gives images
  *and* validated labels, so segmentation accuracy (Dice) is measurable.
- **Visible Human head** — single cadaver, coarse, limited pharyngeal air (repo
  synthesizes a conduit). Fine for a full head-to-trachea geometry demo.
- **TCIA** head-and-neck CT collections and **HaN-Seg** widen anatomy variety, but
  lack airway labels (organs-at-risk only).
- **Own scans / DICOM** — convert with `dcm2niix` or Slicer to NIfTI/NRRD. Confirm
  true HU-calibrated CT.

## 4. CFD: solver, mesh, physics

**OpenFOAM is a defensible, free choice** but expect to spend most effort on meshing,
not physics. In the literature the workhorses are **Ansys Fluent** and
**Siemens STAR-CCM+**, chosen mainly for easier boundary-layer meshing and reviewer
trust. Open-source nasal CFD exists (pimpleFoam-family solvers).

- **Meshing is the hard part.** Nasal passages are thin, tortuous, high-surface-area.
  A **boundary-layer prism mesh** at the mucosa (~5 layers, total ~0.2 mm) is
  mandatory or wall shear and heat flux (the surgically relevant quantities) are
  unreliable. Laying clean prisms with `snappyHexMesh` on a complex STL is fiddly.
- **Physics — start simple.** Resting breathing (~18 L/min, 15–20 LPM) is
  transitional, largely laminar. **Laminar steady-state inspiration** is validated
  and correlates with measurements. Use **LES** only for higher flow or fine
  turbulent structure (much finer mesh, transient). **Avoid RANS k-ε**: poor in
  thin transitional passages.
- **Alternatives:** SimVascular (free, vascular CFD stack transfers), NASAL-Geom
  (free upper-airway reconstruction), or a Fluent academic license if budget allows.
- **Repo status.** The current potential-flow / Darcy field is a good *visualization
  preview* (runs, hits the flow target) but is **not** Navier–Stokes; it cannot give
  trustworthy pressure drop, wall shear, or heat flux. Real CFD (§6, stages 3–4) is
  still ahead.

## 5. Constriction detection & tissue-removal recommendation

- **Geometry first (no CFD needed).** Extract the **centerline**, compute
  **cross-sectional area** along it; the **minimal cross-sectional area (MCA)** and
  its location is the classic constriction detector.
- **Flow metrics that correlate with symptoms**, roughly by strength of evidence:
  1. **Wall heat flux / mucosal cooling** — strongest correlate with perceived
     patency (the nose senses airflow via TRPM8 cooling, not pressure).
  2. **Nasal resistance / pressure drop.**
  3. **Wall shear stress.**
  4. **Airflow allocation** between passages.
  Compute these **regionally** so the tool can point at *where* the problem is.
- **Virtual surgery loop.** Digitally edit the mesh to mimic the operation (shave
  inferior turbinate, straighten septum, enlarge ostium), re-mesh, re-run CFD,
  compare pre/post. The literature has done exactly this for turbinate reduction and
  septoplasty and shown the simulated change predicts the direction and rough
  magnitude of real outcomes.
- **Honest limits.** "Which tissue to remove" is currently a human decision informed
  by the sim, not credibly auto-prescribed. You can *rank* parameterized candidate
  edits by predicted metric improvement. Validation evidence is real but limited:
  small cohorts, the nasal cycle confounds pre/post, correlations vary.

## 6. Pathology: polyps & sinus drainage

- **Polyps** = soft tissue filling would-be air space. Detect via a learned
  segmentation model (**nnU-Net**, already scaffolded) rather than thresholding, to
  distinguish polyp vs turbinate vs thickened mucosa.
- **Sinus drainage** is not mainly a CFD-of-air problem; sinuses clear via
  **mucociliary transport** through tiny ostia. Pragmatic v1: segment each sinus,
  trace its ostium/drainage pathway, flag those that are occluded or below a caliber
  threshold. Full mucus/particle transport modeling is a research frontier.

## 7. Navigation path planning (frontal sinus, eustachian tube)

- **Method.** Skeletonize the air lumen; find the most-open path from nostril to
  target using a distance-transform-weighted geodesic (repo's `open_path.py` /
  `demo_open_paths.py`). **VMTK** (with 3D Slicer) is the mature tool for centerline
  + radius-along-path.
- **Frontal sinus** drains through the narrow, variable frontal recess. Constrain the
  path to the real drainage pathway with anatomical waypoints or it will cut through
  bone.
- **Eustachian tube** is normally **collapsed, soft-tissue-walled**, so it may not
  appear in an air mask at all. Trace from soft-tissue landmarks; accept that CT
  alone may not show a patent lumen.
- **Making it real navigation.** The path must be **registered** into the navigation
  system's coordinate frame (fiducial or surface registration) and exported in a
  format the nav system ingests. That is an integration + regulatory problem more
  than an algorithms one.

## 8. Staged build plan

Each stage produces something testable.

1. **Segmentation you trust.** Reproduce NasalSeg labels with threshold + region-grow;
   measure Dice vs ground truth. Add nnU-Net once classical works. → clean mask + STL.
2. **Geometry analysis (no CFD).** Centerline, cross-section-area profile, MCA
   detection, path planning to targets.
3. **Real CFD, one case.** Volume mesh with prism layers; laminar steady inspiration
   at ~18 L/min; extract pressure drop / resistance / wall shear / heat flux;
   validate resistance against published ranges.
4. **Virtual surgery loop.** Parameterized edits → re-mesh → re-run → compare.
5. **Pathology detection.** Polyp segmentation; ostium patency flags.
6. **Navigation export + registration.** Only after geometry is solid.

## 9. Current-repo assessment

Sound, sensibly staged architecture. Physiology (~18 L/min from tidal volume), the
nostril-inlet / trachea-outlet / mouth-closed BCs, centerline / most-open paths, and
the Streamlit demo stack are the right primitives.

**Implemented (research demo — see `architecture.md`):**

| Area | Status |
|------|--------|
| Whole-head + CT L/R cavities + tip vestibules | Working (Visible Human) |
| OpenFOAM `simpleFoam` import → `*_flow.npz` | Working (case scaffold + import) |
| Turbulent-looking wispy pathlines → trachea | Working (visualization, not LES) |
| Dual naris→frontal instrument corridors | Working |
| High-\|u\| IT / MT / septum zones + treatment ranking | Heuristic demo only |
| Potential / Darcy flow | Preview path when foam U missing |

**Still the bulk of research-grade work (stages 3–4 above):**

- Boundary-layer prism mesh quality; validated resistance / WSS / heat flux  
- Virtual surgery edit → remesh → re-CFD compare loop  
- Patient CT upload + labeled segmentation accuracy (Dice)  
- Navigation registration / export  

The imported simpleFoam field is real NS-class CFD for the current mesh, but **wall
quantities and mesh independence are not yet at publication / clinical-validation
quality.** Potential flow remains only a visualization fallback.

---

## References

- NasalSeg dataset (Zenodo): https://zenodo.org/records/13893419 — paper: https://www.nature.com/articles/s41597-024-04176-1
- CFD & Virtual Septoplasty in Nasal Airway Obstruction, narrative review (2026): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12791179/
- Virtual septoplasty, predicting surgical outcomes: https://pmc.ncbi.nlm.nih.gov/articles/PMC7148186/
- Identifying patients who benefit from inferior turbinate reduction via simulation: https://pmc.ncbi.nlm.nih.gov/articles/PMC4641847/
- Marquette Airway Biomechanics Lab, Virtual Surgery Planning: https://mcw.marquette.edu/biomedical-engineering/airway-biomechanics-lab/virtual-surgery-planning.php
- Open-Source CFD Analysis of Nasal Flows (OpenFOAM): https://www.researchgate.net/publication/359869687_Open-Source_CFD_Analysis_of_Nasal_Flows
- Computational modeling and validation of human nasal airflow: https://pmc.ncbi.nlm.nih.gov/articles/PMC5694356/
- 3D modeling & automatic analysis of nasal cavity/paranasal sinuses for CFD: https://link.springer.com/article/10.1007/s00405-020-06428-3
- VMTK centerline extraction in 3D Slicer: https://www.slicer.org/wiki/Modules:VMTK_in_3D_Slicer_Tutorial:_Coronary_Artery_Centerline_Extraction
