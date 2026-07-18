# Sinus_CFD product roadmap

## Vision

Clinical decision-support (future): upload head CT → structured airway/sinus analysis → interactive airflow/drainage under baseline and virtual surgery.

**Not a medical device.** Current code is a research prototype.

---

## MVP progress (`MVP` branch)

Building the minimum viable thread of the vision — **CT → trustworthy
segmentation → per-side clinical geometry readout** — against the staged plan
in `docs/architecture_and_roadmap.md`.

| Stage | Status | Evidence |
|-------|--------|----------|
| **1. Segmentation you trust** | **Done** | nnU-Net (`nnUNetTrainer_250epochs`, fold 0) **0.885 airway Dice** vs 0.260 classical baseline on 26 held-out cases; per-structure 0.72–0.97, cleanly separating nasal cavity from sinus. `docs/stage1_segmentation_baseline.md` |
| **2. Geometry analysis (no CFD)** | **Done** | Per-side volume, minimal cross-sectional area (MCA), L/R asymmetry ratio. `docs/stage2_geometry_metrics.md`, `scripts/geometry_report.py`, viewer "Geometry report" mode |
| 3. Real Navier–Stokes CFD | **In progress (resistance validated)** | Full pipeline runs end-to-end in Docker OpenFOAM on P001: CT → passage → geometry export → **watertight prism-layer mesh** (259k cells, checkMesh OK, skewness 3.2) → `simpleFoam` laminar → nasal resistance R=ρΔP/Q (`scripts/compute_nasal_resistance.py`). Flow sweep (18/36/60/90 L/min) shows physiological Rohrer nonlinearity: **at ΔP≈150 Pa (rhinomanometry's own condition) R=0.156 Pa·s/mL, inside the published 0.10–0.35 range** — the resting R=0.052 is simply a lower operating point, not an error (`scripts/flow_sweep_report.py`). **Mucosal cooling** (passive-T solve): inspired air 20°C→**34.4°C** at the nasopharynx (85% conditioning, 4.9 W heat loss), matching published ~32–34°C with no tuning; resistance unchanged confirms passive coupling (`scripts/compute_mucosal_cooling.py`). Remaining: formal mesh-independence study, spatial wall-heat-flux map. `docs/stage3_cfd_methodology.md` |
| 4. Virtual surgery loop | Not started | Depends on Stage 3; MVP geometry metrics are the pre/post quantities to compare |
| 5. Pathology (polyps, ostium patency) | Not started | |
| 6. Navigation export | Not started | |

**How to run the MVP thread today:**

```powershell
# Segmentation quality (needs trained weights under data/nnUNet_results/)
py -3.12 scripts/compare_nnunet_vs_classical.py --pred-dir <fold_0/validation> --nasalseg-root data

# Per-side geometry report (expert labels, or --mask-source nnunet on any CT)
py -3.12 scripts/geometry_report.py --case P001 --data-root data

# Real CFD one case (needs Docker Desktop running)
py -3.12 scripts/process_case.py --case P001 --data-root data
py -3.12 scripts/analyze_passage.py --case P001 --skip-flow
py -3.12 scripts/export_openfoam_geometry.py --case P001
py -3.12 scripts/scaffold_openfoam_case.py --case P001 --wall-layers 5
powershell -File scripts/run_openfoam_docker.ps1 -Case P001
py -3.12 scripts/compute_nasal_resistance.py --case P001
```

Training the model on Colab: `docs/nnunet_colab_training.md` +
`notebooks/train_nnunet_colab.ipynb`.

---

## Implemented (demo — Visible Human)

| Capability | Status |
|------------|--------|
| Whole-head CT process + skin/airway | Done |
| CT L/R cavities, tip vestibules | Done |
| OpenFOAM simpleFoam import | Done (case foam + import script) |
| Turbulent wispy pathlines (volume + naris seeds) | Done |
| Dual naris→frontal instrument paths | Done |
| High-\|u\| zones: IT / MT / septum + toggles | Done |
| Heuristic treatment ranking | Done |
| Streamlit viewer | Done (`app/viewer.py`) |
| NasalSeg process + potential flow | Done |
| nnU-Net Dataset501 (trained, fold 0) | Done — 0.885 airway Dice (see MVP progress) |
| Per-side geometry report (volume, MCA, L/R asymmetry) | Done (`scripts/geometry_report.py`) |

See **`AGENTS.md`** and **`docs/architecture.md`** for how to run the current stack.
For methodology (segmentation thresholds, CFD mesh quality, clinical metrics, staged
engineering plan), see **`docs/architecture_and_roadmap.md`**.

---

## Near term

| Item | Notes |
|------|--------|
| Patient CT upload path | De-ID DICOM → same pipeline as VH |
| Better sinus/turbinate labels | nnU-Net or TotalSegmentator-class models |
| Virtual surgery: edit mask → re-run pathlines/CFD | Compare pre/post Q and high-\|u\| volume |
| Calibrated resistance / pressure drop metrics | Beyond visualization speed maps |
| OpenFOAM dual-inlet balance checks | Explicit 50/50 flux reporting |

---

## Medium term

### Pathway analysis

- Naris → ostium → sinus geodesic (per sinus)  
- Metrics: length, min cross-section, tortuosity  

### Drainage / mucus proxies

- Surface advection toward ostium  
- Virtual ostium widening scenarios  

### Imaging assists (need labels + validation)

| Code | Condition |
|------|-----------|
| CRS | Chronic rhinosinusitis patterns |
| NAO | Septal deviation, turbinate hypertrophy, MCA |
| NVC | Nasal valve geometry |
| Polyps | Soft-tissue masses |

---

## Long term

1. Secure CT upload + de-identification  
2. Auto-segmentation of airway, sinuses, septum, turbinates, ostia  
3. Report + interactive virtual surgery comparison  
4. Regulatory pathway if clinical claims are made  

---

## Viewer evolution

| Capability | Status |
|------------|--------|
| 3D skin + cavities + wispy pathlines | **Now** |
| Frontal paths + zone pink toggles | **Now** |
| Treatment panel | **Now** |
| Geometry report mode (per-side volume / MCA / L/R asymmetry) | **Now** |
| Side-by-side pre/post virtual surgery | Planned |
| Time-resolved breath cycle | Planned |
