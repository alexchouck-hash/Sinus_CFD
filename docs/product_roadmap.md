# Sinus_CFD product roadmap

## Vision

Clinical decision-support (future): upload head CT → structured airway/sinus analysis → interactive airflow/drainage under baseline and virtual surgery.

**Not a medical device.** Current code is a research prototype.

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
| nnU-Net Dataset501 scaffold | Scaffold only |

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
| Side-by-side pre/post virtual surgery | Planned |
| Time-resolved breath cycle | Planned |
