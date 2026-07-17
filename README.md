# Sinus_CFD

**CT-based nasal airflow visualization and surgical-planning *research demo*.**

Ingest head CT → reconstruct nasal airspaces → estimate or import CFD velocity → show **turbulent pathlines**, **naris→frontal instrument corridors**, and **high-velocity tissue targets** with least-invasive treatment suggestions.

> **Not a medical device.** For research and education only. Do not use for clinical decisions without validation and regulatory clearance.

**GitHub:** https://github.com/alexchouck-hash/Sinus_CFD (`main`)

---

## Current demo (Visible Human)

| Capability | Status |
|------------|--------|
| Whole-head CT (Visible Human Female 1 mm) | Working |
| Tip nares + L/R cavities + tip vestibule open | Working |
| OpenFOAM simpleFoam velocity (when imported) | Working |
| Turbulent **wispy** pathlines (volume + naris seeds → trachea) | Working |
| Dual purple **naris → frontal** paths | Working |
| Pink **IT / MT / septum** high-\|u\| zones (toggles) | Working |
| Treatment ranking (demo heuristic) | Working |
| Streamlit viewer | Working (`app/viewer.py` **0.15.x**) |
| nnU-Net NasalSeg scaffold | Scaffold only (weak GPU) |

### Boundary conditions

| Boundary | Role |
|----------|------|
| Left + right nostrils | Inlets (~50/50 of total Q) |
| Trachea | Outlet (pressure reference) |
| Mouth | Closed |

Resting breath scale: \(V_T \approx 0.5\) L, RR 12 → mean inspiratory **~18 L/min** (quasi-steady). Details: [`docs/boundary_conditions.md`](docs/boundary_conditions.md).

---

## Quick start (demo case)

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
py -3.12 -m pip install -r requirements.txt

# Data (once)
py -3.12 scripts\download_visible_human_head.py

# Full chain if outputs missing (subset if already processed)
py -3.12 scripts\process_whole_head.py --case VisibleHuman_Head
py -3.12 scripts\refine_nasal_ct.py --case VisibleHuman_Head
py -3.12 scripts\extend_nasal_to_tip.py --case VisibleHuman_Head
# Optional CFD:
#   export + OpenFOAM Docker/WSL + import_openfoam_results.py
py -3.12 scripts\regenerate_curvy_pathlines.py --case VisibleHuman_Head
py -3.12 scripts\compute_surgical_guidance.py --case VisibleHuman_Head

py -3.12 -m streamlit run app\viewer.py --server.address 127.0.0.1 --server.port 8501
```

Open **http://127.0.0.1:8501**. Sidebar: toggle frontal paths and pink zones (IT / MT / septum).

---

## Agent / developer docs

| Doc | Audience |
|-----|----------|
| **[`AGENTS.md`](AGENTS.md)** | **Start here for AI agents** — pipeline, modules, conventions |
| [`docs/architecture.md`](docs/architecture.md) | System architecture + data flow |
| [`docs/viewer.md`](docs/viewer.md) | Viewer layers and cache |
| [`docs/surgical_guidance.md`](docs/surgical_guidance.md) | Frontal paths, zones, treatments |
| [`docs/data-sources.md`](docs/data-sources.md) | NasalSeg vs Visible Human |
| [`docs/docker_openfoam.md`](docs/docker_openfoam.md) | OpenFOAM in Docker |
| [`docs/open_paths.md`](docs/open_paths.md) | Most-open path algorithm |
| [`docs/product_roadmap.md`](docs/product_roadmap.md) | Product direction |

---

## Pipeline overview

```text
CT (Visible Human / NasalSeg)
  → process_whole_head / process_case
  → refine_nasal_ct + extend_nasal_to_tip
  → OpenFOAM import or potential-flow compute_flow
  → regenerate_curvy_pathlines   # wispy turbulent seeds
  → compute_surgical_guidance  # frontal paths + pink zones + treatments
  → Streamlit viewer
```

### Key scripts

| Script | Purpose |
|--------|---------|
| `process_whole_head.py` | Head mask, airway, BCs, initial flow |
| `refine_nasal_ct.py` | L/R cavities, septum, naris shell |
| `extend_nasal_to_tip.py` | Open vestibules to skin tip |
| `import_openfoam_results.py` | Foam `U` → CT grid |
| `regenerate_curvy_pathlines.py` | Turbulent pathlines JSON |
| `compute_surgical_guidance.py` | Sinuses, frontal paths, zones, treatments |
| `rebuild_skin_surface.py` | Cleaner skin STL |

### Key library modules (`src/sinus_cfd/`)

| Module | Role |
|--------|------|
| `flow_field.py` | Velocity field + curvy pathlines |
| `open_path.py` | Most-open geodesics, frontal corridors |
| `surgical_zones.py` | IT/MT/septum + treatment ranking |
| `sinus_anatomy.py` | Frontal/sphenoid/maxillary heuristics |
| `nasal_airway_ct.py` | CT naris / cavities / septum |
| `openfoam_import.py` | Foam → NPZ |

---

## Repository layout

```text
Sinus_CFD/
├── AGENTS.md                 # agent onboarding
├── README.md
├── app/viewer.py             # Streamlit UI
├── docs/                     # architecture, surgical, OpenFOAM, …
├── scripts/                  # CLI pipelines
├── src/sinus_cfd/            # Python package
├── foam/VisibleHuman_Head/   # OpenFOAM case
├── data/                     # local CTs (gitignored)
└── outputs/                  # generated (gitignored)
```

---

## NasalSeg (labeled FOV)

```powershell
py -3.12 scripts\download_nasalseg.py
py -3.12 scripts\process_case.py --case P001
py -3.12 scripts\compute_flow.py --case P001
```

Labels 1–3 = L/R nasal + nasopharynx (default continuous path). See [`docs/nnunet_nasal.md`](docs/nnunet_nasal.md) for Dataset501 scaffold.

---

## OpenFOAM (optional)

```powershell
py -3.12 scripts\export_openfoam_geometry.py --case VisibleHuman_Head
py -3.12 scripts\scaffold_openfoam_case.py --case VisibleHuman_Head
# Docker or WSL: Allrun / Allrun.docker
py -3.12 scripts\import_openfoam_results.py --case VisibleHuman_Head
```

See [`docs/docker_openfoam.md`](docs/docker_openfoam.md), [`docs/openfoam.md`](docs/openfoam.md).

---

## License & ethics

- Use de-identified, publicly licensed imaging for development.  
- Respect dataset licenses and citations.  
- **Research only** — not validated for clinical use.

### NasalSeg

```bibtex
@article{nasalseg,
  title={NasalSeg},
  # See Zenodo record 13893419 for citation details
}
```
