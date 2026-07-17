# AGENTS.md — orientation for coding agents

Read this first when continuing work on **Sinus_CFD**. Prefer updating code + docs together when behavior changes.

## What this project is

Research prototype for **CT-based nasal / sinus airflow visualization and surgical planning demos**.

Primary demo case: **Visible Human Female head CT** (`VisibleHuman_Head`).

**Not a medical device.** Outputs are heuristics + CFD preview for research/education only.

## Repo map

| Path | Role |
|------|------|
| `app/viewer.py` | Streamlit 3D viewer (current UI entrypoint) |
| `src/sinus_cfd/` | Library modules (import via `sys.path` → `src`) |
| `scripts/` | CLI pipelines (process, OpenFOAM, pathlines, surgical) |
| `outputs/` | Generated masks, STL, flow, streamlines (**gitignored**) |
| `data/` | Local CT downloads (**gitignored** except `data/README.md`) |
| `foam/VisibleHuman_Head/` | OpenFOAM case (simpleFoam, converged ~181) |
| `docs/` | Technical notes (architecture, surgical, OpenFOAM, data) |

## Current demo state (as of viewer `0.15.x`)

- **Skin** surface + wireframe; L/R nasal cavities; optional frontal sinus mesh  
- **Turbulent wispy pathlines** seeded through airspace (+ denser / more swirl near nares), generally toward **trachea**  
- **Purple** dual instrument paths: L naris→L frontal, R naris→R frontal (medial then slight superior lateral flare)  
- **Pink** high-|u| “areas to remove”, split into:
  - inferior turbinate  
  - middle turbinate  
  - septum (distal–medial)  
- Sidebar toggles those layers; treatment recommendations below the 3D view  
- **OpenFOAM** velocity imported into `*_flow.npz` for the head case (not only potential flow)

## Canonical pipeline (Visible Human)

Run from repo root with Python 3.12:

```powershell
# 1) Whole-head process (if starting fresh)
py -3.12 scripts\process_whole_head.py --case VisibleHuman_Head

# 2) CT nasal L/R + septum refine
py -3.12 scripts\refine_nasal_ct.py --case VisibleHuman_Head

# 3) Force open tip vestibules (1 mm CT often seals nares)
py -3.12 scripts\extend_nasal_to_tip.py --case VisibleHuman_Head

# 4) Optional: OpenFOAM geometry + Docker/WSL solve, then:
py -3.12 scripts\import_openfoam_results.py --case VisibleHuman_Head

# 5) Turbulent pathlines (sinuses included in seed domain)
py -3.12 scripts\regenerate_curvy_pathlines.py --case VisibleHuman_Head

# 6) Frontal paths + pink zones + treatment JSON
py -3.12 scripts\compute_surgical_guidance.py --case VisibleHuman_Head

# 7) Viewer
py -3.12 -m streamlit run app\viewer.py --server.address 127.0.0.1 --server.port 8501
```

After data changes: hard-refresh browser; use sidebar **Reload data** if cache is stale.

## Important modules

| Module | Responsibility |
|--------|----------------|
| `whole_head.py` | Edge-aware tissue + airway path head→trachea |
| `nasal_airway_ct.py` | CT naris openings, L/R cavities, septum |
| `extend_nasal_to_tip.py` (script) | Paint vestibules tip→cavity |
| `flow_field.py` | Potential flow + **curvy pathlines** (trilinear, swirl, trachea attract) |
| `open_path.py` | Most-open geodesics, frontal instrument corridors, restriction tubes |
| `sinus_anatomy.py` | Heuristic frontal / sphenoid / maxillary labels |
| `surgical_zones.py` | IT / MT / septum classification + treatment ranking |
| `openfoam_import.py` | Map simpleFoam U onto CT grid |

## Key outputs (`outputs/VisibleHuman_Head/`)

| Pattern | Meaning |
|---------|---------|
| `*_flow.npz` | `airway, speed, ux, uy, uz, pressure, spacing, origin, inlet/outlet masks` |
| `*_streamlines.json` | `lines` + optional `speeds_m_s` (pathlines) |
| `*_nares.json` | Tip naris centers (`skin_tip_vestibule` preferred) |
| `*_surgical_guidance.json` | Frontal paths, zones, treatment list |
| `*_removal_highlight.npz` | Combined + `inferior_turbinate` / `middle_turbinate` / `septum` point clouds |
| `*_sinus_*.nrrd/stl` | Paranasal sinus masks/meshes |
| `*_cavity_{left,right}.*` | Nasal cavity L/R |
| `*_skin.stl` | Outer skin surface |

Large volumes and `outputs/**` are **not** in git. Agents must regenerate or use local data.

## Conventions

- **Coordinates:** physical mm; arrays often `(z, y, x)`; spacing/origin from SimpleITK/NRRD  
- **This CT:** `y_anterior_is_low=True`, `superior_is_high_z=True`, **high x ≈ patient left**  
- **Inspiration BCs:** both nares ≈ 50/50 inlets, trachea outlet, mouth closed  
- **Viewer version:** bump `APP_VERSION` / `APP_VERSION_LABEL` in `app/viewer.py` when UI or data layout changes  
- Prefer **hex colors + `opacity=`** for Plotly 3D markers (rgba fills can wash white)  
- Python **3.12**; repo root on `sys.path` via `src`  

## What not to do

- Do not commit DICOM/NRRD/STL under `data/` or `outputs/`  
- Do not treat heuristic IT/MT/septum zones or treatment text as clinical advice  
- Do not assume passage_lumen alone includes both L/R nares (historical left bias) — use cavities + tip extension  
- Do not add large OpenFOAM `polyMesh` binaries to git  

## Deeper docs

| Doc | Content |
|-----|---------|
| `docs/architecture.md` | Full system architecture + data flow (current implementation) |
| `docs/architecture_and_roadmap.md` | Field methodology, CFD quality bar, staged plan |
| `docs/product_roadmap.md` | Product milestones |
| `docs/viewer.md` | Viewer layers and toggles |
| `docs/surgical_guidance.md` | Zones, frontal paths, treatments |
| `docs/data-sources.md` | NasalSeg vs Visible Human |
| `docs/docker_openfoam.md` | OpenFOAM via Docker |
| `docs/open_paths.md` | Most-open path algorithm |
| `README.md` | Human-facing quick start |

## Git

- Default branch: **`main`**  
- Remote: `https://github.com/alexchouck-hash/Sinus_CFD`  
- Feature work historically lived on `feature/curvy-volume-pathlines` (merged into main)
