# Architecture (current)

## Purpose

Sinus_CFD reconstructs nasal and paranasal airspaces from CT, estimates or imports airflow, visualizes **turbulent-looking pathlines**, **instrument corridors** (naris → frontal), and **high-velocity tissue targets** with least-invasive surgical suggestions for a research demo.

```text
                    ┌─────────────────┐
   CT (Visible Human / NasalSeg)      │
                    └────────┬────────┘
                             ▼
              process_whole_head / process_case
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   tissues/skin          airway/lumen         BCs (nares, trachea)
         │                   │                   │
         ▼                   ▼                   ▼
   refine_nasal_ct     extend_nasal_to_tip    physiology Q
   sinus_anatomy            │                   │
         │                  ▼                   ▼
         │            OpenFOAM optional ──► import_openfoam_results
         │            or potential flow ──► compute_flow
         │                  │
         │                  ▼
         │         regenerate_curvy_pathlines
         │                  │
         └────────► compute_surgical_guidance
                             │
                             ▼
                      app/viewer.py (Streamlit)
```

## Coordinate system (Visible Human demo)

| Convention | Value |
|------------|--------|
| Array order | `(z, y, x)` |
| Spacing | typically 1 mm isotropic on the cropped grid |
| Anterior | **low y** (`y_anterior_is_low=True`) |
| Superior | **high z** (`superior_is_high_z=True`) |
| Patient left | **high x** |

Port centers and pathlines are stored in **physical mm** using `origin_xyz` + `index * spacing`.

## Python package (`src/sinus_cfd`)

| Module | Role |
|--------|------|
| `pipeline.py` | NasalSeg case: labels → mask → mesh |
| `whole_head.py` | Visible Human: edge segmentation, head shell, airway bridge |
| `edge_segment.py` | Body/skin edges, tip landmarks |
| `tissues.py` | Soft tissue / bone helpers |
| `skin_and_nares.py` | Skin shell mesh, external naris projection |
| `nasal_airway_ct.py` | CT naris shell, L/R cavity, septum |
| `nasal_passage.py` | Passage lumen, walls, centerline metrics |
| `physiology.py` | VT, RR → mean inspiratory Q |
| `boundary_conditions.py` | Ports JSON, OpenFOAM BC notes |
| `flow_field.py` | Laplace/potential velocity; **curvy pathline integration** |
| `open_path.py` | EDT most-open paths; frontal corridors; restriction tubes |
| `sinus_anatomy.py` | Heuristic frontal / sphenoid / maxillary |
| `surgical_zones.py` | IT / MT / septum classification; treatment ranking |
| `openfoam_export.py` | Watertight solid + patch STL export |
| `openfoam_import.py` | Map foam `U` onto CT grid + streamlines |

Scripts under `scripts/` are thin CLIs over these modules.

## Flow field

### OpenFOAM path (preferred for Visible Human)

1. `export_openfoam_geometry.py` — solid air body + patches  
2. `scaffold_openfoam_case.py` / Docker or WSL `Allrun`  
3. `simpleFoam` → time directory (e.g. `181/`)  
4. `import_openfoam_results.py` → `*_flow.npz` + streamlines  

### Preview path

`compute_flow.py` → potential / Darcy solve on the airway mask, scaled to ~18 L/min.

### Pathlines (viewer)

`regenerate_curvy_pathlines.py` → `compute_curvy_volume_pathlines`:

- Seeds: **volume throughout** air + cavities + sinuses; denser cloud at nares  
- Integration: trilinear U, RK2, **helical swirl** (stronger near nares), soft **trachea attract**  
- Length scales with local seed speed  
- Viewer draws **wispy** segments (opacity/width fade along path); no Turbo colorbar on 3D  

## Surgical guidance

`compute_surgical_guidance.py`:

1. Label paranasal sinuses (`sinus_anatomy.py`)  
2. Dual **instrument paths** naris → ipsilateral frontal (`build_lateral_diverge_frontal_path`: stay medial early, slight lateral flare superiorly)  
3. High-|u| mask along naris→trachea corridor  
4. Split into **inferior_turbinate / middle_turbinate / septum** (`surgical_zones.py`)  
5. Rank **treatment options** (turbinate RF/microdebrider, septoplasty, valve, balloon, antrostomy, frontal drillout)

Outputs: `*_surgical_guidance.json`, `*_treatment_recommendations.json`, `*_removal_*.nrrd`, `*_removal_highlight.npz`.

## Viewer (`app/viewer.py`)

| Layer | Default |
|-------|---------|
| Skin + wireframe | On |
| L/R cavities | On |
| Frontal sinus mesh | On |
| Turbulent pathlines | On |
| Purple frontal paths | Toggle (sidebar) |
| Pink IT / MT / septum | Toggle (sidebar) |
| Treatment recommendations | Panel below 3D |

`APP_VERSION` is the source of truth for UI generation; bump it when data layout or default behavior changes.

## Data products vs git

| Tracked | Not tracked |
|---------|-------------|
| Source, scripts, docs, foam *templates* | `data/**` volumes, `outputs/**`, large STL/NRRD |
| `requirements.txt` | OpenFOAM `polyMesh` dumps if huge |

Agents and developers must run the pipeline locally to regenerate outputs.

## Related docs

- [architecture_and_roadmap.md](architecture_and_roadmap.md) — methodology, CFD/mesh guidance, staged plan  
- [product_roadmap.md](product_roadmap.md) — product milestones  
- [viewer.md](viewer.md)  
- [surgical_guidance.md](surgical_guidance.md)  
- [data-sources.md](data-sources.md)  
- [docker_openfoam.md](docker_openfoam.md)  
- [open_paths.md](open_paths.md)  
