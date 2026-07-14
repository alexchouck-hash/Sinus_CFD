# Nasal passage domain for airflow

## Goal

Simulate inspiratory flow **nostrils → nasal cavity → pharynx → trachea** with
explicit boundaries:

| Region | Role | CFD treatment |
|--------|------|----------------|
| **Lumen** | Air volume inside the passage | Fluid domain |
| **Wall** | Mucosa / tissue interface | No-slip |
| **Inlet open** | External nares | Prescribed volumetric flow |
| **Outlet open** | Trachea | Fixed gauge pressure |
| **Mouth** | Closed | Not in domain |

## How boundaries are built

1. Start from connected **airway mask** (HU/edge air + caudal path).
2. Snap naris/trachea ports to nearest lumen voxels; dilate → **open port** masks.
3. **Wall** = lumen surface voxels that are not open ports.
4. **Centerline** = medial-axis-biased geodesic from nares to trachea.
5. **Cross-sections** ≈ π r² with r = distance-to-wall along the centerline.

## Commands

```powershell
# After whole-head process (masks + BCs exist):
py -3.12 scripts\analyze_passage.py --case VisibleHuman_Head
```

Writes under `outputs/<case>/`:

| File | Content |
|------|---------|
| `*_passage_lumen.nrrd` | Fluid domain |
| `*_passage_wall.nrrd` | Mucosa wall voxels |
| `*_passage_inlet_open.nrrd` | Nares open BC |
| `*_passage_outlet_open.nrrd` | Trachea open BC |
| `*_passage_surface.stl` | Outer surface of lumen |
| `*_passage_wall.stl` | Wall shell (viz) |
| `*_passage.json` | Centerline, areas, metrics |

Flow is re-solved with streamlines seeded from **inlets + centerline**.

## Viewer

App version **0.5.0-nasal-passage** shows:

- Passage metrics in the data-version panel  
- **Magenta centerline** in 3D  
- Dense velocity cones inside the lumen  

## Physics note

Current solver is still **potential / Darcy flow** (Laplace pressure), scaled to
~18 L/min mean inspiration. It respects the passage geometry and open ports, but
is not full Navier–Stokes. OpenFOAM (or similar) is the next step on the same
wall/open-port definition.

## Next upgrades

- Open mesh ends as explicit STL patches for OpenFOAM  
- Separate left/right nasal meatus branches  
- Mucosal roughness / resistance maps  
- Transient breath waveform along the centerline  
