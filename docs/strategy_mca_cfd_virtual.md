# Strategy track: MCA · CFD metrics · virtual IT

Branch: `feature/mca-cfd-metrics-virtual-surgery`

Implements the first three engineering priorities from
[`architecture_and_roadmap.md`](architecture_and_roadmap.md):

1. **Geometry (no CFD)** — dual-side CSA profiles + MCA  
2. **CFD summary** — ΔP, resistance, L/R inlet allocation from `flow.npz`  
3. **Virtual surgery loop** — inferior turbinate reduction → pre/post compare  

## Scripts

```powershell
# 1) MCA / CSA
py -3.12 scripts\compute_geometry_metrics.py --case VisibleHuman_Head

# 2) OpenFOAM / flow metrics
py -3.12 scripts\compute_cfd_metrics.py --case VisibleHuman_Head

# 3) Virtual IT reduction (edits lumen, recomputes geometry + potential flow)
py -3.12 scripts\run_virtual_it_reduction.py --case VisibleHuman_Head --shave-mm 2.0
```

## Outputs

| File | Content |
|------|---------|
| `{case}_geometry_metrics.json` | L/R CSA samples, MCA xyz/mm², notes |
| `{case}_cfd_metrics.json` | ΔP, R, L/R fractions, speed stats |
| `{case}_virtual_IT_compare.json` | Pre/post MCA (+ CFD if both available) |
| `outputs/{case}_virtual_IT/` | Edited masks, optional potential flow |

## Modules

| Module | Role |
|--------|------|
| `geometry_metrics.py` | Plane CSA along centerlines; MCA |
| `cfd_metrics.py` | Port/band pressure + flux probes |
| `virtual_surgery.py` | Lateral-inferior lumen expansion (IT) |

## Caveats (research)

- MCA / CSA are **geometric** estimates (plane voxel count + EDT).  
- CFD ΔP uses the **mapped** `p` field; units may be kinematic pressure. Prefer **|ΔP|** and **R_abs** for magnitude.  
- L/R flux is a **local |U| probe**, then scaled to target Q (~18 L/min).  
- Virtual IT is a **heuristic mask edit**, not a validated resection. Virtual flow is **potential** unless OpenFOAM is re-exported and re-solved.  
- **Not a medical device.**

## Viewer

`app/viewer.py` **0.16.x** shows MCA metric + CSA curves, CFD row, MCA 3D markers, and virtual IT expander when JSON files exist.
