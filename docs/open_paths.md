# Most-open dual centerlines (demo quality)

Also see:

- **Turbulent viewer pathlines:** `scripts/regenerate_curvy_pathlines.py`, `flow_field.py`  
- **Frontal instrument corridors + high-|u| zones:** `docs/surgical_guidance.md`, `compute_surgical_guidance.py`  

## Question

Is there an algorithm for centerlines from nares → trachea that prefer the
**most open** path and stay roughly **left/right symmetric**?

**Yes.** Classic approach used in airway/vessel centerlining:

1. **Distance-to-wall** \(r = \mathrm{EDT}(\mathrm{air})\) (radius of largest sphere inside the lumen).  
2. **Cost** on air voxels: \(c = 1/(r+\varepsilon)^{p}\) (wide lumen = cheap).  
3. **Shortest path** (discrete geodesic) from each naris seed to the trachea.  
4. **Soft symmetry**: blend each path with the midplane mirror of the other.  
5. **Open space**: air inside tubes around both paths (local radius from \(r\)).

Implemented in `src/sinus_cfd/open_path.py`.

## Run

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
py -3.12 scripts\demo_open_paths.py --case VisibleHuman_Head --symmetry 0.35
```

| Flag | Meaning |
|------|---------|
| `--symmetry 0..1` | 0 = independent paths; ~0.3–0.4 = soft symmetry (demo default) |
| `--power 2` | How strongly paths prefer wide lumen |
| `--domain cavity_union` | Navigate in L∪R cavity masks (default) |

## Outputs

- `outputs/<case>/<case>_open_paths.json` — path meta  
- `passage.json` → `centerline_left_mm`, `centerline_right_mm` (viewer magenta)  
- `*_open_space.nrrd` / `.stl` — domain inferred from dual tubes  
- Updates `airway_mask` / `passage_lumen` for demo domain  

## Viewer

Reload Streamlit (**Clear cache**). Version `0.8.x` draws **two magenta** centerlines.
Septum highlight stays off by default.

## Relation to nnU-Net

This path algorithm is **classical** and good for demos. A trained nnU-Net improves
the **air mask**; the same most-open dual-path step still applies on top of NN labels.
