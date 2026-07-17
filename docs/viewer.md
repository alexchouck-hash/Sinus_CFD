# Streamlit viewer

## Run

```powershell
cd <repo-root>
py -3.12 -m streamlit run app\viewer.py --server.address 127.0.0.1 --server.port 8501
```

Requires precomputed `outputs/<case>/*` (see `AGENTS.md` pipeline).

## Version

Displayed as `APP_VERSION` / `APP_VERSION_LABEL` in `app/viewer.py` (e.g. `0.15.1-…`).

Bump when:

- expected on-disk keys change  
- default visibility of layers changes  
- pathline / surgical JSON schema changes  

## Default case

Prefers `VisibleHuman_Head` if present under `outputs/`.

## Sidebar (surgical)

| Control | Effect |
|---------|--------|
| Paths to frontal sinuses (purple) | Dual L/R instrument corridors |
| Inferior turbinates (pink) | High-|u| IT / lateral corridor |
| Middle turbinates (pink) | High-|u| MT (flow split) |
| Septum distal–medial (pink) | High-|u| septal jet |
| Reload data | Clears Streamlit cache |

No dense display sliders: skin/cavity/pathline defaults are fixed for the demo.

## 3D layers

| Layer | Source files |
|-------|----------------|
| Skin | `*_skin.stl` |
| L/R cavity | `*_cavity_left.stl`, `*_cavity_right.stl` |
| Frontal sinus | `*_sinus_frontal.stl` |
| Pathlines | `*_streamlines.json` (+ optional speeds) |
| Frontal paths | `*_surgical_guidance.json` → `paths_mm.naris_*_to_frontal` |
| Removal zones | `*_removal_highlight.npz` keys `inferior_turbinate`, `middle_turbinate`, `septum` |
| Naris / trachea | `*_boundary_conditions.json` / `*_nares.json` |

### Plotly notes

- Prefer **hex color + `opacity=`** for 3D markers; `rgba(...)` often appears white.  
- Pathlines: multi-segment Scatter3d with decaying opacity/width (“wispy”).  
- No Turbo colorbar on the 3D scene (by design).

## Main panel sections

1. Tri-planar velocity (slice index sliders only)  
2. 3D plot  
3. **Identified areas to remove** (zone stats)  
4. **Recommended treatment options** (from surgical JSON)

## Cache

`@st.cache_data` on `load_case` uses a fingerprint of key output mtimes. After regenerating pathlines or surgical JSON, use **Reload data** or clear cache.
