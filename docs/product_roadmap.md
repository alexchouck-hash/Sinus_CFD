# Sinus_CFD product roadmap

## Vision

A clinical decision-support app where surgeons and patients can **upload a head CT**, receive **structured airway/sinus analysis**, and **interactively explore** airflow and drainage under baseline and virtually modified anatomy.

**Not a medical device yet.** All current outputs are research prototypes.

---

## Viewer (current → near term)

| Capability | Status |
|------------|--------|
| Tri-planar velocity with sliders | **Now** (Streamlit) |
| 3D semi-transparent cavity | **Now** |
| Curved streamlines + optional velocity cones | **Now** (potential-flow field) |
| Full Navier–Stokes CFD (OpenFOAM) | Next |
| Patient-specific VT / RR / L–R split | Partial (CLI + physiology module) |

### Run the viewer

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
py -3.12 scripts\process_case.py --case P001
py -3.12 scripts\compute_flow.py --case P001
py -3.12 -m streamlit run app\viewer.py
```

---

## Planned clinical modules

### 1. Pathway analysis (shortest path)

- Shortest path **naris → sinus ostium → sinus lumen** (geodesic in air mask / surface)
- Compare left vs right, pre- vs post–virtual ostium widening
- Metrics: path length, min cross-section along path, tortuosity

### 2. Mucus / drainage

- Mucociliary transport **out** of sinuses (surface advection toward ostium / nasopharynx)
- Scenario: **widened ostium** → recompute drainage proxy and residence time
- Visualization: particle tracks on mucosal surface (distinct from bulk air streamlines)

### 3. Imaging diagnosis assists

| Code | Condition | Imaging / geometry cues (high level) |
|------|-----------|--------------------------------------|
| **CRS** | Chronic rhinosinusitis | Mucosal thickening, opacification, ostiomeatal obstruction patterns |
| **NAO** | Nasal airway obstruction | Septal deviation, turbinate hypertrophy, minimal cross-sectional area, elevated resistance |
| **NVC** | Nasal valve collapse | Narrow internal/external valve angle, dynamic cues if available |
| **Polyps** | Nasal polyps | Soft-tissue masses in nasal cavity / sinuses (ML segmentation) |

These will require **labeled training data**, radiologist review, and regulatory pathway before clinical use.

### 4. Upload & report workflow (long-term)

1. Secure CT upload (DICOM de-identification)  
2. Auto segmentation (airway, sinuses, septum, turbinates, ostia)  
3. CFD / potential-flow + pathway + drainage metrics  
4. Interactive viewer (this app)  
5. Optional virtual surgery (septoplasty, turbinate reduction, ostium enlargement)  
6. Structured report for surgeon–patient discussion  

---

## Boundary condition policy (locked intent)

- **Inlets:** both nostrils  
- **Outlet:** trachea (proxy until FOV includes trachea)  
- **Mouth:** closed  
- **Flow:** typical tidal inhale sustained for typical \(T_i\); later patient-matched  

See `docs/boundary_conditions.md`.

---

## Technical stack (current prototype)

| Layer | Choice |
|-------|--------|
| Segmentation / mesh | SimpleITK, scikit-image, trimesh |
| Flow (preview) | Voxel Laplace / potential flow |
| Flow (target) | OpenFOAM or equivalent |
| Viewer | Streamlit + Plotly |
| Data | NasalSeg + future clinical DICOM |

---

## Disclaimer

Educational / research software only. **Not for clinical diagnosis or treatment decisions** without validated models, clinical studies, and appropriate regulatory clearance.
