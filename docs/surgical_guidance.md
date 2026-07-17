# Surgical guidance (demo)

## Scripts

```powershell
py -3.12 scripts\compute_surgical_guidance.py --case VisibleHuman_Head
```

Depends on: flow field, nares, cavities, optional streamlines, CT HU crop for sinus air.

## Outputs

| File | Content |
|------|---------|
| `*_surgical_guidance.json` | Frontal paths, path metrics, removal_zones, treatments, notes |
| `*_treatment_recommendations.json` | Zones + treatments only |
| `*_sinus_anatomy.json` | Frontal / sphenoid / maxillary labels |
| `*_sinus_*.nrrd` / `.stl` | Sinus masks and surfaces |
| `*_removal_highlight.npz` | Combined points + per-zone arrays |
| `*_removal_{inferior_turbinate,middle_turbinate,septum}.nrrd` | Zone masks |

## Frontal instrument paths

Dual purple polylines in the viewer:

- **Left naris → left frontal**  
- **Right naris → right frontal**

Design goals:

- **Sagittal (y–z):** nearly straight (low RMS to chord)  
- **Coronal (x):** stay relatively **medial** early, then **slight lateral** flare superiorly (into each frontal half)

Implementation: `open_path.build_lateral_diverge_frontal_path` (prescribed x(t)/y(t)/z(t) + local air snap).

These are **planning corridors**, not endoscope kinematics or full FEM tissue models.

## High-|u| “areas to remove”

1. Build a corridor along **naris → trachea** (open-path + streamline samples).  
2. Keep voxels with elevated speed inside that corridor.  
3. Classify into anatomic zones (`surgical_zones.classify_removal_zones`):

| Zone key | Meaning (heuristic) |
|----------|---------------------|
| `inferior_turbinate` | Lateral + inferior — inferior meatus / maxillary corridor |
| `middle_turbinate` | Mid-height, para-septal to mid-lateral — splits airflow |
| `septum` | Near midplane, distal–medial nasal passage |

**Not** expert radiologist segmentation. Good enough for demo toggles and ranking.

## Treatment recommendations

`surgical_zones.recommend_treatments` ranks options by zone severity and invasiveness:

**Airflow (prefer least invasive first)**

- Inferior / middle turbinate reduction (RF or microdebrider)  
- Septoplasty (caudal/anterior or posterior/distal)  
- Nasal valve support  

**CRS / drainage**

- Balloon sinus dilation  
- Maxillary antrostomy  
- Frontal drillout (highest invasiveness; last-line framing)  

Recommendations are **heuristic demos**, not clinical decision support.

## Viewer integration

Sidebar checkboxes control frontal paths and each pink zone independently.  
Main page shows zone metrics and preferred treatments.
