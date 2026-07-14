# Tissue classes (whole-head CT)

Long-term Sinus_CFD models anatomy as separate materials. Current HU heuristics:

| ID | Class | Typical HU (approx) | Role |
|----|--------|---------------------|------|
| 0 | exterior | outside body | not meshed as tissue |
| 1 | **air** | ≤ −300 (default) | fluid domain (airflow) |
| 2 | **soft tissue** | body default | solid head shell, mucosa wall |
| 3 | **cartilage** | ~80–300 (rough) | future structural / nasal valve |
| 4 | **bone** | ≥ 300 | skull, turbinates, septum |

## Outputs

`process_whole_head.py` writes:

- `*_tissues.nrrd` — multi-label volume  
- `*_head_mask.nrrd` / `*_head.stl` — solid body shell  
- `*_airway_mask.nrrd` / `*_airway.stl` — air lumen (nostrils → trachea)  
- `*_bone_mask.nrrd` / `*_bone.stl` — bone (when present)  
- `*_soft_tissue_mask.nrrd`

## Airway direction

For whole-head CT, orientation is taken from the image z-axis (DICOM LPS: +Z superior when applicable). The fluid path is constrained **caudally** (toward the neck / trachea), not into the cranial vault.

## Future

- ML segmentation (nnU-Net) for mucosa, turbinates, ostia, cartilage  
- Separate material properties for CFD / FEA  
- Patient-specific calibration of HU windows  
