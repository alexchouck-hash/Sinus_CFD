# Public CT data sources for Sinus_CFD

## Primary recommendation: NasalSeg

Ideal for this project: **130** 3D CT volumes with pixel-wise labels for:

- left / right nasal cavity  
- nasopharynx  
- left / right maxillary sinus  

| Field | Value |
|-------|--------|
| Size | ~224 MB |
| Zenodo | https://zenodo.org/records/13893419 |
| GitHub | https://github.com/YichiZhang98/NasalSeg |

These labels can seed airway segmentation and virtual-surgery experiments without starting from raw unlabeled CT alone.

## Full head: Visible Human Project CT

- University of Iowa overview: https://mri.medicine.uiowa.edu/equipment-information/scanner-images/visible-human-project-ct-datasets  
- Useful when you need complete craniofacial context beyond the sinonasal region.

## Large research archives

| Resource | Notes | Link |
|----------|--------|------|
| TCIA Head-Neck-PET-CT | 298 patients; planning CT | https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/ |
| TCIA HNSCC | Head & neck SCC imaging | https://www.cancerimagingarchive.net/collection/hnscc/ |
| Stanford AIMI SinoCT | Thousands of head CTs; research terms | https://aimi.stanford.edu/data |
| CT-SCOPE | Paranasal sinus osseous annotations | https://pmc.ncbi.nlm.nih.gov/articles/PMC12398895/ |
| Imaging Data Commons | Browser + bulk public download | https://portal.imaging.datacommons.cancer.gov/ |

TCIA downloads typically require the **[NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)**.

## Notes for CFD readiness

Prefer volumes that:

1. Cover the **nose, sinuses, and nasopharynx** (not brain-only slabs).  
2. Have **thin slices** (≤1 mm if possible) for clean surface meshes.  
3. Are **non-contrast** or soft-tissue/bone reconstructions suitable for air thresholding.  
4. Are clearly **licensed** for your use case (research vs. commercial).

## Do not commit

Never push real patient identifiers or large DICOM trees into GitHub. Keep volumes under `data/` and document the public URL + citation instead.
