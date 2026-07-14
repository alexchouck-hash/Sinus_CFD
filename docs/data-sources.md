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

**nnU-Net training scaffold (this repo):**

```powershell
py -3.12 scripts\download_nasalseg.py
py -3.12 scripts\prepare_nnunet_nasalseg.py
```

See **`docs/nnunet_nasal.md`**.

## Full head: Visible Human Project CT (recommended whole-head)

| | |
|---|---|
| **What** | NLM Visible Human Female 1 mm head CT (~234 slices) |
| **Why** | Whole head: skull, sinuses, nasal airway, soft tissue, path toward trachea |
| **Harvard Dataverse** | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3JDZCT |
| **Iowa notes** | https://mri.medicine.uiowa.edu/equipment-information/scanner-images/visible-human-project-ct-datasets |
| **Local download** | `py -3.12 scripts/download_visible_human_head.py` → `data/VisibleHuman_Head/` |

NasalSeg is a **cropped nasal/sinus FOV with labels**. Visible Human is a **full head without nasal labels** — complementary.

## Which anatomy for the demo?

| Goal | Prefer | Why |
|------|--------|-----|
| **Accurate L/R nasal cavity + maxillary sinuses** | **NasalSeg** (labels) | Expert pixel labels; no guessing air vs mucosa |
| **Whole path nostrils → pharynx → trachea** | **Visible Human head** | Only full FOV with caudal airway |
| **Best of both (longer term)** | nnU-Net on NasalSeg → apply to head CTs | Auto bone / soft tissue / air / nasal classes |

**Recommendation for CFD demo quality today:**

1. Keep **Visible Human** for *anatomy context* (skin, skull outline, trachea).  
2. Trust **CT air thresholding + CT naris opening shell** only as a bootstrap — it is weak at 1 mm partial-volume nares.  
3. Prefer **NasalSeg-labeled cases** when you need correct nasal airways without fighting HU thresholds.  
4. For production patient CTs: **auto-segment** (nnU-Net / TotalSegmentator-style) into bone / soft tissue / air / nasal lumen, then place ports on labeled openings.

Rough physiology for the viewer: **inspiration ~50% left naris + 50% right naris → single trachea outlet** (adjustable later for septal deviation / NAO).

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
