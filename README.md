# Sinus_CFD

**Computational fluid dynamics of nasal airflow and sinus drainage from head CT anatomy.**

This project aims to take a CT scan of a human head, reconstruct the nasal cavity and paranasal sinuses, and simulate how **airflow** and **drainage** change when the anatomy is modified (e.g., septoplasty, turbinate reduction, ostium enlargement, or other structural interventions).

## Goals

1. **Ingest** head / sinus CT volumes (DICOM or NIfTI)
2. **Segment** airways and sinuses (air, bone, soft tissue; optional labeled nasal structures)
3. **Build** 3D surface / volume meshes suitable for CFD
4. **Simulate** inspiratory/expiratory airflow and simple drainage / clearance proxies
5. **Compare** baseline vs. surgically (or virtually) altered anatomy

## Intended pipeline (high level)

```
CT scan (DICOM/NIfTI)
    → preprocessing (windowing, resampling)
    → segmentation (airway / sinus masks)
    → surface extraction + mesh quality cleanup
    → CFD setup (inlet/outlet BCs, fluid properties)
    → solve (OpenFOAM / similar)
    → post-process (ΔP, flow rates, wall shear, stagnation zones)
    → virtual surgery variants → re-run → compare
```

## Sample CT data (public)

Large medical volumes are **not** stored in this repository. Download sample data separately into `data/` (gitignored).

### Best fit for this project: NasalSeg

| | |
|---|---|
| **What** | 130 CT scans of nasal cavity & paranasal sinuses with labels (L/R nasal cavity, nasopharynx, L/R maxillary sinus) |
| **Why** | Directly targets the anatomy this project needs |
| **Size** | ~224 MB |
| **Download** | [Zenodo – NasalSeg](https://zenodo.org/records/13893419) |
| **Also** | [GitHub – NasalSeg](https://github.com/YichiZhang98/NasalSeg) |

```text
# Example (from project root)
mkdir -p data
# Then download NasalSeg.zip from Zenodo into data/ and unzip
```

### Full head CT: Visible Human Project

| | |
|---|---|
| **What** | Classic full-body / head CT from the NLM Visible Human Project |
| **Why** | Complete head geometry if you want skull-to-airway context |
| **Download** | [Iowa MRRF – Visible Human CT](https://mri.medicine.uiowa.edu/equipment-information/scanner-images/visible-human-project-ct-datasets) (files on Harvard Dataverse) |

### Other head / neck CT archives

- **[TCIA Head-Neck-PET-CT](https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/)** – 298 patients, planning CT + PET (large; needs TCIA Data Retriever)
- **[TCIA HNSCC](https://www.cancerimagingarchive.net/collection/hnscc/)** – head & neck squamous cell carcinoma CTs
- **[Stanford AIMI SinoCT](https://aimi.stanford.edu/data)** – large head CT collection (research use; registration may be required)
- **[CT-SCOPE](https://pmc.ncbi.nlm.nih.gov/articles/PMC12398895/)** – paranasal sinus CT with osseous structure annotations

### Quick single-study DICOM samples

For pipeline smoke tests (not always sinus-focused):

- [Saga IT free DICOM samples](https://saga-it.com/dicom/samples) (CC-BY)
- [NCI Imaging Data Commons](https://portal.imaging.datacommons.cancer.gov/) – public cloud access, no egress fees for many collections

## Recommended starter path

1. Download **NasalSeg** (small, sinus-labeled).
2. Load one volume with [3D Slicer](https://www.slicer.org/) or Python (`nibabel` / `pydicom` + `SimpleITK`).
3. Extract the air mask (threshold Hounsfield units ≈ −1000 to −500 for air, then refine).
4. Export STL/OBJ of the airway lumen for meshing.
5. Run a simple laminar/incompressible flow case in OpenFOAM or equivalent.

## Repository layout

```text
Sinus_CFD/
├── README.md
├── data/                 # local CT downloads (not committed)
├── docs/                 # design notes, citations
└── src/                  # processing & simulation scripts (to be added)
```

## License & ethics

- Use only **de-identified, publicly licensed** imaging for development.
- Respect each dataset’s license and citation requirements.
- This software is for research and educational exploration; it is **not** a medical device and must not be used for clinical decision-making without appropriate validation and regulatory clearance.

## Citation tips

If you use NasalSeg, cite the NasalSeg paper / Zenodo record. For TCIA collections, cite the collection DOI listed on each TCIA collection page.

## Status

Project scaffold — data pointers and goals only. Segmentation, meshing, and CFD tooling still to be added.
