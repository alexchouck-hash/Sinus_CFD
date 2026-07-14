# Sinus_CFD

**Computational fluid dynamics of nasal airflow and sinus drainage from head CT anatomy.**

This project aims to take a CT scan of a human head, reconstruct the nasal cavity and paranasal sinuses, and simulate how **airflow** and **drainage** change when the anatomy is modified (e.g., septoplasty, turbinate reduction, ostium enlargement, or other structural interventions).

## Goals

1. **Ingest** head / sinus CT volumes (DICOM, NRRD, or NIfTI)
2. **Segment** airways and sinuses (expert labels and/or HU air threshold)
3. **Build** 3D surface / volume meshes suitable for CFD
4. **Simulate** inspiratory/expiratory airflow and simple drainage / clearance proxies
5. **Compare** baseline vs. surgically (or virtually) altered anatomy

## Current status

| Step | Status |
|------|--------|
| Public CT data (NasalSeg) | Working — download into `data/` |
| Airway mask from labels / HU | Working (`scripts/process_case.py`) |
| Surface export (STL) | Working |
| Volume mesh + OpenFOAM CFD | Not yet |
| Virtual surgery variants | Not yet |

### First case result (NasalSeg `P001`)

- **Mask:** expert labels 1–3 (L/R nasal cavity + nasopharynx), cleaned to largest component  
- **Airway volume:** ~28.5 mL  
- **Surface:** ~30k vertices / ~60k faces (open tube → not watertight; expected for CFD inlets/outlets)  
- **Outputs (local):** `outputs/P001/P001_airway.stl`, `P001_airway_mask.nrrd`, `P001_preview.png`, `P001_stats.json`

## Quick start

### 1. Install dependencies

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
py -3.12 -m pip install -r requirements.txt
```

### 2. Download NasalSeg (once)

```powershell
cd data
# If not already downloaded:
# Invoke-WebRequest -Uri "https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1" -OutFile "NasalSeg.zip"
# Expand-Archive NasalSeg.zip -DestinationPath NasalSeg
```

Layout after unzip:

```text
data/NasalSeg/
  images/P001_img.nrrd … P130_img.nrrd
  labels/P001_seg.nrrd … P130_seg.nrrd
```

### 3. Process one case → mask + STL

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
py -3.12 scripts\process_case.py --case P001
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--mask-source labels` | Expert labels (default; recommended) |
| `--mask-source hu` | HU air threshold only |
| `--mask-source labels_and_hu` | Intersection of both |
| `--include-sinuses` | Also include maxillary sinuses (labels 4–5) |
| `--case P010` | Another subject |

### 4. Inspect results

- **Preview PNG:** open `outputs/P001/P001_preview.png`  
- **STL:** open in [3D Slicer](https://www.slicer.org/), MeshLab, or Blender  
- **Stats JSON:** voxel counts, spacing, mesh size  

## Pipeline (high level)

```
CT (NRRD/DICOM/NIfTI)
    → load + spacing/origin
    → airway mask (labels and/or HU ≈ −1024…−400)
    → morphological clean + largest component
    → marching cubes surface (STL)
    → [next] volume mesh + CFD (OpenFOAM / similar)
    → [next] virtual anatomy edits → re-run → compare
```

NasalSeg label map:

| ID | Structure |
|----|-----------|
| 1 | Left nasal cavity |
| 2 | Right nasal cavity |
| 3 | Nasopharynx |
| 4 | Left maxillary sinus |
| 5 | Right maxillary sinus |

Default CFD airway uses **1–3** (continuous nasal path). Use `--include-sinuses` for drainage-oriented studies.

## Sample CT data (public)

Large medical volumes are **not** stored in this repository. Keep them under `data/` (gitignored).

### Best fit: NasalSeg

| | |
|---|---|
| **What** | 130 CT scans with nasal/paranasal labels |
| **Size** | ~224 MB |
| **Download** | [Zenodo](https://zenodo.org/records/13893419) · [GitHub](https://github.com/YichiZhang98/NasalSeg) |

### Other sources

- [Visible Human Project CT](https://mri.medicine.uiowa.edu/equipment-information/scanner-images/visible-human-project-ct-datasets) — full head  
- [TCIA Head-Neck-PET-CT](https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/)  
- [TCIA HNSCC](https://www.cancerimagingarchive.net/collection/hnscc/)  
- [Stanford AIMI SinoCT](https://aimi.stanford.edu/data)  
- See `docs/data-sources.md` for more detail  

## Repository layout

```text
Sinus_CFD/
├── README.md
├── requirements.txt
├── data/                 # local CT downloads (not committed)
├── docs/                 # design notes, data sources
├── outputs/              # masks, STL, previews (not committed)
├── scripts/
│   └── process_case.py   # CLI entry point
└── src/sinus_cfd/
    └── pipeline.py       # load → mask → surface
```

## License & ethics

- Use only **de-identified, publicly licensed** imaging for development.
- Respect each dataset’s license and citation requirements.
- This software is for research and educational exploration; it is **not** a medical device and must not be used for clinical decision-making without appropriate validation and regulatory clearance.

## Citation

If you use NasalSeg:

```bibtex
@article{zhang2024nasalseg,
  title={NasalSeg: A Dataset for Automatic Segmentation of Nasal Cavity and Paranasal Sinuses from 3D CT Images},
  author={Zhang, Yichi and Wang, Jing and Pan, Tan and Jiang, Quanling and Ge, Jingjie and Guo, Xin and Jiang, Chen and Lu, Jie and Zhang, Jianning and Liu, Xueling and others},
  journal={Scientific Data},
  volume={11},
  number={1},
  pages={1--5},
  year={2024},
  publisher={Nature Publishing Group}
}
```
