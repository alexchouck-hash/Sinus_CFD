# Data directory

Place downloaded CT volumes here. This folder’s contents (except this README) are gitignored.

## 1. NasalSeg (labeled nasal / sinus FOV)

Best for **segmentation labels** (L/R cavity, nasopharynx, maxillary sinuses).  
Not a whole-head FOV.

- https://zenodo.org/records/13893419  

```powershell
Invoke-WebRequest -Uri "https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1" -OutFile "NasalSeg.zip"
Expand-Archive -Path "NasalSeg.zip" -DestinationPath "NasalSeg"
```

```text
NasalSeg/
  images/P001_img.nrrd … P130_img.nrrd
  labels/P001_seg.nrrd … P130_seg.nrrd
```

```powershell
py -3.12 scripts\process_case.py --case P001
```

## 2. Visible Human Female — whole head CT (1 mm)

Full head (skull, sinuses, airways, soft tissue). **No expert nasal labels.**  
NLM Visible Human Project via Harvard Dataverse `doi:10.7910/DVN/3JDZCT`.

```powershell
# From repo root (~120 MB, ~234 DICOM slices → NRRD)
py -3.12 scripts\download_visible_human_head.py
```

```text
VisibleHuman_Head/
  dicom/VHFCT1mm-Head (1).dcm …
  VHFCT1mm_Head.nrrd
  VHFCT1mm_Head_preview.png
  manifest.json
```

Cite: Visible Human Project CT Datasets, Harvard Dataverse, doi:10.7910/DVN/3JDZCT.

See `docs/data-sources.md` for TCIA and other archives.
