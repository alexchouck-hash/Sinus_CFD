# Data directory

Place downloaded CT volumes here. This folder’s contents (except this README) are gitignored.

## NasalSeg (recommended)

- https://zenodo.org/records/13893419  
- Direct zip: https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1  

```powershell
# From this directory (PowerShell)
Invoke-WebRequest -Uri "https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1" -OutFile "NasalSeg.zip"
Expand-Archive -Path "NasalSeg.zip" -DestinationPath "NasalSeg"
```

Expected layout:

```text
NasalSeg/
  images/P001_img.nrrd … P130_img.nrrd
  labels/P001_seg.nrrd … P130_seg.nrrd
```

Then from the repo root:

```powershell
py -3.12 scripts\process_case.py --case P001
```

See the project root `README.md` and `docs/data-sources.md` for other public head CT sources.
