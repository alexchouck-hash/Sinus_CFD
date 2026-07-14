# Data directory

Place downloaded CT volumes here. This folder’s contents (except this README) are gitignored.

## Recommended first download

**NasalSeg** — nasal cavity & paranasal sinus CTs with labels:

- https://zenodo.org/records/13893419  
- Direct zip: https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1  

```powershell
# From this directory (PowerShell)
Invoke-WebRequest -Uri "https://zenodo.org/records/13893419/files/NasalSeg.zip?download=1" -OutFile "NasalSeg.zip"
Expand-Archive -Path "NasalSeg.zip" -DestinationPath "NasalSeg"
```

See the project root `README.md` for other public head CT sources (Visible Human, TCIA, etc.).
