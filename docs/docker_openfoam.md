# OpenFOAM with Docker (recommended on Windows)

You do **not** need a full WSL OpenFOAM install if Docker Desktop is available.

## Why Docker?

| Approach | Pros | Cons |
|----------|------|------|
| **Docker** | Isolated, no reboot dance once Docker works, same image everywhere | Docker Desktop install + resources |
| WSL apt OpenFOAM | Native Linux feel | Needs WSL distro ready (often a reboot) |

## One-time: install Docker Desktop

```powershell
winget install --id Docker.DockerDesktop -e --source winget
```

Then:

1. Start **Docker Desktop** from the Start menu.
2. Wait until it says **Docker is running**.
3. Open a **new** PowerShell window (so `docker` is on PATH).
4. Accept the license / finish first-run wizard if asked.

WSL 2 backend is used by Docker Desktop automatically (already partially installed on this machine).

## Run the nasal CFD case

```powershell
cd C:\Users\houck\Documents\Sinus_CFD

# Ensure geometry + case exist
py -3.12 scripts\export_openfoam_geometry.py --case VisibleHuman_Head
py -3.12 scripts\scaffold_openfoam_case.py --case VisibleHuman_Head

# Pull OpenFOAM image and run blockMesh → snappy → simpleFoam
powershell -ExecutionPolicy Bypass -File scripts\run_openfoam_docker.ps1
```

Interactive shell (debug mesh):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_openfoam_docker.ps1 -ShellOnly
```

## What the container does

1. Mounts `foam/VisibleHuman_Head` as `/case`
2. Runs `blockMesh`, `snappyHexMesh` (multi-region watertight solid), optional `topoSet`/`subsetMesh`, `checkMesh`, `simpleFoam`
3. Writes logs and time directories **on your Windows disk** under `foam/VisibleHuman_Head/`

### Import results into the Streamlit viewer

```powershell
py -3.12 scripts\import_openfoam_results.py --case VisibleHuman_Head
py -3.12 -m streamlit run app/viewer.py
```

The viewer (`APP_VERSION` 0.6.x) shows **OpenFOAM simpleFoam** when
`outputs/<case>/<case>_flow_meta.json` has `method: openfoam_simpleFoam`.

## Image used

Default: `opencfd/openfoam-run:2412` (ESI OpenCFD).

Fallback tags tried by the script: `2312`, then Foundation-style community images.

## Disk / time

- Image download: **several GB**, first time only  
- First CFD run: **minutes to hours** depending on mesh settings  

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `docker` not recognized | Start Docker Desktop; new PowerShell |
| Daemon not running | Wait for Docker engine; restart Docker Desktop |
| Pull fails | Check network; try `-Image opencfd/openfoam-run:2312` |
| snappy empty mesh | Check `log.snappyHexMesh`; adjust `locationInMesh` |
| Permission errors on files | Run Docker Desktop as your user; avoid admin-only folders |

## Compose alternative

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
docker compose -f docker/docker-compose.yml run --rm openfoam
```
