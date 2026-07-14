# Full OpenFOAM install (Windows + WSL)

## Status after automated setup

1. **WSL + Ubuntu** package install was started on this machine.
2. Windows reported: **reboot required** before Ubuntu becomes available.
3. OpenFOAM install and `simpleFoam` must run **after reboot**.

## After you reboot

### Step 1 — Finish Ubuntu first launch

1. Open **Ubuntu** from the Start menu.
2. Create a UNIX username and password when prompted.
3. Leave that window open or note that WSL is ready.

### Step 2 — Refresh the CFD case (Windows PowerShell)

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
py -3.12 scripts\export_openfoam_geometry.py --case VisibleHuman_Head
py -3.12 scripts\scaffold_openfoam_case.py --case VisibleHuman_Head
```

### Step 3 — Install OpenFOAM + run the case

**Option A — one PowerShell script**

```powershell
cd C:\Users\houck\Documents\Sinus_CFD
powershell -ExecutionPolicy Bypass -File scripts\install_openfoam_wsl.ps1
```

**Option B — manual in Ubuntu terminal**

```bash
bash /mnt/c/Users/houck/Documents/Sinus_CFD/scripts/install_openfoam_wsl.sh
```

This will:

- `apt-get install openfoam11` (or ESI package fallback)
- Run `blockMesh` → `snappyHexMesh` → `checkMesh` → `simpleFoam`
- Write logs under `foam/VisibleHuman_Head/log.*`

### Step 4 — View results

In Ubuntu:

```bash
cd /mnt/c/Users/houck/Documents/Sinus_CFD/foam/VisibleHuman_Head
source /opt/openfoam11/etc/bashrc   # path may vary
paraFoam -builtin
```

Or copy `VTK`/`foam` results into ParaView on Windows.

## What the case does

| Item | Value |
|------|--------|
| Domain | Solid air body (nares → nasal passage ± sinuses → trachea) |
| Inlets | left/right nostril ~9 L/min each |
| Outlet | trachea, p = 0 |
| Wall | no-slip mucosa |
| Solver | laminar `simpleFoam` (steady inspiration) |

## If something fails

| Symptom | Action |
|---------|--------|
| `WSL_E_DISTRO_NOT_FOUND` | Reboot; open Ubuntu once |
| `locationInMesh` outside geometry | Edit `system/snappyHexMeshDict` seed; re-run scaffold |
| Patch names mismatch | `checkMesh` → rename entries in `0/U` and `0/p` |
| Mesh too coarse | Increase refinement levels in `snappyHexMeshDict` |

## Disk / time

- Ubuntu + OpenFOAM: often **5–15 GB**
- First `apt install`: **10–40 minutes**
- First `snappyHexMesh` + `simpleFoam`: **minutes to hours** depending on refinement
