# OpenFOAM and the solid air body

## What you are modeling

You want **airflow from the nostrils, through the nasal passage (and sinuses), to the trachea**.

In CFD that means:

1. A **volume of air** with known shape (the *fluid domain*).
2. A **closed surface** around that volume, split into named parts:
   - **Wall** — mucosa / tissue (air cannot cross; velocity = 0 at the wall).
   - **Open ports** — nostrils (air enters) and trachea (air leaves).

The **mouth is closed** — no open patch for the mouth.

```
   left_nostril  ──┐
                   ├──►  nasal cavity / sinuses  ──►  trachea
   right_nostril ──┘         (inside solid air)
```

---

## What is the “solid surface / body of the airway”?

Imagine pouring plastic into every open air space in the nose, sinuses, and throat, then removing the plastic casting.

- That casting is the **solid air body**.
- Its **outer surface** is what we export as `*_solid_air_body.stl`.
- CFD does **not** treat that STL as “plastic”; it treats the **inside** as air.

| File | Meaning |
|------|---------|
| `*_solid_air_body.stl` | Outer surface of the air region (passage ± connected sinuses) |
| `*_solid_air_body.nrrd` | Same region as a voxel mask |
| `*_patch_wall.stl` | Mucosa wall (no-slip) |
| `*_patch_left_nostril.stl` | Open inlet (left) |
| `*_patch_right_nostril.stl` | Open inlet (right) |
| `*_patch_trachea.stl` | Open outlet |

Together, **wall + open ports** describe the full boundary of the air solid.

---

## What is OpenFOAM?

**OpenFOAM** is free, open-source software for **computational fluid dynamics** (air, water, heat, etc.).

Roughly:

| Piece | Role |
|-------|------|
| **Mesh** | Tiny cells filling the air volume |
| **Boundaries** | Named faces on the mesh (wall, inlets, outlet) |
| **Solver** | Equations for velocity & pressure (e.g. `simpleFoam`) |
| **BCs** | Rules on each patch (flow rate in, pressure out, no-slip wall) |
| **Post** | Contours, streamlines, ΔP, wall shear |

### Why people use it for nasal CFD

- Handles complex 3D anatomy.
- Steady or breathing-cycle (transient) flow.
- Same patch idea we already use: nares in, trachea out, wall no-slip.

### What it is *not*

- Not a medical device by itself.
- Not automatic: you still supply geometry, mesh quality, and physics settings.
- Our current in-app flow is a **fast potential-flow preview**. OpenFOAM is the step up to real viscous CFD.

---

## Minimal OpenFOAM workflow (after our export)

```text
1. Geometry (we export this)
   solid_air_body.stl + patch_*.stl

2. Background box
   blockMesh  →  coarse box around the head airway

3. Snap mesh to anatomy
   snappyHexMesh  →  fills the solid air body with hex cells,
                     names patches from the open-port STLs

4. Initial & boundary fields
   0/U , 0/p
   - left_nostril / right_nostril : flowRateInletVelocity (~9 L/min each)
   - trachea : p = 0 gauge
   - wall : noSlip

5. Solve
   simpleFoam   (steady inspiration)
   or pimpleFoam (time-varying breath)

6. View
   ParaView — velocity, pressure drop, wall shear
```

Typical resting set-point we already use: **~18 L/min** total, split L/R, held for **~1.7 s** (mean inspiratory flow of a quiet breath).

---

## Export + scaffold (this repo)

```powershell
# 1) Passage walls + centerline (if not done)
py -3.12 scripts\analyze_passage.py --case VisibleHuman_Head

# 2) Solid air body + open-port STLs (mm medical coords)
py -3.12 scripts\export_openfoam_geometry.py --case VisibleHuman_Head

# 3) Full OpenFOAM case (STLs scaled to metres + all dictionaries)
py -3.12 scripts\scaffold_openfoam_case.py --case VisibleHuman_Head
```

### Geometry export

```text
outputs/VisibleHuman_Head/openfoam_geometry/
  VisibleHuman_Head_solid_air_body.stl      # air solid (mm)
  VisibleHuman_Head_patch_wall.stl
  VisibleHuman_Head_patch_left_nostril.stl
  VisibleHuman_Head_patch_right_nostril.stl
  VisibleHuman_Head_patch_trachea.stl
  ...
```

### Ready-to-run case

```text
foam/VisibleHuman_Head/
  0/U  0/p                          # BCs (~9 L/min each naris)
  constant/triSurface/*.stl         # scaled to METRES
  system/blockMeshDict
  system/snappyHexMeshDict
  system/controlDict, fvSchemes, fvSolution
  Allrun / Allclean / run_in_wsl.ps1
  README.md
```

### Run in WSL / Linux (OpenFOAM installed)

```powershell
cd C:\Users\houck\Documents\Sinus_CFD\foam\VisibleHuman_Head
.\run_in_wsl.ps1
```

Or inside WSL:

```bash
cd /mnt/c/Users/houck/Documents/Sinus_CFD/foam/VisibleHuman_Head
chmod +x Allrun && ./Allrun
```

---

## How this relates to “airway must go nostril → trachea”

| Requirement | How we enforce it |
|-------------|-------------------|
| Inlets at real nostrils | Port centers + open inlet masks + inlet STLs |
| Path to trachea | Connected lumen + caudal outlet open mask + trachea STL |
| Solid air for CFD | `solid_air_body` = that lumen (+ optional sinuses) |
| Walls vs openings | Separate STL patches for wall / nares / trachea |

If free air is incomplete on a cadaver CT, a thin conduit may fill gaps so the domain stays continuous; that is documented in case notes.

After meshing, use ParaView (`paraFoam`) to confirm velocity streams from nares to trachea.
