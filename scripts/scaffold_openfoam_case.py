#!/usr/bin/env python3
"""
Scaffold a minimal OpenFOAM case from exported solid-air geometry.

- Scales STLs from millimetres (medical coords) to metres (OpenFOAM SI)
- Writes blockMeshDict, snappyHexMeshDict, 0/U, 0/p, controlDict, etc.
- Documents how to run under WSL / native Linux OpenFOAM

Does not require OpenFOAM installed to generate the case.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]


def _scale_stl_mm_to_m(src: Path, dst: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Scale STL mm → m. Preserves multi-region ASCII 'solid name' blocks for OpenFOAM.
    """
    text = src.read_text(encoding="utf-8", errors="replace")
    # Multi-region ASCII: rewrite vertex lines in place
    if "endsolid" in text.lower() and "facet" in text.lower():
        def scale_vertex(m: re.Match) -> str:
            x, y, z = (float(m.group(i)) / 1000.0 for i in (1, 2, 3))
            return f"vertex {x:.8e} {y:.8e} {z:.8e}"

        scaled = re.sub(
            r"vertex\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)",
            scale_vertex,
            text,
            flags=re.I,
        )
        # also scale normals? normals are unitless direction — leave as-is
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(scaled, encoding="utf-8")
        # bounds from vertices
        nums = re.findall(
            r"vertex\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)",
            scaled,
            flags=re.I,
        )
        pts = np.array([[float(a), float(b), float(c)] for a, b, c in nums], dtype=float)
        return pts.min(axis=0), pts.max(axis=0)

    mesh = trimesh.load(src, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    mesh = mesh.copy()
    mesh.vertices = mesh.vertices / 1000.0  # mm → m
    dst.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(dst)
    return mesh.bounds


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.lstrip("\n"), encoding="utf-8")


def scaffold(
    case_id: str = "VisibleHuman_Head",
    outputs_root: Path | None = None,
    foam_root: Path | None = None,
    cells: int = 32,
    wall_layers: int = 5,
) -> Path:
    """
    Write a runnable OpenFOAM case.

    ``wall_layers`` adds that many prism boundary layers on the mucosa wall
    patch (snappyHexMesh addLayers). The roadmap flags a boundary-layer prism
    mesh as mandatory for trustworthy wall shear stress and heat flux — the
    surgically relevant quantities — so this defaults on. Set to 0 to skip
    (faster mesh, but wall-adjacent gradients are then unreliable).
    """
    outputs_root = outputs_root or (REPO_ROOT / "outputs")
    geom_dir = outputs_root / case_id / "openfoam_geometry"
    if not geom_dir.is_dir():
        raise FileNotFoundError(
            f"Missing {geom_dir}\n"
            f"Run: py -3.12 scripts/export_openfoam_geometry.py --case {case_id}"
        )

    foam_root = foam_root or (REPO_ROOT / "foam" / case_id)
    tri = foam_root / "constant" / "triSurface"
    system = foam_root / "system"
    zero = foam_root / "0"
    constant = foam_root / "constant"

    # Flow rates from BC JSON if present
    bc_path = outputs_root / case_id / f"{case_id}_boundary_conditions.json"
    q_left = q_right = 1.5e-4  # m³/s  (9 L/min each)
    q_total = 3.0e-4
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        fa = bc.get("flow_assignment") or {}
        q_total = float(fa.get("total_inflow_m3_s", q_total))
        per = fa.get("per_port") or {}
        q_left = float(per.get("left_nostril", {}).get("flow_m3_s", q_total / 2))
        q_right = float(per.get("right_nostril", {}).get("flow_m3_s", q_total / 2))

    # Scale & copy STLs to metres
    stl_map = {
        "solid_air_body.stl": f"{case_id}_solid_air_body.stl",
        "left_nostril.stl": f"{case_id}_patch_left_nostril.stl",
        "right_nostril.stl": f"{case_id}_patch_right_nostril.stl",
        "trachea.stl": f"{case_id}_patch_trachea.stl",
        "wall.stl": f"{case_id}_patch_wall.stl",
    }
    bounds = None
    for dest_name, src_name in stl_map.items():
        src = geom_dir / src_name
        if not src.is_file():
            print(f"warning: missing {src}")
            continue
        b = _scale_stl_mm_to_m(src, tri / dest_name)
        if dest_name == "solid_air_body.stl":
            bounds = b

    if bounds is None:
        raise FileNotFoundError("solid_air_body.stl missing after export")

    bmin = bounds[0]
    bmax = bounds[1]
    # Background box margin (m)
    margin = 0.020
    xmin, ymin, zmin = (bmin - margin).tolist()
    xmax, ymax, zmax = (bmax + margin).tolist()

    # Prefer sealed-mask centroid for locationInMesh (mm → m)
    loc_json = geom_dir / f"{case_id}_locationInMesh_mm.json"
    if loc_json.is_file():
        loc_mm = json.loads(loc_json.read_text(encoding="utf-8"))["locationInMesh_mm"]
        loc = np.array(loc_mm, dtype=float) / 1000.0
    else:
        loc = 0.5 * (bmin + bmax)

    # ---- system/blockMeshDict ----
    _write(
        system / "blockMeshDict",
        f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

// Background box in metres (anatomy STL also in metres)
convertToMeters 1;

vertices
(
    ({xmin:.6f} {ymin:.6f} {zmin:.6f})
    ({xmax:.6f} {ymin:.6f} {zmin:.6f})
    ({xmax:.6f} {ymax:.6f} {zmin:.6f})
    ({xmin:.6f} {ymax:.6f} {zmin:.6f})
    ({xmin:.6f} {ymin:.6f} {zmax:.6f})
    ({xmax:.6f} {ymin:.6f} {zmax:.6f})
    ({xmax:.6f} {ymax:.6f} {zmax:.6f})
    ({xmin:.6f} {ymax:.6f} {zmax:.6f})
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({cells} {cells} {cells}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    box
    {{
        type patch;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
            (0 1 5 4)
            (2 3 7 6)
            (1 2 6 5)
            (0 4 7 3)
        );
    }}
);

mergePatchPairs
(
);
""",
    )

    # ---- system/snappyHexMeshDict ----
    # Single multi-region closed solid; locationInMesh must be inside fluid
    _write(
        system / "snappyHexMeshDict",
        f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}

// Castellation + snap on ONE watertight multi-region solid (ports as regions)
castellatedMesh true;
snap            true;
addLayers       {"true" if wall_layers > 0 else "false"};

geometry
{{
    solid_air_body.stl
    {{
        type triSurfaceMesh;
        name solid_air_body;
        regions
        {{
            left_nostril  {{ name left_nostril; }}
            right_nostril {{ name right_nostril; }}
            trachea       {{ name trachea; }}
            wall          {{ name wall; }}
        }}
    }}
}}

castellatedMeshControls
{{
    // Docker ~8 GB: moderate snap; true fluid keep uses topoSet surfaceToCell
    maxLocalCells 600000;
    maxGlobalCells 900000;
    minRefinementCells 0;
    maxLoadUnbalance 0.10;
    nCellsBetweenLevels 2;

    features
    (
    );

    refinementSurfaces
    {{
        solid_air_body
        {{
            level (1 2);
            regions
            {{
                left_nostril  {{ level (2 2); patchInfo {{ type patch; }} }}
                right_nostril {{ level (2 2); patchInfo {{ type patch; }} }}
                trachea       {{ level (2 2); patchInfo {{ type patch; }} }}
                wall          {{ level (1 2); patchInfo {{ type wall; }} }}
            }}
        }}
    }}

    resolveFeatureAngle 30;

    refinementRegions
    {{
    }}

    // Seed for snap (fluid keep is enforced later by surfaceToCell + subsetMesh)
    locationInMesh ({loc[0]:.6f} {loc[1]:.6f} {loc[2]:.6f});

    allowFreeStandingZoneFaces false;
}}

snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
    nFeatureSnapIter 5;
    implicitFeatureSnap true;
    explicitFeatureSnap false;
    multiRegionFeatureSnap true;
}}

addLayersControls
{{
    // relativeSizes: layer thicknesses are fractions of the local cell size.
    // ~5 layers on the mucosa wall resolve the near-wall velocity/thermal
    // gradient so wall shear stress and heat flux are meaningful.
    relativeSizes true;
    layers
    {{
        wall
        {{
            nSurfaceLayers {wall_layers};
        }}
    }}
    expansionRatio 1.2;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedialAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}

meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}

debug 0;
mergeTolerance 1e-6;
""",
    )

    # ---- system/controlDict ----
    # Resistance functionObjects: area-averaged kinematic pressure p and
    # volumetric flow (sum of phi) on each open patch, logged every write.
    # scripts/compute_nasal_resistance.py turns these into nasal resistance
    # R = ρ·ΔP / Q and checks it against published ranges.
    _patch_fos = "\n".join(
        f"""    p_{patch}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        writeFields     false;
        log             true;
        regionType      patch;
        name            {patch};
        operation       areaAverage;
        fields          (p);
    }}
    Q_{patch}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        writeControl    writeTime;
        writeFields     false;
        log             true;
        regionType      patch;
        name            {patch};
        operation       sum;
        fields          (phi);
    }}"""
        for patch in ("left_nostril", "right_nostril", "trachea")
    )
    _write(
        system / "controlDict",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}

application     simpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         500;
deltaT          1;
writeControl    timeStep;
writeInterval   50;
purgeWrite      6;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{
__PATCH_FOS__
}
""".replace("__PATCH_FOS__", _patch_fos),
    )

    # ---- system/fvSchemes ----
    _write(
        system / "fvSchemes",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}

ddtSchemes
{
    default         steadyState;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}
""",
    )

    # ---- system/fvSolution ----
    _write(
        system / "fvSolution",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}

solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-6;
        relTol          0.1;
        smoother        GaussSeidel;
    }

    "(U|k|epsilon|omega|f|v2)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
}

SIMPLE
{
    nNonOrthogonalCorrectors 2;
    consistent      no;
    pRefCell        0;
    pRefValue       0;
    residualControl
    {
        p               1e-3;
        U               1e-3;
        "(k|epsilon|omega|f|v2)" 1e-3;
    }
}

relaxationFactors
{
    fields
    {
        p               0.3;
    }
    equations
    {
        U               0.5;
        ".*"            0.5;
    }
}
""",
    )

    # ---- system/decomposeParDict (optional multi-core) ----
    _write(
        system / "decomposeParDict",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}

numberOfSubdomains 4;
method          scotch;
""",
    )

    # ---- system/surfaceFeatureExtractDict ----
    _write(
        system / "surfaceFeatureExtractDict",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeatureExtractDict;
}

solid_air_body.stl
{
    extractionMethod    extractFromSurface;
    extractFromSurfaceCoeffs
    {
        includedAngle   150;
    }
    writeObj            yes;
}
""",
    )

    # ---- system/topoSetDict: keep cells geometrically inside watertight solid ----
    # outsidePoints = background box corner (outside airway)
    _write(
        system / "topoSetDict",
        f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}}

actions
(
    {{
        name    fluidCells;
        type    cellSet;
        action  new;
        source  surfaceToCell;
        sourceInfo
        {{
            file            "constant/triSurface/solid_air_body.stl";
            outsidePoints   (({xmin:.6f} {ymin:.6f} {zmin:.6f}));
            includeCut      false;
            includeInside   true;
            includeOutside  false;
            nearDistance    -1;
            curvature       -100;
        }}
    }}
);
""",
    )

    # ---- constant/transportProperties (air ~37 C) ----
    _write(
        constant / "transportProperties",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}

transportModel  Newtonian;
// kinematic viscosity of air ~ 1.5e-5 m^2/s
nu              [0 2 -1 0 0 0 0] 1.5e-05;
""",
    )

    # ---- constant/turbulenceProperties (laminar first) ----
    _write(
        constant / "turbulenceProperties",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}

simulationType  laminar;
""",
    )

    # ---- 0/U ----
    # After snappy, patches should be named; until mesh exists we list expected names
    # including defaultFaces for residual box faces that may remain
    _write(
        zero / "U",
        f"""
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    left_nostril
    {{
        type            flowRateInletVelocity;
        volumetricFlowRate constant {q_left:.8e};
        value           uniform (0 0 0);
    }}
    right_nostril
    {{
        type            flowRateInletVelocity;
        volumetricFlowRate constant {q_right:.8e};
        value           uniform (0 0 0);
    }}
    trachea
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
    wall
    {{
        type            noSlip;
    }}
    // residual background-box faces after snap (if any)
    box
    {{
        type            slip;
    }}
    defaultFaces
    {{
        type            slip;
    }}
    solid_air_body
    {{
        // if surface kept as single patch before region split
        type            noSlip;
    }}
}}
""",
    )

    # ---- 0/p ----
    _write(
        zero / "p",
        """
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{
    left_nostril
    {
        type            zeroGradient;
    }
    right_nostril
    {
        type            zeroGradient;
    }
    trachea
    {
        type            fixedValue;
        value           uniform 0;
    }
    wall
    {
        type            zeroGradient;
    }
    box
    {
        type            zeroGradient;
    }
    defaultFaces
    {
        type            zeroGradient;
    }
    solid_air_body
    {
        type            zeroGradient;
    }
}
""",
    )

    # ---- Allrun / Allclean (bash for WSL/Linux) ----
    _write(
        foam_root / "Allrun",
        """#!/bin/bash
set -e
cd "${0%/*}" || exit 1

# Source OpenFOAM (adjust to your install)
if [ -f /usr/lib/openfoam/openfoam2312/etc/bashrc ]; then
  # shellcheck disable=SC1091
  source /usr/lib/openfoam/openfoam2312/etc/bashrc
elif [ -f "$HOME/OpenFOAM/OpenFOAM-11/etc/bashrc" ]; then
  # shellcheck disable=SC1091
  source "$HOME/OpenFOAM/OpenFOAM-11/etc/bashrc"
elif [ -n "$WM_PROJECT_DIR" ]; then
  echo "Using existing OpenFOAM env: $WM_PROJECT_DIR"
else
  echo "ERROR: OpenFOAM bashrc not found. Install OpenFOAM and source it first."
  exit 1
fi

echo "== blockMesh =="
blockMesh | tee log.blockMesh

echo "== surfaceFeatureExtract =="
surfaceFeatureExtract 2>/dev/null | tee log.surfaceFeatureExtract || true

echo "== snappyHexMesh =="
snappyHexMesh -overwrite | tee log.snappyHexMesh

echo "== checkMesh =="
checkMesh | tee log.checkMesh

echo "== simpleFoam =="
simpleFoam | tee log.simpleFoam

echo "Done. View with: paraFoam  or  paraFoam -builtin"
""",
    )

    _write(
        foam_root / "Allclean",
        """#!/bin/bash
cd "${0%/*}" || exit 1
rm -rf 0.* [1-9]* processor* constant/polyMesh postProcessing
rm -f log.*
# keep 0/ templates
echo "Cleaned mesh and time directories (kept 0/ field templates)."
""",
    )

    # ---- README ----
    _write(
        foam_root / "README.md",
        f"""# OpenFOAM case: {case_id}

Minimal **steady inspiratory** nasal CFD case generated by Sinus_CFD.

## Physics intent

| Boundary | Type | Value |
|----------|------|-------|
| left_nostril | flow rate inlet | {q_left:.4e} m³/s (~{q_left*60000:.1f} L/min) |
| right_nostril | flow rate inlet | {q_right:.4e} m³/s (~{q_right*60000:.1f} L/min) |
| trachea | pressure outlet | p = 0 (gauge) |
| wall | no-slip | mucosa |

Total ~ **{q_total*60000:.1f} L/min** (mean quiet inspiration). Mouth closed.

Geometry STLs are in **metres** (scaled from medical mm).

## Background box (m)

```
x [{xmin:.4f}, {xmax:.4f}]
y [{ymin:.4f}, {ymax:.4f}]
z [{zmin:.4f}, {zmax:.4f}]
locationInMesh ({loc[0]:.4f} {loc[1]:.4f} {loc[2]:.4f})
```

## Run (Linux / WSL)

```bash
# Install OpenFOAM once, e.g. (Ubuntu WSL):
#   sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
#   sudo add-apt-repository http://dl.openfoam.org/ubuntu
#   sudo apt-get update && sudo apt-get install openfoam11
#
# Or ESI: openfoam.com packages

cd foam/{case_id}
chmod +x Allrun Allclean
./Allrun
```

From Windows PowerShell with WSL:

```powershell
wsl -e bash -lc "cd /mnt/c/Users/houck/Documents/Sinus_CFD/foam/{case_id} && ./Allrun"
```

(Adjust the path to your username/drive.)

## View results

```bash
paraFoam
# or
paraFoam -builtin
```

## Notes / pitfalls

1. **First mesh may need tuning** — if `locationInMesh` is outside the air solid,
   snappy keeps the wrong region. Re-export geometry or nudge the seed point.
2. **Patch names** after snappy must match `0/U` and `0/p`. Run `checkMesh` and
   inspect boundary names; rename dict entries if needed.
3. **Laminar simpleFoam** is a starting point; nasal Re can warrant k-ω SST later.
4. This case is **research scaffolding**, not a clinical device.

## Regenerate this case

```powershell
py -3.12 scripts\\export_openfoam_geometry.py --case {case_id}
py -3.12 scripts\\scaffold_openfoam_case.py --case {case_id}
```
""",
    )

    # Windows helper
    _write(
        foam_root / "run_in_wsl.ps1",
        f"""# Run Allrun inside WSL (OpenFOAM must be installed in the distro)
$caseUnix = (wsl wslpath -a "{foam_root.resolve()}").Trim()
Write-Host "Case path in WSL: $caseUnix"
wsl -e bash -lc "cd '$caseUnix' && chmod +x Allrun Allclean && ./Allrun"
""",
    )

    man = {
        "case_id": case_id,
        "foam_dir": str(foam_root),
        "units": "SI_metres",
        "q_left_m3_s": q_left,
        "q_right_m3_s": q_right,
        "q_total_m3_s": q_total,
        "bounds_m": {"min": bmin.tolist(), "max": bmax.tolist()},
        "locationInMesh_m": loc.tolist(),
        "stl_scaled_from_mm": True,
    }
    (foam_root / "case_manifest.json").write_text(
        json.dumps(man, indent=2), encoding="utf-8"
    )

    print(f"[{case_id}] OpenFOAM case scaffold → {foam_root}")
    print(f"  box (m): {xmin:.4f}..{xmax:.4f}, {ymin:.4f}..{ymax:.4f}, {zmin:.4f}..{zmax:.4f}")
    print(f"  Q left/right: {q_left:.4e} / {q_right:.4e} m3/s")
    print(f"  STLs (metres) in {tri}")
    return foam_root


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="VisibleHuman_Head")
    p.add_argument("--outputs-root", type=Path, default=REPO_ROOT / "outputs")
    p.add_argument("--foam-root", type=Path, default=None)
    p.add_argument(
        "--cells",
        type=int,
        default=32,
        help="blockMesh cells per edge (32 helps seal watertight cut in Docker ~8 GB)",
    )
    p.add_argument(
        "--wall-layers",
        type=int,
        default=5,
        help="prism boundary layers on the mucosa wall (0 to disable; needed for wall shear/heat flux)",
    )
    args = p.parse_args()
    try:
        scaffold(
            case_id=args.case,
            outputs_root=args.outputs_root,
            foam_root=args.foam_root,
            cells=args.cells,
            wall_layers=args.wall_layers,
        )
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
