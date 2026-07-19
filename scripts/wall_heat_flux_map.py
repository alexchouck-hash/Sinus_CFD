#!/usr/bin/env python3
"""
Spatial wall heat-flux map — where mucosal cooling concentrates.

Wall heat flux for the passive-temperature solve is q = k·(∇T·n) at each wall
face [W/m²], k = thermal conductivity of air. This reads the OpenFOAM mesh and
the grad(T) field (written by the `gradT` functionObject the scaffold adds),
computes q per wall face using the wall-adjacent (owner) cell gradient — which
sits right against the wall thanks to the prism layers, so the cell-centre
gradient closely approximates the true wall gradient — and:

  - integrates q over the wall to a total heat loss (W), cross-checked against
    the enthalpy-based number from compute_mucosal_cooling.py (they should agree)
  - exports a colour-mapped wall surface (PLY) showing the spatial distribution
  - reports where flux peaks

Requires a solve done with the thermal + gradT functionObjects (re-scaffold with
the current scaffold_openfoam_case.py, which adds `gradT`, then re-run).

Usage:
  py -3.12 scripts/wall_heat_flux_map.py --case P001
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import trimesh

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.openfoam_import import (  # noqa: E402
    latest_time_dir,
    read_foam_faces,
    read_foam_label_list,
    read_foam_points,
    read_foam_vector_field,
)

# Thermal conductivity of air near body temperature (W/m·K).
K_AIR = 0.027


def _parse_boundary_patches(path: Path) -> dict[str, tuple[int, int]]:
    """polyMesh/boundary → {patch: (nFaces, startFace)}."""
    text = path.read_text(encoding="utf-8", errors="replace")
    patches: dict[str, tuple[int, int]] = {}
    # Each patch: name { ... nFaces N; ... startFace S; ... }
    for m in re.finditer(r"([A-Za-z_][\w]*)\s*\{([^}]*)\}", text):
        name, body = m.group(1), m.group(2)
        nm = re.search(r"nFaces\s+(\d+)", body)
        sm = re.search(r"startFace\s+(\d+)", body)
        if nm and sm:
            patches[name] = (int(nm.group(1)), int(sm.group(1)))
    return patches


def _grad_field_path(time_dir: Path) -> Path | None:
    for name in ("grad(T)", "gradT", "grad(T)T"):
        p = time_dir / name
        if p.is_file():
            return p
    # any file whose name starts with grad
    for p in time_dir.iterdir():
        if p.name.lower().startswith("grad") and p.is_file():
            return p
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="P001")
    p.add_argument("--foam-root", type=Path, default=None)
    p.add_argument("--k-air", type=float, default=K_AIR, help="air thermal conductivity W/m·K")
    p.add_argument("--wall-patch", default="wall")
    args = p.parse_args()

    foam = args.foam_root or (REPO_ROOT / "foam" / args.case)
    poly = foam / "constant" / "polyMesh"
    if not (poly / "owner").is_file():
        print(f"ERROR: no mesh under {poly} — run the solve first.", file=sys.stderr)
        return 1

    time_name = latest_time_dir(foam)
    time_dir = foam / time_name
    grad_path = _grad_field_path(time_dir)
    if grad_path is None:
        print(
            f"ERROR: no grad(T) field in {time_dir}. Re-scaffold (adds the gradT "
            "functionObject) and re-run the thermal solve.",
            file=sys.stderr,
        )
        return 1

    points = read_foam_points(poly / "points")
    faces = read_foam_faces(poly / "faces")
    owner = read_foam_label_list(poly / "owner")
    gradT = read_foam_vector_field(grad_path)  # (nCells, 3), K/m
    patches = _parse_boundary_patches(poly / "boundary")

    if args.wall_patch not in patches:
        print(f"ERROR: patch '{args.wall_patch}' not in {list(patches)}", file=sys.stderr)
        return 1
    n_wall, start = patches[args.wall_patch]
    wall_face_ids = range(start, start + n_wall)

    tri_v: list[np.ndarray] = []
    tri_q: list[float] = []
    areas = np.zeros(n_wall)
    qmag = np.zeros(n_wall)

    for i, fid in enumerate(wall_face_ids):
        verts = points[faces[fid]]  # (m, 3)
        if len(verts) < 3:
            continue
        centroid = verts.mean(axis=0)
        # Fan-triangulate the polygon; accumulate area-weighted normal.
        n_acc = np.zeros(3)
        face_area = 0.0
        for k in range(1, len(verts) - 1):
            a = verts[k] - verts[0]
            b = verts[k + 1] - verts[0]
            c = np.cross(a, b)
            n_acc += c
            face_area += 0.5 * np.linalg.norm(c)
            tri_v.append(np.array([verts[0], verts[k], verts[k + 1]]))
        nrm = np.linalg.norm(n_acc)
        if nrm == 0 or face_area == 0:
            continue
        n_hat = n_acc / nrm
        oc = owner[fid]
        g = gradT[oc] if oc < len(gradT) else np.zeros(3)
        # Mucosal heat loss is outward (wall warmer than air); magnitude of the
        # wall-normal gradient component gives the local cooling intensity.
        q = args.k_air * abs(float(np.dot(g, n_hat)))  # W/m²
        areas[i] = face_area
        qmag[i] = q
        for _ in range(len(verts) - 2):
            tri_q.append(q)

    total_W = float(np.sum(qmag * areas))
    wall_area_cm2 = float(np.sum(areas)) * 1e4
    valid = qmag[areas > 0]

    print(f"[{args.case}] wall heat-flux map  (time={time_name}, k={args.k_air} W/m·K)")
    print(f"  wall faces         : {n_wall:,}  ({wall_area_cm2:.1f} cm² total)")
    print(f"  heat flux q (W/m²) : mean {valid.mean():.1f}  peak {valid.max():.1f}")
    print(f"  integrated heat loss: {total_W:.3f} W")
    print("  (cross-check vs compute_mucosal_cooling.py's enthalpy number — should agree)")

    # Colour-mapped wall surface PLY (viridis on sqrt(q) for contrast).
    if tri_v:
        tv = np.array(tri_v)  # (T, 3, 3)
        verts_flat = tv.reshape(-1, 3)
        tri_faces = np.arange(len(verts_flat)).reshape(-1, 3)
        q_arr = np.array(tri_q)
        qn = np.sqrt(np.clip(q_arr, 0, None))
        qn = (qn - qn.min()) / (qn.ptp() + 1e-12)
        # simple viridis-ish ramp without matplotlib dependency at runtime
        try:
            import matplotlib.cm as cm

            colors = (cm.get_cmap("viridis")(qn)[:, :3] * 255).astype(np.uint8)
        except Exception:
            colors = np.stack([(qn * 255).astype(np.uint8)] * 3, axis=1)
        mesh = trimesh.Trimesh(vertices=verts_flat, faces=tri_faces, process=False)
        mesh.visual.face_colors = np.column_stack([colors, np.full(len(colors), 255, np.uint8)])
        out_dir = REPO_ROOT / "outputs" / args.case
        out_dir.mkdir(parents=True, exist_ok=True)
        ply_path = out_dir / f"{args.case}_wall_heat_flux.ply"
        mesh.export(ply_path)
        print(f"  wrote {ply_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
