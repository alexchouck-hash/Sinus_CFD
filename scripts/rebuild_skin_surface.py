#!/usr/bin/env python3
"""
Rebuild a cleaner outer skin surface mesh from the head/body mask.

Uses filled-body marching cubes + mild Gaussian smooth, then exports STL
with more faces so the Streamlit viewer can show a true surface + wireframe.

Example:
  py -3.12 scripts/rebuild_skin_surface.py --case VisibleHuman_Head
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import morphology
from skimage.filters import gaussian

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sinus_cfd.pipeline import _mask_to_mesh  # noqa: E402


def _decimate(mesh, target_faces: int):
    if mesh is None or len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(target_faces)
    except Exception:
        import trimesh

        idx = np.linspace(0, len(mesh.faces) - 1, target_faces, dtype=int)
        return trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces[idx], process=False
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", default="VisibleHuman_Head")
    ap.add_argument("--target-faces", type=int, default=45000)
    ap.add_argument("--smooth-sigma", type=float, default=0.9)
    args = ap.parse_args()

    case_id = args.case
    out = REPO_ROOT / "outputs" / case_id

    # Prefer head soft-tissue solid; fall back to filled body from tissues
    candidates = [
        out / f"{case_id}_head_mask.nrrd",
        out / f"{case_id}_soft_tissue_mask.nrrd",
        out / f"{case_id}_skin_shell_mask.nrrd",
    ]
    body = None
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    src = None
    for c in candidates:
        if not c.is_file():
            continue
        img = sitk.ReadImage(str(c))
        arr = sitk.GetArrayFromImage(img).astype(bool)
        if not arr.any():
            continue
        body = arr
        spacing = img.GetSpacing()  # x,y,z
        origin = img.GetOrigin()
        src = c.name
        break

    if body is None:
        # Reconstruct body from airway complement is wrong; need flow npz + tissues
        tissues = out / f"{case_id}_tissues.nrrd"
        if tissues.is_file():
            img = sitk.ReadImage(str(tissues))
            t = sitk.GetArrayFromImage(img)
            # labels: exterior=0 air=1 soft=2 cart=3 bone=4 (whole_head notes)
            body = t >= 2
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            src = tissues.name
        else:
            print("No head/body mask found to rebuild skin.")
            return 1

    # Filled body so sinuses don't punch holes in the outer skin shell
    filled = ndi.binary_fill_holes(body)
    filled = morphology.closing(filled, footprint=morphology.ball(1))
    filled = ndi.binary_fill_holes(filled)

    # Keep largest component
    lab, n = ndi.label(filled)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        filled = lab == int(np.argmax(counts))

    vol = filled.astype(np.float32)
    if args.smooth_sigma > 0:
        vol = gaussian(vol, sigma=args.smooth_sigma, preserve_range=True)
    mask = vol >= 0.45
    lab, n = ndi.label(mask)
    if n > 1:
        counts = np.bincount(lab.ravel())
        counts[0] = 0
        mask = lab == int(np.argmax(counts))

    # pipeline expects spacing/origin as xyz
    mesh = _mask_to_mesh(mask, spacing, origin)
    if mesh is None or len(mesh.faces) == 0:
        print("Marching cubes produced empty skin mesh.")
        return 1

    n_raw = len(mesh.faces)
    mesh = _decimate(mesh, args.target_faces)
    # Light Taubin-like smooth if available
    try:
        trimesh_mod = __import__("trimesh")
        if hasattr(mesh, "smoothed"):
            pass
        mesh = mesh.smoothed() if False else mesh
        # filter_laplacian if present
        try:
            trimesh_mod.smoothing.filter_laplacian(mesh, iterations=2)
        except Exception:
            pass
    except Exception:
        pass

    stl_path = out / f"{case_id}_skin.stl"
    mesh.export(str(stl_path))

    # Also store shell mask for QC
    shell = filled & ~morphology.erosion(filled, footprint=morphology.ball(1))
    shell_img = sitk.GetImageFromArray(shell.astype(np.uint8))
    shell_img.SetSpacing(spacing)
    shell_img.SetOrigin(origin)
    sitk.WriteImage(shell_img, str(out / f"{case_id}_skin_shell_mask.nrrd"))

    meta = {
        "case_id": case_id,
        "source_mask": src,
        "raw_faces": int(n_raw),
        "exported_faces": int(len(mesh.faces)),
        "exported_vertices": int(len(mesh.vertices)),
        "watertight": bool(getattr(mesh, "is_watertight", False)),
        "smooth_sigma": args.smooth_sigma,
        "bounds_mm": np.asarray(mesh.bounds).tolist(),
    }
    (out / f"{case_id}_skin_surface_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"OK skin surface {stl_path.name}: faces {n_raw}→{len(mesh.faces)} "
        f"verts={len(mesh.vertices)} src={src}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
