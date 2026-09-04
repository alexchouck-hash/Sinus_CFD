"""A cavity inside the solid air body must survive as an inner wall.

With the sinuses excluded from the voxel solid they are cavities inside it.
Marching cubes gives an outer shell plus one inner shell per cavity, and the
mesher used to keep only the largest component -- so the surface handed to
snappyHexMesh wrapped the sinuses back in. On VH male the outer shell enclosed
74.1 mL against a 51.8 mL voxel solid; the two dropped shells were the antra,
13.6 and 9.5 mL. The mesh-vs-voxel guard caught it. The mesher now keeps every
closed shell above a sliver floor and orients cavity walls away from the
fluid, so the mesh's signed volume is outer minus cavities.
"""

from __future__ import annotations

import numpy as np

from sinus_cfd.openfoam_export import (
    MIN_SHELL_VOLUME_MM3,
    _keep_solid_shells,
    export_openfoam_geometry,
    solid_mask_to_watertight_mesh,
)

ISO = (1.0, 1.0, 1.0)
ORIGIN = (0.0, 0.0, 0.0)


def _block_with_cavity():
    solid = np.zeros((40, 40, 40), dtype=bool)
    solid[8:32, 8:32, 8:32] = True            # 24^3 = 13824 mm^3
    cavity = np.zeros_like(solid)
    cavity[16:24, 16:24, 16:24] = True        # 8^3 = 512 mm^3, fully enclosed
    solid &= ~cavity
    return solid, cavity


def test_mesher_keeps_the_cavity_wall_and_subtracts_it():
    solid, cavity = _block_with_cavity()
    mesh, notes = solid_mask_to_watertight_mesh(solid, ISO, ORIGIN, target_faces=40000)
    parts = mesh.split(only_watertight=False)
    assert len(parts) >= 2, f"cavity wall was dropped: {len(parts)} shell(s); {notes}"
    assert mesh.is_watertight
    expected = float(solid.sum())               # mm^3 at ISO spacing
    assert abs(abs(float(mesh.volume)) - expected) / expected < 0.06, (
        abs(float(mesh.volume)), expected, notes)
    # the inner shell is oriented away from the fluid: its own volume is negative
    inner = min(parts, key=lambda p: abs(float(p.volume)))
    assert float(inner.volume) < 0


def test_slivers_are_still_dropped():
    import trimesh
    big = trimesh.creation.box(extents=(10, 10, 10))
    speck = trimesh.creation.box(extents=(0.5, 0.5, 0.5))   # 0.125 mm^3 < floor
    speck.apply_translation((30, 30, 30))
    kept = _keep_solid_shells(trimesh.util.concatenate([big, speck]))
    assert len(kept.split(only_watertight=False)) == 1
    assert abs(float(kept.volume)) > MIN_SHELL_VOLUME_MM3


def test_export_guard_passes_on_a_solid_with_a_cavity(tmp_path):
    """The mesh-vs-voxel guard compares against the voxel solid with the cavity
    removed; with the inner wall kept and oriented, they agree."""
    solid, cavity = _block_with_cavity()
    inlet = np.zeros_like(solid); inlet[8:32, 8:10, 8:32] = True
    outlet = np.zeros_like(solid); outlet[8:32, 30:32, 8:32] = True
    res = export_openfoam_geometry(
        case_id="cavity", output_dir=tmp_path, lumen=solid, inlet_open=inlet,
        outlet_open=outlet, spacing=ISO, origin=ORIGIN, all_interior_air=None,
        sinus_exclude=cavity)
    assert any("good for snappyHexMesh" in n for n in res.notes), res.notes
    assert any("sinus-free domain preserved through sealing" in n for n in res.notes), res.notes
