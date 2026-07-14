"""
Import OpenFOAM simpleFoam results onto the CT/airway voxel grid for visualization.

Reads:
  foam/<case>/constant/polyMesh/{points,faces,owner,neighbour,boundary}
  foam/<case>/<time>/U  (and optional p)

Maps cell-centred velocity (SI m/s) to the CT grid (mm) via nearest cell centre,
restricted to the solid-air / airway mask.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from scipy.spatial import cKDTree

from .flow_field import (
    compute_inhale_streamlines,
    compute_streamlines,
    extend_paths_to_outlet_via_centerline,
)


@dataclass
class OpenFoamImportResult:
    case_id: str
    time_name: str
    n_cells: int
    n_mapped_voxels: int
    max_speed_m_s: float
    mean_speed_m_s: float
    mesh_volume_m3: float
    method: str = "openfoam_simpleFoam"
    notes: list[str] = field(default_factory=list)
    out_npz: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NUM_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def _strip_foam_comments(text: str) -> str:
    # remove // line comments and /* */ blocks
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*?$", " ", text, flags=re.M)
    return text


def _find_list_body(text: str) -> str:
    """Return content inside the first top-level ( ... ); after FoamFile."""
    # Drop FoamFile { ... } block if present
    text = re.sub(r"FoamFile\s*\{.*?\}", " ", text, count=1, flags=re.S)
    text = _strip_foam_comments(text)
    start = text.find("(")
    if start < 0:
        raise ValueError("No list '(' found in OpenFOAM file")
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    raise ValueError("Unbalanced parentheses in OpenFOAM list")


def read_foam_points(path: Path) -> np.ndarray:
    """Read constant/polyMesh/points → (N, 3) float64 in metres."""
    text = path.read_text(encoding="utf-8", errors="replace")
    body = _find_list_body(text)
    # points are (x y z)
    nums = _NUM_RE.findall(body)
    arr = np.array([float(x) for x in nums], dtype=np.float64)
    if arr.size % 3 != 0:
        raise ValueError(f"points: expected multiple of 3 values, got {arr.size}")
    return arr.reshape(-1, 3)


def read_foam_faces(path: Path) -> list[np.ndarray]:
    """Read faces as list of point-index arrays."""
    text = path.read_text(encoding="utf-8", errors="replace")
    body = _find_list_body(text)
    faces: list[np.ndarray] = []
    # format: n(i j k ...) or just (i j k) in some dumps; OpenFOAM uses n(...)
    for m in re.finditer(r"(\d+)\s*\(([^)]*)\)", body):
        n = int(m.group(1))
        ids = [int(x) for x in m.group(2).split()]
        if len(ids) != n:
            # tolerate mismatch by using parsed ids
            pass
        faces.append(np.asarray(ids, dtype=np.int64))
    if not faces:
        raise ValueError(f"No faces parsed from {path}")
    return faces


def read_foam_label_list(path: Path) -> np.ndarray:
    """owner / neighbour label lists."""
    text = path.read_text(encoding="utf-8", errors="replace")
    body = _find_list_body(text)
    nums = re.findall(r"-?\d+", body)
    return np.asarray([int(x) for x in nums], dtype=np.int64)


def read_foam_vector_field(path: Path) -> np.ndarray:
    """
    Read volVectorField internalField → (nCells, 3).
    Supports uniform and nonuniform List<vector>.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    text_nc = _strip_foam_comments(text)

    # uniform (ux uy uz)
    m_uni = re.search(
        r"internalField\s+uniform\s*\(\s*"
        r"([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*\)",
        text_nc,
    )
    if m_uni:
        # Need cell count from mesh — caller may expand; return single vector
        v = np.array(
            [float(m_uni.group(1)), float(m_uni.group(2)), float(m_uni.group(3))],
            dtype=np.float64,
        )
        return v.reshape(1, 3)

    m = re.search(r"internalField\s+nonuniform\s+List<vector>", text_nc)
    if not m:
        # try without List<vector>
        m = re.search(r"internalField\s+nonuniform", text_nc)
    if not m:
        raise ValueError(f"Could not find internalField in {path}")

    # Count then list
    rest = text_nc[m.end() :]
    m_count = re.search(r"(\d+)\s*\(", rest)
    if not m_count:
        raise ValueError(f"Could not parse nonuniform count in {path}")
    n = int(m_count.group(1))
    start = m_count.end() - 1  # at '('
    depth = 0
    body_start = None
    for i, ch in enumerate(rest[start:], start=start):
        if ch == "(":
            if depth == 0:
                body_start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                body = rest[body_start:i]
                break
    else:
        raise ValueError(f"Unbalanced vector list in {path}")

    nums = _NUM_RE.findall(body)
    arr = np.array([float(x) for x in nums], dtype=np.float64)
    if arr.size != n * 3:
        # truncate/pad defensively
        if arr.size < n * 3:
            raise ValueError(f"U field: expected {n*3} values, got {arr.size}")
        arr = arr[: n * 3]
    return arr.reshape(n, 3)


def read_foam_scalar_field(path: Path) -> np.ndarray:
    """Read volScalarField internalField → (nCells,)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    text_nc = _strip_foam_comments(text)

    m_uni = re.search(
        r"internalField\s+uniform\s+([-+eE0-9.]+)",
        text_nc,
    )
    if m_uni:
        return np.array([float(m_uni.group(1))], dtype=np.float64)

    m = re.search(r"internalField\s+nonuniform\s+List<scalar>", text_nc)
    if not m:
        m = re.search(r"internalField\s+nonuniform", text_nc)
    if not m:
        raise ValueError(f"Could not find scalar internalField in {path}")
    rest = text_nc[m.end() :]
    m_count = re.search(r"(\d+)\s*\(", rest)
    if not m_count:
        raise ValueError(f"Could not parse scalar count in {path}")
    n = int(m_count.group(1))
    start = m_count.end() - 1
    depth = 0
    body_start = None
    for i, ch in enumerate(rest[start:], start=start):
        if ch == "(":
            if depth == 0:
                body_start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                body = rest[body_start:i]
                break
    else:
        raise ValueError(f"Unbalanced scalar list in {path}")
    nums = _NUM_RE.findall(body)
    arr = np.array([float(x) for x in nums], dtype=np.float64)
    if arr.size < n:
        raise ValueError(f"p field: expected {n} values, got {arr.size}")
    return arr[:n]


def cell_centres_from_poly_mesh(
    points: np.ndarray,
    faces: list[np.ndarray],
    owner: np.ndarray,
    neighbour: np.ndarray | None = None,
) -> np.ndarray:
    """
    Approximate cell centres as average of unique face-vertex points per cell.
    Good enough for nearest-neighbour mapping to CT voxels.
    """
    n_cells = int(owner.max()) + 1
    if neighbour is not None and neighbour.size:
        n_cells = max(n_cells, int(neighbour.max()) + 1)

    # Accumulate vertex index sums per cell (via face vertices)
    sum_xyz = np.zeros((n_cells, 3), dtype=np.float64)
    count = np.zeros(n_cells, dtype=np.int64)

    n_internal = 0 if neighbour is None else len(neighbour)
    for fi, face in enumerate(faces):
        if face.size == 0:
            continue
        c = points[face].mean(axis=0)
        oc = int(owner[fi])
        sum_xyz[oc] += c
        count[oc] += 1
        if fi < n_internal:
            nc = int(neighbour[fi])
            sum_xyz[nc] += c
            count[nc] += 1

    count = np.maximum(count, 1)
    return sum_xyz / count[:, None]


def latest_time_dir(foam_case: Path) -> str:
    times: list[tuple[float, str]] = []
    for p in foam_case.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if name in ("0", "constant", "system"):
            continue
        if (p / "U").is_file():
            try:
                times.append((float(name), name))
            except ValueError:
                continue
    if not times:
        # fall back to 0 if only initial
        if (foam_case / "0" / "U").is_file():
            return "0"
        raise FileNotFoundError(f"No time directories with U under {foam_case}")
    times.sort()
    return times[-1][1]


def import_openfoam_to_grid(
    case_id: str = "VisibleHuman_Head",
    foam_root: Path | str | None = None,
    outputs_root: Path | str | None = None,
    time_name: str | None = None,
    n_streamline_seeds: int = 48,
) -> OpenFoamImportResult:
    """
    Sample OpenFOAM U onto the solid-air / airway mask grid and write NPZ + streamlines.
    """
    repo = Path(__file__).resolve().parents[2]
    foam_root = Path(foam_root or (repo / "foam" / case_id))
    outputs_root = Path(outputs_root or (repo / "outputs" / case_id))
    notes: list[str] = []

    mesh_dir = foam_root / "constant" / "polyMesh"
    for req in ("points", "faces", "owner"):
        if not (mesh_dir / req).is_file():
            raise FileNotFoundError(f"Missing {mesh_dir / req}")

    time_name = time_name or latest_time_dir(foam_root)
    u_path = foam_root / time_name / "U"
    if not u_path.is_file():
        raise FileNotFoundError(f"Missing {u_path}")

    # Prefer passage lumen (nares→trachea, no maxillary detours) for viz domain
    passage_nrrd = outputs_root / f"{case_id}_passage_lumen.nrrd"
    solid_nrrd = outputs_root / "openfoam_geometry" / f"{case_id}_solid_air_body.nrrd"
    airway_nrrd = outputs_root / f"{case_id}_airway_mask.nrrd"
    mask_path = (
        passage_nrrd
        if passage_nrrd.is_file()
        else (solid_nrrd if solid_nrrd.is_file() else airway_nrrd)
    )
    if not mask_path.is_file():
        raise FileNotFoundError(f"No solid/airway mask for {case_id}")

    img = sitk.ReadImage(str(mask_path))
    airway = sitk.GetArrayFromImage(img).astype(bool)
    spacing = tuple(float(v) for v in img.GetSpacing())  # mm
    origin = tuple(float(v) for v in img.GetOrigin())  # mm
    notes.append(f"Mapped onto mask: {mask_path.name} (passage preferred over sinuses)")

    print(f"[{case_id}] Reading polyMesh…")
    points = read_foam_points(mesh_dir / "points")
    faces = read_foam_faces(mesh_dir / "faces")
    owner = read_foam_label_list(mesh_dir / "owner")
    neighbour = None
    if (mesh_dir / "neighbour").is_file():
        neighbour = read_foam_label_list(mesh_dir / "neighbour")

    centres = cell_centres_from_poly_mesh(points, faces, owner, neighbour)
    n_cells = centres.shape[0]
    print(f"[{case_id}] cells={n_cells}  faces={len(faces)}  points={len(points)}")

    U = read_foam_vector_field(u_path)
    if U.shape[0] == 1 and n_cells > 1:
        U = np.repeat(U, n_cells, axis=0)
    if U.shape[0] != n_cells:
        notes.append(
            f"WARNING: U has {U.shape[0]} entries, mesh has {n_cells} cells — truncating/padding."
        )
        if U.shape[0] > n_cells:
            U = U[:n_cells]
        else:
            pad = np.zeros((n_cells - U.shape[0], 3), dtype=np.float64)
            U = np.vstack([U, pad])

    p_path = foam_root / time_name / "p"
    p_cells = None
    if p_path.is_file():
        try:
            p_cells = read_foam_scalar_field(p_path)
            if p_cells.shape[0] == 1 and n_cells > 1:
                p_cells = np.repeat(p_cells, n_cells)
            if p_cells.shape[0] != n_cells:
                p_cells = None
        except Exception as exc:
            notes.append(f"Could not read p: {exc}")

    # CT airway voxel centres in metres (OpenFOAM SI)
    zz, yy, xx = np.where(airway)
    sx, sy, sz = spacing
    ox, oy, oz = origin
    # physical mm then → m
    vx_mm = ox + xx * sx
    vy_mm = oy + yy * sy
    vz_mm = oz + zz * sz
    voxel_m = np.column_stack([vx_mm, vy_mm, vz_mm]) / 1000.0

    tree = cKDTree(centres)
    dist, idx = tree.query(voxel_m, k=1, workers=-1)
    # discard voxels far from any cell (outside foam domain)
    # cell size ~ few mm; threshold 8 mm = 0.008 m
    max_dist = 0.008
    ok = dist <= max_dist
    notes.append(
        f"Nearest-cell map: {int(ok.sum())}/{len(ok)} voxels within {max_dist*1000:.1f} mm."
    )

    ux = np.zeros(airway.shape, dtype=np.float32)
    uy = np.zeros(airway.shape, dtype=np.float32)
    uz = np.zeros(airway.shape, dtype=np.float32)
    pressure = np.zeros(airway.shape, dtype=np.float32)

    if ok.any():
        ii = idx[ok]
        ux[zz[ok], yy[ok], xx[ok]] = U[ii, 0]
        uy[zz[ok], yy[ok], xx[ok]] = U[ii, 1]
        uz[zz[ok], yy[ok], xx[ok]] = U[ii, 2]
        if p_cells is not None:
            pressure[zz[ok], yy[ok], xx[ok]] = p_cells[ii]

    speed = np.sqrt(ux * ux + uy * uy + uz * uz)
    mapped = airway & (speed > 0)
    # also mark mapped-by-distance even if U~0
    if ok.any():
        mapped = np.zeros(airway.shape, dtype=bool)
        mapped[zz[ok], yy[ok], xx[ok]] = True

    max_speed = float(speed[mapped].max()) if mapped.any() else 0.0
    mean_speed = float(speed[mapped].mean()) if mapped.any() else 0.0
    # approximate mesh volume from cell bounding boxes is hard; use cell count * mean volume
    # from point span / n_cells rough
    bbox = points.max(axis=0) - points.min(axis=0)
    mesh_vol = float(np.prod(bbox))  # upper bound box volume
    notes.append(
        f"OpenFOAM time={time_name}; max|U|={max_speed:.4f} m/s mean|U|={mean_speed:.4f} m/s"
    )
    notes.append("Velocity from simpleFoam (incompressible), SI m/s.")

    # Inlet/outlet masks from BC if available
    inlet_mask = np.zeros(airway.shape, dtype=bool)
    outlet_mask = np.zeros(airway.shape, dtype=bool)
    bc_path = outputs_root / f"{case_id}_boundary_conditions.json"
    seed_pts: list[np.ndarray] = []
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports", []):
            c = np.array(port["center_mm"], dtype=float)
            seed_pts.append(c)
            # small sphere on mask
            d2 = (vx_mm - c[0]) ** 2 + (vy_mm - c[1]) ** 2 + (vz_mm - c[2]) ** 2
            near = d2 <= 8.0**2
            if port.get("role") == "inlet":
                inlet_mask[zz[near], yy[near], xx[near]] = True
            elif port.get("role") == "outlet":
                outlet_mask[zz[near], yy[near], xx[near]] = True

    # Port geometry: skin nares (external) + lumen inlet_open + caudal outlet_open
    inlet_centers: list[list[float]] = []
    skin_naris_centers: list[list[float]] = []
    outlet_center: list[float] | None = None

    # Caudal trachea = centroid of passage outlet_open (most reliable)
    outlet_open_p = outputs_root / f"{case_id}_passage_outlet_open.nrrd"
    if outlet_open_p.is_file():
        oimg = sitk.ReadImage(str(outlet_open_p))
        om = sitk.GetArrayFromImage(oimg).astype(bool)
        oz_, oy_, ox_ = np.where(om)
        if len(oz_):
            sp_o = oimg.GetSpacing()
            org_o = oimg.GetOrigin()
            outlet_center = [
                float(org_o[0] + ox_.mean() * sp_o[0]),
                float(org_o[1] + oy_.mean() * sp_o[1]),
                float(org_o[2] + oz_.mean() * sp_o[2]),
            ]
            notes.append(f"Trachea marker from outlet_open centroid: {outlet_center}")

    # Prefer centerline end if closer caudal / more posterior
    centerline_mm: list | None = None
    passage_json = outputs_root / f"{case_id}_passage.json"
    if passage_json.is_file():
        pj = json.loads(passage_json.read_text(encoding="utf-8"))
        cl = pj.get("centerline_mm") or []
        if len(cl) >= 2:
            centerline_mm = cl
            cl_end = [float(v) for v in cl[-1]]
            if outlet_center is None:
                outlet_center = cl_end
            notes.append(f"Passage centerline end (nares→trachea path): {cl_end}")

    # Lumen-side inlet openings (where streamlines integrate)
    inlet_open_p = outputs_root / f"{case_id}_passage_inlet_open.nrrd"
    if inlet_open_p.is_file():
        iimg = sitk.ReadImage(str(inlet_open_p))
        im = sitk.GetArrayFromImage(iimg).astype(bool)
        # split L/R by x median
        iz_, iy_, ix_ = np.where(im)
        if len(ix_):
            sp_i = iimg.GetSpacing()
            org_i = iimg.GetOrigin()
            xmed = float(np.median(ix_))
            for side, mask_x in (("left", ix_ >= xmed), ("right", ix_ < xmed)):
                if not mask_x.any():
                    continue
                inlet_centers.append(
                    [
                        float(org_i[0] + ix_[mask_x].mean() * sp_i[0]),
                        float(org_i[1] + iy_[mask_x].mean() * sp_i[1]),
                        float(org_i[2] + iz_[mask_x].mean() * sp_i[2]),
                    ]
                )
            notes.append(f"Lumen inlet_open centers (L/R): {inlet_centers}")

    # External skin nares (face surface)
    nares_json = outputs_root / f"{case_id}_nares.json"
    if nares_json.is_file():
        nj = json.loads(nares_json.read_text(encoding="utf-8"))
        for npnt in nj.get("naris_points") or []:
            if npnt.get("center_mm"):
                skin_naris_centers.append([float(v) for v in npnt["center_mm"]])
    if not skin_naris_centers and bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        for port in bc.get("ports", []):
            if port.get("role") == "inlet" and port.get("center_mm"):
                skin_naris_centers.append([float(v) for v in port["center_mm"]])

    if not inlet_centers and seed_pts:
        inlet_centers = [list(s) for s in seed_pts[:2]]
    if not inlet_centers and skin_naris_centers:
        inlet_centers = list(skin_naris_centers)

    # Persist corrected trachea location into BC JSON for the viewer
    if bc_path.is_file() and outlet_center is not None:
        try:
            bc_fix = json.loads(bc_path.read_text(encoding="utf-8"))
            for port in bc_fix.get("ports", []):
                if port.get("role") == "outlet" or port.get("name") == "trachea":
                    port["center_mm"] = outlet_center
                    port["method"] = "passage_outlet_open_centroid"
                    port["notes"] = (
                        "Caudal airway outlet (trachea direction) from passage "
                        "outlet_open mask centroid — not maxillary sinus."
                    )
            bc_path.write_text(json.dumps(bc_fix, indent=2), encoding="utf-8")
            notes.append("Updated BC trachea center_mm for viewer labels.")
        except Exception as exc:
            notes.append(f"Could not update BC trachea center: {exc}")

    # Streamlines restricted to passage lumen (no sinus chambers)
    domain = airway & mapped
    if inlet_centers:
        lines = compute_inhale_streamlines(
            ux.astype(float),
            uy.astype(float),
            uz.astype(float),
            domain,
            spacing,
            origin,
            inlet_centers_mm=inlet_centers,
            outlet_center_mm=outlet_center,
            skin_naris_centers_mm=skin_naris_centers or None,
            n_per_naris=max(14, n_streamline_seeds // 2),
            max_steps=1200,
            step_mm=0.3,
            reach_outlet_mm=14.0,
        )
        # Complete short CFD traces to trachea along the anatomical centerline
        if centerline_mm is not None and outlet_center is not None and lines:
            before = len(lines)
            lines = extend_paths_to_outlet_via_centerline(
                lines,
                np.asarray(centerline_mm, dtype=float),
                outlet_center,
                max_end_dist_mm=14.0,
            )
            notes.append(
                f"Extended {before} paths to trachea via passage centerline "
                "(CFD weak near caudal outlet)."
            )
        notes.append(
            f"Inhale streamlines: {len(lines)} paths on passage lumen; "
            f"skin nares={len(skin_naris_centers)}; trachea={outlet_center}."
        )
    else:
        lines = compute_streamlines(
            ux.astype(float),
            uy.astype(float),
            uz.astype(float),
            domain,
            spacing,
            origin,
            np.array(seed_pts, dtype=float) if seed_pts else np.zeros((0, 3)),
            max_steps=600,
            step_mm=0.4,
        )
        notes.append(f"Streamlines: {len(lines)} traces (no inlet centers found).")

    # Write NPZ (viewer-compatible keys)
    out_npz = outputs_root / f"{case_id}_flow.npz"
    # Keep a backup of potential-flow if present and different method
    backup = outputs_root / f"{case_id}_flow_potential.npz"
    meta_old = outputs_root / f"{case_id}_flow_meta.json"
    if out_npz.is_file() and meta_old.is_file():
        try:
            old = json.loads(meta_old.read_text(encoding="utf-8"))
            if "openfoam" not in str(old.get("method", "")).lower():
                if not backup.is_file():
                    out_npz.replace(backup)
                    notes.append(f"Backed up prior potential-flow NPZ → {backup.name}")
        except Exception:
            pass

    np.savez_compressed(
        out_npz,
        airway=airway.astype(np.uint8),
        speed=speed.astype(np.float32),
        ux=ux,
        uy=uy,
        uz=uz,
        pressure=pressure.astype(np.float32),
        spacing_xyz_mm=np.array(spacing, dtype=np.float64),
        origin_xyz_mm=np.array(origin, dtype=np.float64),
        inlet_mask=inlet_mask.astype(np.uint8),
        outlet_mask=outlet_mask.astype(np.uint8),
        mapped_mask=mapped.astype(np.uint8),
    )

    sl_path = outputs_root / f"{case_id}_streamlines.json"
    with sl_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "case_id": case_id,
                "source": "openfoam",
                "time": time_name,
                "lines": [line.tolist() for line in lines],
            },
            f,
        )

    meta = {
        "case_id": case_id,
        "method": "openfoam_simpleFoam",
        "openfoam_time": time_name,
        "foam_case": str(foam_root),
        "n_cells": n_cells,
        "n_mapped_voxels": int(mapped.sum()),
        "max_speed_m_s": max_speed,
        "mean_speed_m_s": mean_speed,
        "target_flow_L_per_min": 18.0,
        "mesh_bbox_volume_m3": mesh_vol,
        "notes": notes,
    }
    # merge BC flow if present
    if bc_path.is_file():
        bc = json.loads(bc_path.read_text(encoding="utf-8"))
        meta["target_flow_L_per_min"] = float(
            bc.get("flow_assignment", {}).get("total_inflow_L_per_min", 18.0)
        )
    meta_path = outputs_root / f"{case_id}_flow_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[{case_id}] Wrote {out_npz}")
    print(f"[{case_id}] max|U|={max_speed:.4f} m/s  mean={mean_speed:.4f} m/s  mapped={int(mapped.sum())}")

    return OpenFoamImportResult(
        case_id=case_id,
        time_name=time_name,
        n_cells=n_cells,
        n_mapped_voxels=int(mapped.sum()),
        max_speed_m_s=max_speed,
        mean_speed_m_s=mean_speed,
        mesh_volume_m3=mesh_vol,
        notes=notes,
        out_npz=str(out_npz),
    )
