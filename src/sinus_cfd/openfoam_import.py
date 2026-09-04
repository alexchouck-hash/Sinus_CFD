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
    pressure_drop: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# simpleFoam is incompressible and solves KINEMATIC pressure (m^2/s^2); multiply
# by density to report pascals.
RHO_AIR_KG_M3 = 1.2
# The quantity a surgeon acts on is the pressure drop, not the p residual. A
# solve counts as converged when the inlet pressure has stopped MOVING. Two
# statistics, both always reported, each with its own bound:
#   drift  -- mean of the last DP_STABLE_SAMPLES against the mean of the ones
#             before them. A trend fails this; oscillation averages out.
#   wobble -- largest relative change within the last window. The amplitude.
# Measured on five solved cases: drift 0.05-0.52%, wobble 0.26-2.06%. THCA sat
# at wobble 1.53% for 400 extra iterations while its resistance moved 0.6% --
# a per-sample rule at 1% called that unconverged; the drift rule (0.08%) does
# not. The choked-outlet guard is separate and catches the case whose wobble
# was harmless but whose cap was not (VH female, 41x).
DP_DRIFT_MAX_REL = 0.01
DP_WOBBLE_MAX_REL = 0.03
DP_STABLE_MAX_REL = DP_DRIFT_MAX_REL   # legacy name
DP_STABLE_SAMPLES = 3


def _read_surface_field_value(foam_root: Path, name: str) -> list[tuple[float, float]]:
    """(time, value) rows from postProcessing/<name>/*/surfaceFieldValue.dat."""
    rows: list[tuple[float, float]] = []
    base = foam_root / "postProcessing" / name
    if not base.is_dir():
        return rows
    # OpenFOAM does not overwrite an existing surfaceFieldValue.dat; a re-run in
    # a case that still holds an old history writes surfaceFieldValue_0.dat,
    # _1.dat, ... beside it. Globbing only the bare name read a July history for
    # a September solve on VisibleHuman_Male_Head and called it settled. Take
    # the NEWEST file, and only that one.
    dats = sorted(base.rglob("surfaceFieldValue*.dat"), key=lambda p: p.stat().st_mtime)
    for dat in dats[-1:]:
        for line in dat.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (IndexError, ValueError):
                continue
    rows.sort(key=lambda r: r[0])
    return rows


def pressure_drop_verdict(foam_root: Path) -> dict[str, Any] | None:
    """Pressure drop, resistance and whether they had SETTLED when the run ended.

    Reads the surfaceFieldValue history the scaffold writes every writeInterval.
    Returns None when that history is absent (older cases). Raises when the
    history exists but the inlet pressure was still moving by more than
    ``DP_STABLE_MAX_REL`` over the last ``DP_STABLE_SAMPLES`` samples -- that is
    a solve that stopped before it converged, and its resistance is not a
    measurement.

    Why this and not the p residual: on CQ500CT390 the first-corrector p
    residual plateaued at 3.8e-3 against a 1e-3 residualControl and never
    tripped it, while the inlet pressures were flat to 0.17% from iteration 50
    onward. The residual floor was set by two highly-skew boundary faces where
    the flat nostril cap meets the curved wall; the answer had long stopped
    changing. Judging convergence on the residual would have called 40 minutes
    of extra iterations necessary. They were not.
    """
    p_l = _read_surface_field_value(foam_root, "p_left_nostril")
    p_r = _read_surface_field_value(foam_root, "p_right_nostril")
    p_o = _read_surface_field_value(foam_root, "p_trachea")
    q_o = _read_surface_field_value(foam_root, "Q_trachea")
    if not (p_l and p_r and p_o and q_o):
        return None

    def last(rows, n):
        return [v for _t, v in rows[-n:]]

    def rel_change(vals):
        ref = max(abs(vals[-1]), 1e-12)
        return max(abs(v - vals[-1]) for v in vals) / ref

    def drift(rows, n):
        # mean of the last n samples against the mean of the n before them:
        # the answer is still moving if these disagree. Oscillation between
        # samples averages out; a trend does not.
        vals = [v for _t, v in rows]
        if len(vals) < 2 * n:
            return float("nan")
        a = float(np.mean(vals[-n:]))
        b = float(np.mean(vals[-2 * n:-n]))
        return abs(a - b) / max(abs(a), 1e-12)

    n = DP_STABLE_SAMPLES
    rel_l = rel_change(last(p_l, n))
    rel_r = rel_change(last(p_r, n))
    worst = max(rel_l, rel_r)                       # wobble: amplitude within the window
    drift_worst = max(drift(p_l, n), drift(p_r, n))  # drift: is the window still moving

    pl, pr, po = p_l[-1][1], p_r[-1][1], p_o[-1][1]
    q_m3_s = abs(q_o[-1][1])
    dp_kin = 0.5 * (pl + pr) - po
    dp_pa = RHO_AIR_KG_M3 * dp_kin
    q_ml_s = q_m3_s * 1e6
    r_pa_s_ml = dp_pa / q_ml_s if q_ml_s > 0 else float("nan")
    out = {
        "time": p_l[-1][0],
        "n_samples": len(p_l),
        "p_left_kin": pl, "p_right_kin": pr, "p_outlet_kin": po,
        "dp_pa": dp_pa,
        "dp_left_pa": RHO_AIR_KG_M3 * (pl - po),
        "dp_right_pa": RHO_AIR_KG_M3 * (pr - po),
        "q_m3_s": q_m3_s,
        "q_L_min": q_m3_s * 6e4,
        "resistance_pa_s_per_ml": r_pa_s_ml,
        "stability_rel_change_last_samples": worst,   # kept: this is the wobble
        "wobble_rel_change": worst,
        "drift_rel_change": drift_worst,
        "stability_samples": n,
        "stable": False,
    }
    drift_ok = (drift_worst != drift_worst) or drift_worst <= DP_DRIFT_MAX_REL  # nan: too few samples to judge drift
    wobble_ok = worst <= DP_WOBBLE_MAX_REL
    out["stable"] = bool(drift_ok and wobble_ok)
    if not out["stable"]:
        why = []
        if not drift_ok:
            why.append(f"the window mean is still DRIFTING {100 * drift_worst:.2f}% "
                       f"(limit {100 * DP_DRIFT_MAX_REL:.1f}%)")
        if not wobble_ok:
            why.append(f"inlet pressure WOBBLES {100 * worst:.2f}% within the last {n} "
                       f"samples (limit {100 * DP_WOBBLE_MAX_REL:.1f}%)")
        raise ValueError(
            f"simpleFoam stopped before the pressure drop settled: {'; '.join(why)}. "
            f"The resistance {r_pa_s_ml:.4f} Pa*s/mL is not a measurement; raise "
            f"endTime or inspect the mesh. History: {foam_root / 'postProcessing'}"
        )
    return out


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


# Largest face speed on the outlet patch, as a multiple of the patch mean.
# Measured populations, every solve in this repo:
#   clean caps      1.8, 2.5, 3.0, 3.5x      (CQ500CT390, VH male, P001, THCA)
#   choked caps     8x, 8.9x, 41x, 57x        (VH male old trachea -- never settled;
#                                              VH female choana on the sinus-free
#                                              domain -- 119 faces above 5x, dP 434 Pa;
#                                              VH female choana at 800 iters; VH
#                                              female old trachea)
# 5x separates them with the nearest members 1.4x apart on either side. The
# earlier 10x let the 8.9x case through and reported its 434 Pa as settled.
OUTLET_HOT_FACE_MAX_RATIO = 5.0


def outlet_patch_velocity_stats(u_path: Path, patch: str) -> dict[str, float] | None:
    """Per-face |U| statistics on one boundary patch of a volVectorField.

    Returns None when the patch has no nonuniform value list (uniform, or not
    written). This is the post-solve test the pre-solve geometry could not
    provide: the cap's shape (PCA axes) and its position (touching the image
    boundary) both failed to separate the choked Visible Human trachea outlets
    from clean ones -- every cap is a fat 6 mm ball clipped by the lumen, and
    none touches the volume edge. The solved field separates them at 15x.
    """
    text = _strip_foam_comments(u_path.read_text(encoding="utf-8", errors="replace"))
    m = re.search(r"boundaryField\s*\{(.*)\}\s*$", text, re.S)
    body = m.group(1) if m else text
    pm = re.search(r"\b" + re.escape(patch) + r"\s*\{(.*?)\n\s*\}", body, re.S)
    if not pm:
        return None
    vm = re.search(r"value\s+nonuniform\s+List<vector>\s*(\d+)\s*\((.*?)\)\s*;", pm.group(1), re.S)
    if not vm:
        return None
    vals = np.array(
        re.findall(r"\(\s*([-+eE0-9.]+)\s+([-+eE0-9.]+)\s+([-+eE0-9.]+)\s*\)", vm.group(2)),
        dtype=float,
    )
    if vals.size == 0:
        return None
    speed = np.linalg.norm(vals, axis=1)
    mean = float(speed.mean())
    ratio = float(speed.max() / mean) if mean > 0 else float("inf")
    return {
        "n_faces": int(len(speed)),
        "max_m_s": float(speed.max()),
        "mean_m_s": mean,
        "max_over_mean": ratio,
        "hot_faces": int((speed > 5.0 * mean).sum()) if mean > 0 else int(len(speed)),
    }


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


def _time_dirs_desc(foam_case: Path) -> list[str]:
    """Numeric time directories that hold a U field, newest first."""
    out: list[tuple[float, str]] = []
    for p in foam_case.iterdir():
        if not p.is_dir() or p.name in ("constant", "system"):
            continue
        if not (p / "U").is_file():
            continue
        try:
            out.append((float(p.name), p.name))
        except ValueError:
            continue
    return [n for _t, n in sorted(out, reverse=True)]


def select_time_dir_matching_cells(foam_case: Path, n_cells: int) -> str | None:
    """Latest time directory whose U field has exactly ``n_cells`` entries.

    A case directory that was solved more than once can hold time directories
    from different meshes. The newest number is not the newest run: OpenFOAM
    leaves old time directories in place, so a re-solve to endTime 400 sits
    beside an earlier run's 500/. The field that fits the mesh is the one from
    this mesh. None when no time directory fits.
    """
    for name in _time_dirs_desc(foam_case):
        if float(name) == 0.0:
            continue  # the initial condition is not a solve
        try:
            u = read_foam_vector_field(foam_case / name / "U")
        except Exception:
            continue
        if u.shape[0] == 1:
            continue  # a uniform field fits every mesh and proves nothing
        if u.shape[0] == n_cells:
            return name
    return None


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


def persist_outlet_viewer_marker(bc_path: Path, marker_mm) -> dict[str, Any]:
    """Record where the viewer should draw the outlet -- WITHOUT moving the outlet.

    ``ports[].center_mm`` is the domain definition: ``analyze_passage`` seeds the
    outlet_open cap from the lumen voxel nearest to it, and export, scaffold and
    the solve all descend from that cap. This function used to overwrite it with
    the *centroid* of the previous cap "for viewer labels". The lumen clips the
    cap asymmetrically, so its centroid sits anterior of its own seed; writing it
    back moved the seed, the next cap grew from the new seed, and its centroid
    sat further forward still. On CQ500CT390 one import walked the seed 3 voxels
    (1.1 mm) anteriorly and the outlet cap from 1,780 to 2,450 voxels -- the CFD
    domain changed because a plot label was saved. Proven by rebuilding the cap
    from candidate seeds: only (27,232,220), the pre-import seed, reproduces the
    solved geometry byte for byte.

    The marker now lives in its own field. ``center_mm`` is never touched here.
    """
    bc = json.loads(bc_path.read_text(encoding="utf-8"))
    touched = []
    for port in bc.get("ports", []):
        if port.get("role") == "outlet" or port.get("name") == "trachea":
            port["viewer_marker_mm"] = [float(v) for v in marker_mm]
            port["viewer_marker_method"] = "passage_outlet_open_centroid"
            port["viewer_marker_note"] = (
                "Display position only. The outlet the solver used is center_mm."
            )
            touched.append(port.get("name"))
    bc_path.write_text(json.dumps(bc, indent=2), encoding="utf-8")
    return {"ports": touched, "marker_mm": [float(v) for v in marker_mm]}


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
        # A field that does not fit the mesh is from ANOTHER mesh. This used to
        # truncate or pad and carry on: on VisibleHuman_Male_Head a September
        # solve (127,198 cells, written to 400/) sat beside July's untouched
        # 500/ (240,479 cells); latest_time_dir took 500, and July's velocities
        # were padded onto the new mesh and reported as the new result.
        picked = select_time_dir_matching_cells(foam_root, n_cells)
        if picked is None:
            raise ValueError(
                f"[{case_id}] no time directory holds a U field with {n_cells} "
                f"entries (mesh cell count); latest '{time_name}' has {U.shape[0]}. "
                "The solve and the mesh in this case directory are from different "
                "runs. Clean stale time directories (Allclean) and re-solve."
            )
        notes.append(
            f"time '{time_name}' has {U.shape[0]} U entries for a {n_cells}-cell mesh "
            f"(stale run); using '{picked}', the latest time whose field fits."
        )
        time_name = picked
        u_path = foam_root / time_name / "U"
        U = read_foam_vector_field(u_path)
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

    # Viewer marker for the outlet. NEVER written into center_mm -- see the
    # helper's docstring for the ratchet that caused.
    if bc_path.is_file() and outlet_center is not None:
        try:
            persist_outlet_viewer_marker(bc_path, outlet_center)
            notes.append(
                "Outlet viewer marker written to BC ports[].viewer_marker_mm "
                "(center_mm left untouched)."
            )
        except Exception as exc:
            notes.append(f"Could not write outlet viewer marker: {exc}")

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

    # Outlet patch health, read off the solved field. A choked cap puts the whole
    # pressure drop on a few faces; the number that comes out is then about the
    # cap, not the airway, and must not be reported.
    outlet_patch = None
    outlet_name = "trachea"
    if bc_path.is_file():
        try:
            outlet_name = str(json.loads(bc_path.read_text(encoding="utf-8")).get("outlet_name") or "trachea")
        except Exception:
            pass
    # The scaffold always names the mesh outlet patch "trachea" whatever the BC
    # calls the port (P001's BC says trachea_outlet_proxy). Try the mesh name
    # first, then the BC name; and never stay silent when neither reads.
    outlet_patch_name = None
    for cand in dict.fromkeys(["trachea", outlet_name]):
        try:
            stats = outlet_patch_velocity_stats(u_path, cand)
        except Exception as exc:
            notes.append(f"Could not read outlet patch '{cand}': {exc}")
            stats = None
        if stats is not None:
            outlet_patch, outlet_patch_name = stats, cand
            break
    if outlet_patch is None:
        notes.append(
            f"WARNING: outlet patch velocities not readable (tried "
            f"{list(dict.fromkeys(['trachea', outlet_name]))}); the choked-outlet guard "
            "was NOT applied to this import."
        )
    else:
        outlet_name = outlet_patch_name
    if outlet_patch is not None:
        if outlet_patch["max_over_mean"] > OUTLET_HOT_FACE_MAX_RATIO:
            raise ValueError(
                f"[{case_id}] outlet patch '{outlet_name}' is choked: max face speed is "
                f"{outlet_patch['max_over_mean']:.0f}x the patch mean "
                f"({outlet_patch['max_m_s']:.1f} vs {outlet_patch['mean_m_s']:.2f} m/s, "
                f"{outlet_patch['hot_faces']} faces above 5x; limit "
                f"{OUTLET_HOT_FACE_MAX_RATIO:.0f}x). The pressure drop is set by the cap, "
                "not the airway. Move the outlet (auto_process_head --outlet nasopharynx) "
                "and re-solve."
            )
        notes.append(
            f"Outlet patch '{outlet_name}': {outlet_patch['n_faces']} faces, max/mean "
            f"{outlet_patch['max_over_mean']:.1f}x, {outlet_patch['hot_faces']} above 5x -- clean."
        )

    pressure_drop = pressure_drop_verdict(foam_root)
    if pressure_drop is not None:
        pressure_drop["outlet_patch"] = outlet_patch
    if pressure_drop is None:
        notes.append(
            "WARNING: no postProcessing/p_*/surfaceFieldValue.dat history; "
            "pressure-drop convergence UNVERIFIED for this import."
        )
    else:
        notes.append(
            f"Pressure drop settled: drift "
            f"{100 * pressure_drop['drift_rel_change']:.2f}%, wobble "
            f"{100 * pressure_drop['wobble_rel_change']:.2f}% over "
            f"the last {pressure_drop['stability_samples']} samples. "
            f"dP={pressure_drop['dp_pa']:.2f} Pa at "
            f"{pressure_drop['q_L_min']:.1f} L/min -> "
            f"R={pressure_drop['resistance_pa_s_per_ml']:.4f} Pa*s/mL "
            f"(L {pressure_drop['dp_left_pa']:.2f} / R {pressure_drop['dp_right_pa']:.2f} Pa)."
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
        "pressure_drop": pressure_drop,
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
        pressure_drop=pressure_drop,
    )
