#!/usr/bin/env python3
"""
Sinus_CFD interactive viewer.

- Tri-planar CT/speed slices with sliders
- Semi-transparent 3D airway cavity
- Curved streamlines + velocity glyphs

Run from repo root:
  py -3.12 -m pip install -r requirements.txt
  py -3.12 -m streamlit run app/viewer.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Bump when viewer behavior or expected data layout changes (shown in UI).
APP_VERSION = "0.12.0-dual-frontal-curvy-main"
APP_VERSION_LABEL = (
    "dual purple L/R naris→frontal · curvy inhale pathlines → trachea · magenta removal"
)

DEFAULT_CASE = "P001"
OUTPUTS = REPO_ROOT / "outputs"


def _file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def case_data_fingerprint(case_id: str) -> str:
    """Stamp from key output mtimes so cache invalidates after reprocess."""
    case_dir = OUTPUTS / case_id
    keys = [
        f"{case_id}_flow.npz",
        f"{case_id}_flow_meta.json",
        f"{case_id}_streamlines.json",
        f"{case_id}_boundary_conditions.json",
        f"{case_id}_stats.json",
        f"{case_id}_skin.stl",
        f"{case_id}_airway.stl",
        f"{case_id}_nares.json",
        f"{case_id}_passage.json",
        f"{case_id}_open_paths.json",
        f"{case_id}_restriction.npz",
        f"{case_id}_surgical_guidance.json",
        f"{case_id}_removal_highlight.npz",
        f"{case_id}_sinus_frontal.stl",
        f"{case_id}_septum.stl",
        f"{case_id}_cavity_left.stl",
        f"{case_id}_cavity_right.stl",
        f"{case_id}_ct_nasal_meta.json",
    ]
    return "|".join(f"{n}:{int(_file_mtime(case_dir / n))}" for n in keys)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading flow field…")
def load_case(case_id: str, data_fingerprint: str) -> dict:
    """data_fingerprint must change when on-disk outputs are reprocessed."""
    case_dir = OUTPUTS / case_id
    npz_path = case_dir / f"{case_id}_flow.npz"
    if not npz_path.is_file():
        return {
            "error": (
                f"Missing {npz_path}. Run: "
                f"py -3.12 scripts/process_whole_head.py --case {case_id}"
            )
        }

    data = np.load(npz_path)
    out = {
        "case_id": case_id,
        "data_fingerprint": data_fingerprint,
        "airway": data["airway"].astype(bool),
        "speed": data["speed"].astype(np.float32),
        "ux": data["ux"].astype(np.float32),
        "uy": data["uy"].astype(np.float32),
        "uz": data["uz"].astype(np.float32),
        "pressure": data["pressure"].astype(np.float32),
        "spacing": data["spacing_xyz_mm"].astype(float),
        "origin": data["origin_xyz_mm"].astype(float),
        "inlet": data["inlet_mask"].astype(bool),
        "outlet": data["outlet_mask"].astype(bool),
    }

    sl_path = case_dir / f"{case_id}_streamlines.json"
    if sl_path.is_file():
        with sl_path.open(encoding="utf-8") as f:
            sl = json.load(f)
        out["streamlines"] = sl.get("lines") or []
        out["streamline_speeds"] = sl.get("speeds_m_s") or []
    else:
        out["streamlines"] = []
        out["streamline_speeds"] = []

    meta_path = case_dir / f"{case_id}_flow_meta.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            out["meta"] = json.load(f)
    else:
        out["meta"] = {}

    open_paths = case_dir / f"{case_id}_open_paths.json"
    if open_paths.is_file():
        with open_paths.open(encoding="utf-8") as f:
            out["open_paths"] = json.load(f)
    else:
        out["open_paths"] = {}

    rest_path = case_dir / f"{case_id}_restriction.npz"
    if rest_path.is_file():
        rd = np.load(rest_path)
        out["restriction_pts"] = rd["points_xyz_score_r_mm"].astype(np.float32)
        out["restriction_thr"] = float(rd["threshold_1_over_r"]) if "threshold_1_over_r" in rd else 0.0
        out["restriction_min_r"] = float(rd["min_radius_mm"]) if "min_radius_mm" in rd else 0.0
    else:
        out["restriction_pts"] = None
        out["restriction_thr"] = 0.0
        out["restriction_min_r"] = 0.0

    nares_path = case_dir / f"{case_id}_nares.json"
    if nares_path.is_file():
        with nares_path.open(encoding="utf-8") as f:
            out["nares"] = json.load(f)
    else:
        out["nares"] = {}

    surg_path = case_dir / f"{case_id}_surgical_guidance.json"
    if surg_path.is_file():
        with surg_path.open(encoding="utf-8") as f:
            out["surgical"] = json.load(f)
    else:
        out["surgical"] = {}

    rem_npz = case_dir / f"{case_id}_removal_highlight.npz"
    if rem_npz.is_file():
        rd = np.load(rem_npz)
        out["removal_pts"] = rd["points_xyz_r_mm"].astype(np.float32)
    else:
        out["removal_pts"] = None

    for key, fname in (
        ("frontal_stl", f"{case_id}_sinus_frontal.stl"),
        ("sphenoid_stl", f"{case_id}_sinus_sphenoid.stl"),
        ("maxillary_left_stl", f"{case_id}_sinus_maxillary_left.stl"),
        ("maxillary_right_stl", f"{case_id}_sinus_maxillary_right.stl"),
    ):
        fp = case_dir / fname
        out[key] = str(fp) if fp.is_file() else None

    stl_path = case_dir / f"{case_id}_airway.stl"
    out["stl_path"] = str(stl_path) if stl_path.is_file() else None
    head_stl = case_dir / f"{case_id}_head.stl"
    out["head_stl_path"] = str(head_stl) if head_stl.is_file() else None
    skin_stl = case_dir / f"{case_id}_skin.stl"
    out["skin_stl_path"] = str(skin_stl) if skin_stl.is_file() else None
    bone_stl = case_dir / f"{case_id}_bone.stl"
    out["bone_stl_path"] = str(bone_stl) if bone_stl.is_file() else None
    septum_stl = case_dir / f"{case_id}_septum.stl"
    out["septum_stl_path"] = str(septum_stl) if septum_stl.is_file() else None
    left_stl = case_dir / f"{case_id}_cavity_left.stl"
    out["left_cavity_stl_path"] = str(left_stl) if left_stl.is_file() else None
    right_stl = case_dir / f"{case_id}_cavity_right.stl"
    out["right_cavity_stl_path"] = str(right_stl) if right_stl.is_file() else None
    mucosa_stl = case_dir / f"{case_id}_mucosa_wall.stl"
    out["mucosa_stl_path"] = str(mucosa_stl) if mucosa_stl.is_file() else None
    ct_meta = case_dir / f"{case_id}_ct_nasal_meta.json"
    if ct_meta.is_file():
        with ct_meta.open(encoding="utf-8") as f:
            out["ct_nasal"] = json.load(f)
    else:
        out["ct_nasal"] = {}

    bc_path = case_dir / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        with bc_path.open(encoding="utf-8") as f:
            out["bc"] = json.load(f)
        out["bc_mtime"] = datetime.fromtimestamp(
            _file_mtime(bc_path), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        out["bc"] = {}
        out["bc_mtime"] = None

    stats_path = case_dir / f"{case_id}_stats.json"
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8") as f:
            out["stats"] = json.load(f)
        out["stats_mtime"] = datetime.fromtimestamp(
            _file_mtime(stats_path), tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        out["stats"] = {}
        out["stats_mtime"] = None

    face_qc = case_dir / f"{case_id}_preview_face_nares.png"
    out["face_qc_path"] = str(face_qc) if face_qc.is_file() else None
    preview_path = case_dir / f"{case_id}_preview.png"
    out["preview_path"] = str(preview_path) if preview_path.is_file() else None

    passage_path = case_dir / f"{case_id}_passage.json"
    if passage_path.is_file():
        with passage_path.open(encoding="utf-8") as f:
            out["passage"] = json.load(f)
    else:
        out["passage"] = {}

    return out


def list_cases() -> list[str]:
    if not OUTPUTS.is_dir():
        return []
    cases = []
    for d in sorted(OUTPUTS.iterdir()):
        if d.is_dir() and (d / f"{d.name}_flow.npz").is_file():
            cases.append(d.name)
    return cases


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _slice_fig(
    speed: np.ndarray,
    airway: np.ndarray,
    iz: int,
    iy: int,
    ix: int,
    spacing: np.ndarray,
    vmax: float,
) -> go.Figure:
    """Tri-planar speed maps (axial / coronal / sagittal)."""
    # speed shape (z,y,x)
    axial = np.where(airway[iz], speed[iz], np.nan)
    coronal = np.where(airway[:, iy, :], speed[:, iy, :], np.nan)
    sagittal = np.where(airway[:, :, ix], speed[:, :, ix], np.nan)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"Axial  z={iz}",
            f"Coronal  y={iy}",
            f"Sagittal  x={ix}",
        ),
        horizontal_spacing=0.06,
    )

    common = dict(
        colorscale="Turbo",
        zmin=0,
        zmax=max(vmax, 1e-9),
        colorbar=dict(title="|u| m/s", x=1.02),
    )

    fig.add_trace(
        go.Heatmap(z=axial, **common, showscale=True),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Heatmap(z=coronal, **{**common, "showscale": False}),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Heatmap(z=sagittal, **{**common, "showscale": False}),
        row=1,
        col=3,
    )

    # Crosshairs
    for col, (h, v) in enumerate(
        [(iy, ix), (iz, ix), (iz, iy)], start=1
    ):
        fig.add_hline(y=h, line_width=1, line_color="white", opacity=0.5, row=1, col=col)
        fig.add_vline(x=v, line_width=1, line_color="white", opacity=0.5, row=1, col=col)

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=60, t=40, b=20),
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
    )
    fig.update_xaxes(showticklabels=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(showticklabels=False)
    # Match anatomical feel: axial origin lower
    fig.update_yaxes(autorange="reversed", row=1, col=1)
    return fig


def _load_mesh_decimated(stl_path: str, target_faces: int = 14000):
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        except Exception:
            idx = np.linspace(0, len(mesh.faces) - 1, target_faces, dtype=int)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)
    return mesh


def _mesh_wireframe_trace(mesh, max_edges: int = 8000, color: str = "#d8e6f0") -> go.Scatter3d | None:
    """Dark-on-light edge overlay so the cavity silhouette reads clearly."""
    try:
        edges = mesh.edges_unique
    except Exception:
        return None
    if edges is None or len(edges) == 0:
        return None
    if len(edges) > max_edges:
        idx = np.linspace(0, len(edges) - 1, max_edges, dtype=int)
        edges = edges[idx]
    v = mesh.vertices
    # NaN breaks create separate segments
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [v[a, 0], v[b, 0], None]
        ys += [v[a, 1], v[b, 1], None]
        zs += [v[a, 2], v[b, 2], None]
    return go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="lines",
        line=dict(color=color, width=1.5),
        name="Cavity edges",
        hoverinfo="skip",
        opacity=0.55,
    )


def _add_surface_mesh(
    fig: go.Figure,
    mesh,
    *,
    name: str,
    color: str,
    opacity: float,
    show_wireframe: bool,
    edge_color: str,
    wireframe_max_edges: int = 6000,
    is_skin: bool = False,
) -> None:
    v = mesh.vertices
    f = mesh.faces
    # Solid surface (not a point cloud): Mesh3d with lighting.
    # Skin gets slightly stronger lighting so the head silhouette reads clearly.
    if is_skin:
        lighting = dict(
            ambient=0.42,
            diffuse=0.95,
            specular=0.55,
            roughness=0.35,
            fresnel=0.25,
        )
        lightpos = dict(x=120, y=180, z=280)
    else:
        lighting = dict(
            ambient=0.55,
            diffuse=0.85,
            specular=0.35,
            roughness=0.45,
            fresnel=0.15,
        )
        lightpos = dict(x=80, y=120, z=200)
    mesh_kwargs = dict(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1],
        k=f[:, 2],
        color=color,
        opacity=float(np.clip(opacity, 0.02, 1.0)),
        name=name,
        flatshading=False,
        lighting=lighting,
        lightposition=lightpos,
        hoverinfo="skip",
        showlegend=True,
    )
    fig.add_trace(go.Mesh3d(**mesh_kwargs))
    if show_wireframe:
        wf = _mesh_wireframe_trace(mesh, max_edges=wireframe_max_edges, color=edge_color)
        if wf is not None:
            wf.name = f"{name} edges"
            wf.opacity = 0.75 if is_skin else 0.55
            fig.add_trace(wf)


def _sample_speed_along_line(
    line: np.ndarray,
    speed: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbor sample of |u| (m/s) at each pathline vertex."""
    ox, oy, oz = origin
    sx, sy, sz = spacing
    nz, ny, nx = speed.shape
    ix = np.clip(np.rint((line[:, 0] - ox) / sx).astype(int), 0, nx - 1)
    iy = np.clip(np.rint((line[:, 1] - oy) / sy).astype(int), 0, ny - 1)
    iz = np.clip(np.rint((line[:, 2] - oz) / sz).astype(int), 0, nz - 1)
    return speed[iz, iy, ix].astype(float)


def _fig_3d(
    mesh,
    head_mesh,
    skin_mesh,
    bone_mesh,
    streamlines: list,
    streamline_speeds: list | None,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    speed: np.ndarray,
    airway: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    show_head: bool,
    head_opacity: float,
    show_skin: bool,
    skin_opacity: float,
    show_bone: bool,
    bone_opacity: float,
    show_mesh: bool,
    mesh_opacity: float,
    show_wireframe: bool,
    show_streamlines: bool,
    streamline_width: float,
    streamline_opacity: float,
    max_pathlines: int,
    show_vectors: bool,
    vector_stride: int,
    max_vector_speed: float,
    ports: list[dict],
    bg_mode: str = "dark",
    max_vectors: int = 6000,
    centerline_mm: list | None = None,
    centerline_left_mm: list | None = None,
    centerline_right_mm: list | None = None,
    show_centerlines: bool = False,
    left_mesh=None,
    right_mesh=None,
    septum_mesh=None,
    mucosa_mesh=None,
    show_left: bool = False,
    show_right: bool = False,
    show_septum: bool = False,
    show_mucosa: bool = False,
    left_opacity: float = 0.5,
    right_opacity: float = 0.5,
    septum_opacity: float = 0.9,
    mucosa_opacity: float = 0.35,
    restriction_pts: np.ndarray | None = None,
    show_restriction: bool = True,
    animate_pathlines: bool = False,
    n_anim_frames: int = 24,
    frontal_path_mm: list | None = None,
    frontal_path_left_mm: list | None = None,
    frontal_path_right_mm: list | None = None,
    show_frontal_path: bool = False,
    removal_pts: np.ndarray | None = None,
    show_removal: bool = False,
    frontal_mesh=None,
    sphenoid_mesh=None,
    max_l_mesh=None,
    max_r_mesh=None,
    show_frontal_sinus: bool = False,
    show_sphenoid: bool = False,
    show_maxillary: bool = False,
    sinus_opacity: float = 0.35,
) -> go.Figure:
    fig = go.Figure()
    dark = bg_mode == "dark"
    scene_bg = "#1a1f2b" if dark else "#f4f6f8"
    paper_bg = "#0e1117" if dark else "#ffffff"
    font_c = "#fafafa" if dark else "#111111"
    mesh_color = "#5ec8ff"  # airway cyan
    head_color = "#c4a484"  # soft tissue volume
    skin_color = "#e8b896"  # outer skin surface
    bone_color = "#f0e6d8"  # bone
    edge_color = "#e8f1f8" if dark else "#1a3a52"
    head_edge = "#8b7355" if dark else "#5c4033"
    stream_width = float(streamline_width)

    # --- Outer skin surface (preferred visual for head shape) ---
    if show_skin and skin_mesh is not None:
        _add_surface_mesh(
            fig,
            skin_mesh,
            name="Skin surface",
            color=skin_color,
            opacity=skin_opacity,
            show_wireframe=show_wireframe,
            edge_color="#3d2314" if not dark else "#ffd7b0",
            wireframe_max_edges=18000,
            is_skin=True,
        )
    elif show_head and head_mesh is not None:
        _add_surface_mesh(
            fig,
            head_mesh,
            name="Soft tissue / head",
            color=head_color,
            opacity=head_opacity,
            show_wireframe=False,
            edge_color=head_edge,
        )

    # --- Bone ---
    if show_bone and bone_mesh is not None:
        _add_surface_mesh(
            fig,
            bone_mesh,
            name="Bone",
            color=bone_color,
            opacity=bone_opacity,
            show_wireframe=False,
            edge_color="#dddddd",
        )

    # --- Airway cavity (combined air space) ---
    if show_mesh and mesh is not None and not (show_left or show_right):
        _add_surface_mesh(
            fig,
            mesh,
            name="Air space",
            color=mesh_color,
            opacity=mesh_opacity,
            show_wireframe=show_wireframe,
            edge_color=edge_color,
            wireframe_max_edges=5000,
        )

    # --- L/R cavities (CT-native; septum is the gap/wall between them) ---
    if show_left and left_mesh is not None:
        _add_surface_mesh(
            fig,
            left_mesh,
            name="Left nasal cavity",
            color="#4fc3f7",
            opacity=left_opacity,
            show_wireframe=False,
            edge_color=edge_color,
        )
    if show_right and right_mesh is not None:
        _add_surface_mesh(
            fig,
            right_mesh,
            name="Right nasal cavity",
            color="#81d4fa",
            opacity=right_opacity,
            show_wireframe=False,
            edge_color=edge_color,
        )
    if show_septum and septum_mesh is not None:
        _add_surface_mesh(
            fig,
            septum_mesh,
            name="Nasal septum",
            color="#ff8a65",
            opacity=septum_opacity,
            show_wireframe=False,
            edge_color="#bf360c",
        )
    if show_mucosa and mucosa_mesh is not None:
        _add_surface_mesh(
            fig,
            mucosa_mesh,
            name="Mucosa / walls",
            color="#ce93d8",
            opacity=mucosa_opacity,
            show_wireframe=False,
            edge_color="#6a1b9a",
        )

    # --- Paranasal sinuses (labeled) ---
    if show_frontal_sinus and frontal_mesh is not None:
        _add_surface_mesh(
            fig,
            frontal_mesh,
            name="Frontal sinus",
            color="#ffb74d",
            opacity=sinus_opacity,
            show_wireframe=False,
            edge_color="#e65100",
        )
    if show_sphenoid and sphenoid_mesh is not None:
        _add_surface_mesh(
            fig,
            sphenoid_mesh,
            name="Sphenoid sinus",
            color="#4db6ac",
            opacity=sinus_opacity,
            show_wireframe=False,
            edge_color="#00695c",
        )
    if show_maxillary:
        if max_l_mesh is not None:
            _add_surface_mesh(
                fig,
                max_l_mesh,
                name="Maxillary sinus (L)",
                color="#64b5f6",
                opacity=sinus_opacity,
                show_wireframe=False,
                edge_color="#1565c0",
            )
        if max_r_mesh is not None:
            _add_surface_mesh(
                fig,
                max_r_mesh,
                name="Maxillary sinus (R)",
                color="#4fc3f7",
                opacity=sinus_opacity,
                show_wireframe=False,
                edge_color="#0277bd",
            )

    # --- Dense pathlines colored by local flow speed (Turbo) ---
    path_arrays: list[np.ndarray] = []
    path_speeds: list[np.ndarray] = []
    if show_streamlines and streamlines:
        # subsample if too many for interactive Plotly
        n_avail = len(streamlines)
        n_show = min(int(max_pathlines), n_avail)
        if n_show < n_avail:
            idx = np.linspace(0, n_avail - 1, n_show, dtype=int)
        else:
            idx = np.arange(n_avail)
        speed_lists = streamline_speeds or []
        cmax_u = max(float(max_vector_speed), 1e-6)
        for k, li in enumerate(idx):
            full = np.asarray(streamlines[li], dtype=float)
            if full.ndim != 2 or full.shape[0] < 2 or full.shape[1] < 3:
                continue
            if li < len(speed_lists) and speed_lists[li]:
                sp_full = np.asarray(speed_lists[li], dtype=float)
                if len(sp_full) != len(full):
                    sp_full = _sample_speed_along_line(full, speed, spacing, origin)
            else:
                sp_full = _sample_speed_along_line(full, speed, spacing, origin)
            # light decimation for long traces (keeps plot responsive)
            if len(full) > 280:
                step = max(1, len(full) // 220)
                arr = full[::step]
                sp = sp_full[::step]
            else:
                arr = full
                sp = sp_full
            if len(sp) != len(arr):
                n = min(len(sp), len(arr))
                arr, sp = arr[:n], sp[:n]
            path_arrays.append(arr)
            path_speeds.append(sp)
            fig.add_trace(
                go.Scatter3d(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    z=arr[:, 2],
                    mode="lines",
                    line=dict(
                        width=stream_width,
                        color=sp,
                        colorscale="Turbo",
                        cmin=0.0,
                        cmax=cmax_u,
                        colorbar=dict(title="|u| m/s", x=1.02) if k == 0 else None,
                    ),
                    name="Pathlines (velocity)" if k == 0 else f"pathline {k}",
                    showlegend=(k == 0),
                    hoverinfo="skip",
                    opacity=float(np.clip(streamline_opacity, 0.12, 1.0)),
                )
            )

    # --- Max-restriction cloud (narrow lumen / high 1/r) ---
    if show_restriction and restriction_pts is not None and len(restriction_pts) > 0:
        rp = np.asarray(restriction_pts, dtype=float)
        # columns: x,y,z, score(1/r), r_mm
        fig.add_trace(
            go.Scatter3d(
                x=rp[:, 0],
                y=rp[:, 1],
                z=rp[:, 2],
                mode="markers",
                marker=dict(
                    size=3.5,
                    color=rp[:, 3] if rp.shape[1] > 3 else "orangered",
                    colorscale=[
                        [0.0, "rgba(255,200,80,0.15)"],
                        [0.45, "rgba(255,120,40,0.55)"],
                        [1.0, "rgba(180,0,40,0.95)"],
                    ],
                    opacity=0.55,
                    line=dict(width=0),
                    colorbar=dict(title="1/r (1/mm)", x=1.12) if not show_streamlines else None,
                    showscale=not show_streamlines,
                ),
                name="Max restriction (narrow lumen)",
                hovertemplate=(
                    "restriction 1/r=%{marker.color:.2f}<br>"
                    + ("r≈%{customdata:.2f} mm" if rp.shape[1] > 4 else "")
                    + "<extra></extra>"
                ),
                customdata=rp[:, 4] if rp.shape[1] > 4 else None,
            )
        )

    # --- Magenta / pink: surgical areas to remove (path bottlenecks) ---
    if show_removal and removal_pts is not None and len(removal_pts) > 0:
        rp = np.asarray(removal_pts, dtype=float)
        fig.add_trace(
            go.Scatter3d(
                x=rp[:, 0],
                y=rp[:, 1],
                z=rp[:, 2],
                mode="markers",
                marker=dict(
                    size=4.5,
                    color="#ff2d95",  # magenta/pink
                    opacity=0.75,
                    line=dict(width=0),
                    symbol="circle",
                ),
                name="Areas to remove (narrow / ostium)",
                hovertemplate=(
                    "removal target"
                    + (" · r≈%{customdata:.2f} mm" if rp.shape[1] > 3 else "")
                    + "<extra></extra>"
                ),
                customdata=rp[:, 3] if rp.shape[1] > 3 else None,
            )
        )

    # --- Purple: dual instrument paths L/R naris → ipsilateral frontal ---
    if show_frontal_path:
        dual_paths = []
        if frontal_path_left_mm is not None and len(frontal_path_left_mm) >= 2:
            dual_paths.append(("L naris → L frontal", frontal_path_left_mm, "#ab47bc"))
        if frontal_path_right_mm is not None and len(frontal_path_right_mm) >= 2:
            dual_paths.append(("R naris → R frontal", frontal_path_right_mm, "#8e24aa"))
        # Fallback single path
        if not dual_paths and frontal_path_mm is not None and len(frontal_path_mm) >= 2:
            dual_paths.append(("Naris → frontal", frontal_path_mm, "#9c27b0"))
        for pi, (name, path, color) in enumerate(dual_paths):
            fp = np.asarray(path, dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=fp[:, 0],
                    y=fp[:, 1],
                    z=fp[:, 2],
                    mode="lines",
                    line=dict(color=color, width=9),
                    name=name,
                    hoverinfo="skip",
                    opacity=0.95,
                )
            )
            side_tag = "L" if "L naris" in name else ("R" if "R naris" in name else "")
            fig.add_trace(
                go.Scatter3d(
                    x=[fp[0, 0], fp[-1, 0]],
                    y=[fp[0, 1], fp[-1, 1]],
                    z=[fp[0, 2], fp[-1, 2]],
                    mode="markers+text",
                    marker=dict(
                        size=[8, 9],
                        color=["#ce93d8", "#6a1b9a"],
                        symbol=["circle", "diamond"],
                        line=dict(width=1, color="white"),
                    ),
                    text=[
                        f"{side_tag} naris" if side_tag else "Naris",
                        f"{side_tag} frontal" if side_tag else "Frontal",
                    ],
                    textposition="top center",
                    textfont=dict(size=10, color="#6a1b9a"),
                    name=f"{name} ends",
                    showlegend=False,
                )
            )

    # --- Velocity cones (optional, denser cones) ---
    if show_vectors:
        zz, yy, xx = np.where(airway)
        step = max(int(vector_stride), 1)
        # 3D striding for more uniform density
        sel = (
            (zz % step == 0)
            & (yy % step == 0)
            & (xx % step == 0)
        )
        zz, yy, xx = zz[sel], yy[sel], xx[sel]
        # Prefer higher-speed samples when over max
        if len(zz) > max_vectors:
            sp_all = speed[zz, yy, xx]
            # mix top-speed and random coverage
            n_top = max_vectors // 2
            n_rest = max_vectors - n_top
            top_idx = np.argpartition(sp_all, -n_top)[-n_top:]
            rest_pool = np.setdiff1d(np.arange(len(zz)), top_idx, assume_unique=False)
            rng = np.random.default_rng(0)
            if len(rest_pool) > n_rest:
                rest_idx = rng.choice(rest_pool, size=n_rest, replace=False)
            else:
                rest_idx = rest_pool
            keep = np.concatenate([top_idx, rest_idx])
            zz, yy, xx = zz[keep], yy[keep], xx[keep]

        sx, sy, sz = spacing
        ox, oy, oz = origin
        px = ox + xx * sx
        py = oy + yy * sy
        pz = oz + zz * sz
        vx = ux[zz, yy, xx]
        vy = uy[zz, yy, xx]
        vz = uz[zz, yy, xx]
        scale = 2.2 / max(max_vector_speed, 1e-9)
        fig.add_trace(
            go.Cone(
                x=px,
                y=py,
                z=pz,
                u=vx,
                v=vy,
                w=vz,
                colorscale="Turbo",
                cmin=0,
                cmax=max_vector_speed,
                sizemode="absolute",
                sizeref=max(scale * 1.1, 0.15),
                anchor="tail",
                name=f"Velocity ({len(px)} cones)",
                colorbar=dict(title="|u| m/s"),
                showscale=True,
                opacity=0.85,
            )
        )

    # Optional dual centerlines (off by default — low visual value vs pathlines)
    if show_centerlines:
        dual_drawn = False
        for name, cl_pts, width in (
            ("Centerline left naris → trachea", centerline_left_mm, 5),
            ("Centerline right naris → trachea", centerline_right_mm, 5),
        ):
            if cl_pts is not None and len(cl_pts) >= 2:
                cl = np.asarray(cl_pts, dtype=float)
                fig.add_trace(
                    go.Scatter3d(
                        x=cl[:, 0],
                        y=cl[:, 1],
                        z=cl[:, 2],
                        mode="lines",
                        line=dict(color="rgba(255,0,170,0.45)", width=width),
                        name=name,
                        hoverinfo="skip",
                    )
                )
                dual_drawn = True
        if not dual_drawn and centerline_mm is not None and len(centerline_mm) >= 2:
            cl = np.asarray(centerline_mm, dtype=float)
            fig.add_trace(
                go.Scatter3d(
                    x=cl[:, 0],
                    y=cl[:, 1],
                    z=cl[:, 2],
                    mode="lines",
                    line=dict(color="rgba(255,0,170,0.5)", width=6),
                    name="Centerline (nares → trachea)",
                    hoverinfo="skip",
                )
            )

    # Port markers — tip-accurate nares + trachea only (no per-path clutter)
    for port in ports:
        c = port.get("center_mm", [0, 0, 0])
        is_inlet = port.get("role") == "inlet"
        name = str(port.get("name", "port"))
        if is_inlet:
            color = "#00c853"
            short = "L naris" if "left" in name.lower() else (
                "R naris" if "right" in name.lower() else "Naris"
            )
            label = short
            size = 10
            symbol = "circle"
        else:
            color = "#ff1744"
            label = "Trachea"
            size = 11
            symbol = "diamond"
        fig.add_trace(
            go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers+text",
                marker=dict(
                    size=size,
                    color=color,
                    symbol=symbol,
                    line=dict(width=1.5, color="white"),
                    opacity=0.95,
                ),
                text=[label],
                textposition="top center",
                textfont=dict(size=11, color=color, family="Arial"),
                name=name,
            )
        )

    # Animated particles riding pathlines (optional)
    frames: list = []
    if animate_pathlines and path_arrays:
        particle_trace_idx = len(fig.data)
        xs0 = [float(a[0, 0]) for a in path_arrays]
        ys0 = [float(a[0, 1]) for a in path_arrays]
        zs0 = [float(a[0, 2]) for a in path_arrays]
        cs0 = [float(s[0]) for s in path_speeds]
        cmax_u = max(float(max_vector_speed), 1e-6)
        fig.add_trace(
            go.Scatter3d(
                x=xs0,
                y=ys0,
                z=zs0,
                mode="markers",
                marker=dict(
                    size=5,
                    color=cs0,
                    colorscale="Turbo",
                    cmin=0.0,
                    cmax=cmax_u,
                    line=dict(width=0),
                ),
                name="Flow particles",
                showlegend=True,
            )
        )
        n_f = max(8, int(n_anim_frames))
        for fi in range(n_f):
            frac = fi / max(n_f - 1, 1)
            xs, ys, zs, cs = [], [], [], []
            for arr, sp in zip(path_arrays, path_speeds):
                j = int(frac * (len(arr) - 1))
                xs.append(float(arr[j, 0]))
                ys.append(float(arr[j, 1]))
                zs.append(float(arr[j, 2]))
                cs.append(float(sp[min(j, len(sp) - 1)]))
            frames.append(
                go.Frame(
                    data=[
                        go.Scatter3d(
                            x=xs,
                            y=ys,
                            z=zs,
                            mode="markers",
                            marker=dict(
                                size=5,
                                color=cs,
                                colorscale="Turbo",
                                cmin=0.0,
                                cmax=cmax_u,
                                line=dict(width=0),
                            ),
                        )
                    ],
                    name=str(fi),
                    traces=[particle_trace_idx],
                )
            )
        fig.frames = frames

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="X mm",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            aspectmode="data",
            bgcolor=scene_bg,
            xaxis=dict(backgroundcolor=scene_bg, gridcolor="#444" if dark else "#ccc", showbackground=True),
            yaxis=dict(backgroundcolor=scene_bg, gridcolor="#444" if dark else "#ccc", showbackground=True),
            zaxis=dict(backgroundcolor=scene_bg, gridcolor="#444" if dark else "#ccc", showbackground=True),
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
        ),
        paper_bgcolor=paper_bg,
        font_color=font_c,
        legend=dict(bgcolor="rgba(0,0,0,0.3)" if dark else "rgba(255,255,255,0.7)"),
    )
    if frames:
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=0,
                    x=0.05,
                    xanchor="left",
                    yanchor="bottom",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=80, redraw=True),
                                    fromcurrent=True,
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    frame=dict(duration=0, redraw=False),
                                    mode="immediate",
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ]
        )
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Sinus_CFD Viewer",
        page_icon="🫁",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Sinus_CFD — Airflow Viewer")
    st.caption(
        "Tri-planar velocity · 3D cavity · streamlines  ·  "
        "OpenFOAM simpleFoam when imported (else potential-flow preview)"
    )

    # Always-visible version banner (confirm you are not on a stale app/session)
    st.info(
        f"**App version `{APP_VERSION}`** — {APP_VERSION_LABEL}\n\n"
        "If this string does not match, stop the Streamlit process and relaunch "
        "`py -3.12 -m streamlit run app/viewer.py`."
    )

    cases = list_cases()
    with st.sidebar:
        st.header("Version")
        st.markdown(f"**`{APP_VERSION}`**")
        st.caption(APP_VERSION_LABEL)
        if st.button("Clear cache & reload data", type="primary"):
            st.cache_data.clear()
            st.rerun()

        st.header("Case")
        if not cases:
            st.error(
                "No flow fields found under `outputs/`. "
                "Run process_whole_head / process_case then compute_flow."
            )
            st.code(
                "py -3.12 scripts/process_whole_head.py --case VisibleHuman_Head\n"
                "py -3.12 scripts/compute_flow.py --case VisibleHuman_Head",
                language="powershell",
            )
            return

        # Prefer whole-head case when present
        default_idx = 0
        if "VisibleHuman_Head" in cases:
            default_idx = cases.index("VisibleHuman_Head")
        case_id = st.selectbox("Case ID", cases, index=default_idx)
        st.header("3D display")
        preset = st.radio(
            "Preset",
            [
                "Inhale: nares → trachea",
                "Head + airway",
                "Cavity first",
                "Flow overlay",
                "Cavity only",
            ],
            index=0,
            help="Inhale: streamlines seeded at nostrils flowing toward trachea.",
        )
        show_skin = st.checkbox("Show skin surface", value=True)
        skin_opacity = st.slider(
            "Skin opacity",
            0.05,
            0.95,
            0.38,
            0.01,
            help="Filled outer skin mesh from CT body mask (not a point cloud).",
        )
        show_head = st.checkbox("Show soft-tissue solid (if no skin)", value=False)
        head_opacity = st.slider("Soft-tissue opacity", 0.05, 0.85, 0.25, 0.01)
        show_bone = st.checkbox("Show bone", value=False)
        bone_opacity = st.slider("Bone opacity", 0.05, 1.0, 0.45, 0.01)
        show_mesh = st.checkbox("Show combined air space", value=False)
        mesh_opacity = st.slider("Air space opacity", 0.15, 1.0, 0.40, 0.01)
        show_left = st.checkbox("Left nasal cavity (CT)", value=True)
        show_right = st.checkbox("Right nasal cavity (CT)", value=True)
        show_septum = st.checkbox(
            "Nasal septum (off by default)",
            value=False,
            help="CT tissue between L/R cavities; usually leave off for air/path view.",
        )
        show_mucosa = st.checkbox("Mucosa / passage walls", value=False)
        left_opacity = st.slider("Left cavity opacity", 0.05, 1.0, 0.40, 0.01)
        right_opacity = st.slider("Right cavity opacity", 0.05, 1.0, 0.40, 0.01)
        septum_opacity = st.slider("Septum opacity", 0.1, 1.0, 0.55, 0.01)
        show_wireframe = st.checkbox(
            "Skin wireframe edges",
            value=True,
            help="On by default so head extent reads clearly over translucent skin.",
        )
        show_streamlines = st.checkbox(
            "Flow pathlines (velocity-colored)",
            value=(preset not in ("Cavity only",)),
        )
        streamline_width = st.slider("Pathline width", 1.0, 8.0, 2.5, 0.5)
        streamline_opacity = st.slider("Pathline opacity", 0.15, 1.0, 0.55, 0.05)
        max_pathlines = st.slider(
            "Max pathlines shown",
            20,
            500,
            260,
            10,
            help="Dense curvy semi-transparent lines; lower if the plot is slow.",
        )
        animate_pathlines = st.checkbox(
            "Animate particles along pathlines",
            value=False,
            help="Play button on the 3D plot rides particles naris→trachea.",
        )
        show_restriction = st.checkbox(
            "Highlight max restriction (narrow lumen)",
            value=True,
            help="Hot cloud where distance-to-wall is smallest (high 1/r).",
        )
        show_centerlines = st.checkbox(
            "Show anatomical centerlines (dim)",
            value=False,
            help="Usually off — dense pathlines carry more information.",
        )
        st.subheader("Surgical guidance")
        show_frontal_path = st.checkbox(
            "Purple paths: L/R naris → L/R frontal (straighter)",
            value=True,
            help="Dual ipsilateral instrument corridors (straightish, open dark air).",
        )
        show_removal = st.checkbox(
            "Magenta / pink: areas to remove",
            value=True,
            help="Narrow bottlenecks / ostium-like zones along the frontal access path.",
        )
        show_frontal_sinus = st.checkbox("Frontal sinus", value=True)
        show_sphenoid = st.checkbox("Sphenoid sinus", value=False)
        show_maxillary = st.checkbox("Maxillary sinuses (L/R)", value=False)
        sinus_opacity = st.slider("Sinus opacity", 0.05, 0.9, 0.32, 0.01)
        show_vectors = st.checkbox(
            "Velocity cones (glyphs)",
            value=False,
        )
        vector_stride = st.slider(
            "Vector density (1=max, higher=sparser)",
            1,
            8,
            3,
            help="Lower = many more cones. Default 3 is moderate.",
        )
        max_vectors = st.slider("Max velocity cones", 500, 12000, 3000, 500)
        bg_mode = st.selectbox("Background", ["dark", "light"], index=1)

        if preset == "Inhale: nares → trachea":
            show_streamlines = True
            show_vectors = False
            show_restriction = True
            show_centerlines = False
            show_skin = True
            show_wireframe = True
            # Keep skin readable as a surface shell, not vanishing
            skin_opacity = min(max(skin_opacity, 0.28), 0.42)
            show_mesh = False
            show_left = True
            show_right = True
            show_septum = False
            streamline_width = max(streamline_width, 2.5)
            streamline_opacity = min(streamline_opacity, 0.6)
            max_pathlines = max(max_pathlines, 160)
            vector_stride = max(vector_stride, 3)
            max_vectors = min(max_vectors, 2500)
        elif preset == "Cavity only":
            show_streamlines = False
            show_vectors = True
            show_restriction = True
            show_head = False
            show_skin = False
            mesh_opacity = max(mesh_opacity, 0.7)
        elif preset == "Head + airway":
            show_skin = True
            skin_opacity = max(skin_opacity, 0.5)
            mesh_opacity = min(mesh_opacity, 0.45)
            streamline_width = min(streamline_width, 2.5)
            show_vectors = False
            show_streamlines = True
            vector_stride = min(vector_stride, 2)
        elif preset == "Flow overlay":
            show_skin = True
            skin_opacity = min(skin_opacity, 0.28)
            show_vectors = False
            show_streamlines = True
            show_restriction = True
            streamline_opacity = min(streamline_opacity, 0.5)
            max_pathlines = max(max_pathlines, 200)
        elif preset == "Cavity first":
            show_head = False
            show_skin = False
            mesh_opacity = max(mesh_opacity, 0.55)
            streamline_width = min(streamline_width, 3.0)

        st.header("Roadmap (future)")
        st.markdown(
            """
- Whole-head CT (Visible Human)  
- Shortest path: naris → ostium → sinus  
- Mucus clearance / widened ostium  
- CRS · NAO · NVC diagnosis  
- Polyp detection · patient CT upload  
            """
        )

    fp = case_data_fingerprint(case_id)
    data = load_case(case_id, fp)
    if "error" in data:
        st.error(data["error"])
        return

    speed = data["speed"]
    airway = data["airway"]
    nz, ny, nx = speed.shape
    meta = data.get("meta", {})
    bc = data.get("bc", {})
    stats = data.get("stats", {})

    # Data version panel — proves which on-disk outputs are loaded
    with st.expander("Loaded data version (verify nares / skin)", expanded=True):
        st.markdown(
            f"- **App:** `{APP_VERSION}`\n"
            f"- **Case:** `{case_id}`\n"
            f"- **BC file time:** `{data.get('bc_mtime')}`\n"
            f"- **Stats file time:** `{data.get('stats_mtime')}`\n"
            f"- **Fingerprint:** `{fp[-80:]}`"
        )
        # Expected for current pipeline
        expected_methods = []
        for p in bc.get("ports", []):
            if p.get("role") == "inlet":
                expected_methods.append(
                    f"**{p.get('name')}** · method=`{p.get('method')}` · "
                    f"xyz_mm=`{[round(x,1) for x in p.get('center_mm', [])]}`"
                )
        if expected_methods:
            st.markdown("**Inlet ports currently loaded:**")
            for line in expected_methods:
                st.markdown(f"- {line}")
        else:
            st.warning("No inlet ports in loaded BC JSON.")

        # Guardrail: tip-accurate CT naris openings
        methods = [p.get("method") for p in bc.get("ports", []) if p.get("role") == "inlet"]
        tip_ok = {
            "skin_tip_vestibule",
            "ct_naris_opening_air",
            "ct_naris_opening_tip",
            "ct_naris_opening",
            "edge_nose_tip_skin_naris",
        }
        if methods and all(m in tip_ok for m in methods):
            if all(m == "skin_tip_vestibule" for m in methods):
                st.success(
                    "Nares at **skin nose tip** with **open vestibules** painted into "
                    "each cavity (CT often seals this region). "
                    "Pathlines ~50% L / ~50% R → trachea."
                )
            elif all(m == "ct_naris_opening_air" for m in methods):
                st.info(
                    "Nares on CT opening∩air. For tip openings run: "
                    "`py -3.12 scripts/extend_nasal_to_tip.py`"
                )
            else:
                st.info(f"Nares methods: `{methods}`.")
        elif methods:
            st.warning(
                f"Unexpected naris methods `{methods}`. "
                "Click **Clear cache & reload data**, or re-run pathline regen."
            )

        notes = stats.get("notes") or []
        if notes:
            st.markdown("**Pipeline notes (from stats):**")
            for n in notes[:8]:
                st.caption(f"• {n}")

        qc1, qc2 = st.columns(2)
        if data.get("face_qc_path"):
            with qc1:
                st.markdown("**Face QC (nares should be at nose tip, not orbits)**")
                st.image(data["face_qc_path"], use_container_width=True)
        if data.get("preview_path"):
            with qc2:
                st.markdown("**Tri-planar QC**")
                st.image(data["preview_path"], use_container_width=True)

        passage = data.get("passage") or {}
        pm = passage.get("metrics") or {}
        if pm:
            st.markdown("**Nasal passage domain (walls + open ports)**")
            st.markdown(
                f"- Lumen volume: **{pm.get('lumen_volume_ml', 0):.1f} mL**\n"
                f"- Centerline length: **{pm.get('centerline_length_mm', 0):.1f} mm**\n"
                f"- Cross-section min/mean/max: "
                f"**{pm.get('min_cross_section_mm2', 0):.1f} / "
                f"{pm.get('mean_cross_section_mm2', 0):.1f} / "
                f"{pm.get('max_cross_section_mm2', 0):.1f} mm²**\n"
                f"- Wall voxels: `{pm.get('wall_voxels')}` · "
                f"inlet open: `{pm.get('inlet_open_voxels')}` · "
                f"outlet open: `{pm.get('outlet_open_voxels')}`"
            )
        n_sl = len(data.get("streamlines") or [])
        st.caption(
            f"**{n_sl} pathlines** (Turbo = |u|) · "
            "**Purple** = least-resistance naris→frontal · "
            "**Magenta** = areas to remove (narrow bottlenecks)."
        )
        surg = data.get("surgical") or {}
        if surg.get("path_metrics"):
            for pm_ in surg["path_metrics"][:2]:
                st.caption(
                    f"• `{pm_.get('name')}`: **{pm_.get('length_mm', 0):.1f} mm** · "
                    f"min r **{pm_.get('min_radius_mm', 0):.2f} mm**"
                )
        labels = (surg.get("sinus_anatomy") or {}).get("labels") or []
        if labels:
            st.markdown("**Sinus labels (CT air heuristics):**")
            for L in labels:
                st.caption(
                    f"• **{L.get('name')}**: {L.get('voxels')} vx · "
                    f"center mm `{[round(x,1) for x in (L.get('center_mm') or [])]}`"
                )
        if not surg:
            st.info(
                "No surgical guidance yet. Run: "
                "`py -3.12 scripts/compute_surgical_guidance.py --case VisibleHuman_Head`"
            )

    method = str(meta.get("method", "potential_flow"))
    is_openfoam = "openfoam" in method.lower()
    if is_openfoam:
        st.success(
            f"**Flow source: OpenFOAM simpleFoam** · time=`{meta.get('openfoam_time', '?')}` · "
            f"cells=`{meta.get('n_cells', '?')}` · mapped voxels=`{meta.get('n_mapped_voxels', '?')}`"
        )
        st.markdown(
            "**Inspiration pathlines** (Turbo = |u|): **nostrils → trachea**. "
            "**Purple** corridor = least-resistance **naris → frontal sinus** (instrument path). "
            "**Magenta/pink** = **areas to remove** (narrow bottlenecks along frontal access)."
        )
    else:
        st.warning(
            f"**Flow source: `{method}`** (not OpenFOAM). "
            "After a Docker run, import with: "
            "`py -3.12 scripts/import_openfoam_results.py --case VisibleHuman_Head`"
        )

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max |u|", f"{meta.get('max_speed_m_s', float(speed[airway].max() if airway.any() else 0)):.3f} m/s")
    c2.metric("Mean |u|", f"{meta.get('mean_speed_m_s', float(speed[airway].mean() if airway.any() else 0)):.3f} m/s")
    c3.metric("Target Q", f"{meta.get('target_flow_L_per_min', 18):.1f} L/min")
    c4.metric("App ver", APP_VERSION.split("-")[0])

    if meta.get("notes"):
        with st.expander("Method notes / caveats", expanded=is_openfoam):
            st.markdown(f"**Method:** `{method}`")
            for n in meta["notes"]:
                st.write(f"- {n}")
            if bc.get("outlet_is_proxy"):
                st.warning("Outlet is a nasopharynx proxy — NasalSeg FOV has no true trachea.")

    # ---- Tri-planar ----
    st.subheader("Tri-planar velocity")
    z0, y0, x0 = nz // 2, ny // 2, nx // 2
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        iz = st.slider("Axial (z index)", 0, nz - 1, z0, key="iz")
    with sc2:
        iy = st.slider("Coronal (y index)", 0, ny - 1, y0, key="iy")
    with sc3:
        ix = st.slider("Sagittal (x index)", 0, nx - 1, x0, key="ix")

    vmax = float(np.percentile(speed[airway], 99)) if airway.any() else 1.0
    fig2d = _slice_fig(speed, airway, iz, iy, ix, data["spacing"], vmax)
    st.plotly_chart(fig2d, use_container_width=True)

    # ---- 3D ----
    st.subheader("3D head + airway + airflow")
    mesh = None
    head_mesh = None
    skin_mesh = None
    bone_mesh = None
    if data.get("stl_path"):
        try:
            mesh = _load_mesh_decimated(data["stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Could not load airway STL: {exc}")
    if data.get("skin_stl_path"):
        try:
            # Keep more faces so skin reads as a continuous surface + wireframe
            skin_mesh = _load_mesh_decimated(data["skin_stl_path"], target_faces=40000)
        except Exception as exc:
            st.warning(f"Could not load skin STL: {exc}")
    if data.get("head_stl_path"):
        try:
            head_mesh = _load_mesh_decimated(data["head_stl_path"], target_faces=18000)
        except Exception as exc:
            st.warning(f"Could not load head STL: {exc}")
    if show_skin and skin_mesh is None and show_head is False:
        st.info("No skin mesh for this case. Enable soft-tissue solid or run process_whole_head.")
    if data.get("bone_stl_path"):
        try:
            bone_mesh = _load_mesh_decimated(data["bone_stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Could not load bone STL: {exc}")

    left_mesh = right_mesh = septum_mesh = mucosa_mesh = None
    if data.get("left_cavity_stl_path"):
        try:
            left_mesh = _load_mesh_decimated(data["left_cavity_stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Left cavity STL: {exc}")
    if data.get("right_cavity_stl_path"):
        try:
            right_mesh = _load_mesh_decimated(data["right_cavity_stl_path"], target_faces=12000)
        except Exception as exc:
            st.warning(f"Right cavity STL: {exc}")
    if data.get("septum_stl_path"):
        try:
            septum_mesh = _load_mesh_decimated(data["septum_stl_path"], target_faces=10000)
        except Exception as exc:
            st.warning(f"Septum STL: {exc}")
    if data.get("mucosa_stl_path"):
        try:
            mucosa_mesh = _load_mesh_decimated(data["mucosa_stl_path"], target_faces=18000)
        except Exception as exc:
            st.warning(f"Mucosa STL: {exc}")

    frontal_mesh = sphenoid_mesh = max_l_mesh = max_r_mesh = None
    if data.get("frontal_stl"):
        try:
            frontal_mesh = _load_mesh_decimated(data["frontal_stl"], target_faces=10000)
        except Exception as exc:
            st.warning(f"Frontal sinus STL: {exc}")
    if data.get("sphenoid_stl"):
        try:
            sphenoid_mesh = _load_mesh_decimated(data["sphenoid_stl"], target_faces=10000)
        except Exception as exc:
            st.warning(f"Sphenoid STL: {exc}")
    if data.get("maxillary_left_stl"):
        try:
            max_l_mesh = _load_mesh_decimated(data["maxillary_left_stl"], target_faces=8000)
        except Exception as exc:
            st.warning(f"Maxillary L STL: {exc}")
    if data.get("maxillary_right_stl"):
        try:
            max_r_mesh = _load_mesh_decimated(data["maxillary_right_stl"], target_faces=8000)
        except Exception as exc:
            st.warning(f"Maxillary R STL: {exc}")

    ct_nasal = data.get("ct_nasal") or {}
    if ct_nasal:
        st.info(
            f"**CT-native nasal model** (`{ct_nasal.get('method', '?')}`): "
            f"L={ct_nasal.get('left_voxels', '?')} · R={ct_nasal.get('right_voxels', '?')} · "
            f"septum={ct_nasal.get('septum_voxels', '?')} voxels · "
            f"naris openings={ct_nasal.get('naris_opening_voxels', '?')}"
        )
    elif show_septum or show_left or show_right:
        st.warning(
            "No CT L/R/septum meshes yet. Run: "
            "`py -3.12 scripts/refine_nasal_ct.py --case VisibleHuman_Head`"
        )

    streamlines = data["streamlines"] if show_streamlines else []
    streamline_speeds = data.get("streamline_speeds") or []
    ports = bc.get("ports", [])
    # Prefer tip-accurate centers from nares.json when present
    tip_by_name: dict[str, list[float]] = {}
    for npnt in (data.get("nares") or {}).get("naris_points") or []:
        nm = str(npnt.get("name") or "")
        if npnt.get("center_mm"):
            tip_by_name[nm] = [float(v) for v in npnt["center_mm"]]
    ports_lite = []
    for p in ports:
        center = p.get("center_mm")
        name = str(p.get("name") or "")
        if name in tip_by_name:
            center = tip_by_name[name]
        ports_lite.append(
            {
                "name": name,
                "role": p.get("role"),
                "center_mm": center,
                "method": p.get("method"),
            }
        )

    # Prefer dual open-paths when passage dual missing
    op = data.get("open_paths") or {}
    passage = data.get("passage") or {}
    cl_left = passage.get("centerline_left_mm") or op.get("centerline_left_mm")
    cl_right = passage.get("centerline_right_mm") or op.get("centerline_right_mm")
    cl_mid = passage.get("centerline_mm") or op.get("centerline_mid_mm")

    fig3d = _fig_3d(
        mesh=mesh,
        head_mesh=head_mesh,
        skin_mesh=skin_mesh,
        bone_mesh=bone_mesh,
        streamlines=streamlines if show_streamlines else [],
        streamline_speeds=streamline_speeds if show_streamlines else [],
        ux=data["ux"],
        uy=data["uy"],
        uz=data["uz"],
        speed=speed,
        airway=airway,
        spacing=data["spacing"],
        origin=data["origin"],
        show_head=show_head and head_mesh is not None,
        head_opacity=head_opacity,
        show_skin=show_skin and skin_mesh is not None,
        skin_opacity=skin_opacity,
        show_bone=show_bone and bone_mesh is not None,
        bone_opacity=bone_opacity,
        show_mesh=show_mesh,
        mesh_opacity=mesh_opacity,
        show_wireframe=show_wireframe,
        show_streamlines=show_streamlines,
        streamline_width=streamline_width,
        streamline_opacity=streamline_opacity,
        max_pathlines=max_pathlines,
        show_vectors=show_vectors,
        vector_stride=vector_stride,
        max_vector_speed=vmax,
        ports=ports_lite,
        bg_mode=bg_mode,
        max_vectors=max_vectors,
        centerline_mm=cl_mid,
        centerline_left_mm=cl_left,
        centerline_right_mm=cl_right,
        show_centerlines=show_centerlines,
        left_mesh=left_mesh,
        right_mesh=right_mesh,
        septum_mesh=septum_mesh,
        mucosa_mesh=mucosa_mesh,
        show_left=show_left and left_mesh is not None,
        show_right=show_right and right_mesh is not None,
        show_septum=show_septum and septum_mesh is not None,
        show_mucosa=show_mucosa and mucosa_mesh is not None,
        left_opacity=left_opacity,
        right_opacity=right_opacity,
        septum_opacity=septum_opacity,
        restriction_pts=data.get("restriction_pts"),
        show_restriction=show_restriction,
        animate_pathlines=animate_pathlines and show_streamlines,
        n_anim_frames=28,
        frontal_path_mm=(
            ((data.get("surgical") or {}).get("paths_mm") or {}).get(
                "naris_left_to_frontal"
            )
        ),
        frontal_path_left_mm=(
            ((data.get("surgical") or {}).get("paths_mm") or {}).get(
                "naris_left_to_frontal"
            )
        ),
        frontal_path_right_mm=(
            ((data.get("surgical") or {}).get("paths_mm") or {}).get(
                "naris_right_to_frontal"
            )
        ),
        show_frontal_path=show_frontal_path,
        removal_pts=data.get("removal_pts"),
        show_removal=show_removal,
        frontal_mesh=frontal_mesh,
        sphenoid_mesh=sphenoid_mesh,
        max_l_mesh=max_l_mesh,
        max_r_mesh=max_r_mesh,
        show_frontal_sinus=show_frontal_sinus and frontal_mesh is not None,
        show_sphenoid=show_sphenoid and sphenoid_mesh is not None,
        show_maxillary=show_maxillary,
        sinus_opacity=sinus_opacity,
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption(
        "Curvy pathlines (Turbo = |u|): **nostrils → trachea** (inhale). "
        "**Purple** = dual **L/R naris → ipsilateral frontal** instrument corridors "
        "(straighter, open dark air). "
        "**Magenta/pink** = **areas to remove**. Toggle layers in the sidebar."
    )

    # ---- BC summary ----
    with st.expander("Boundary conditions & breathing"):
        breath = bc.get("breathing", {})
        flow = bc.get("flow_assignment", {})
        st.write(
            f"**Inlets:** {', '.join(bc.get('inlet_names', []))}  ·  "
            f"**Outlet:** {bc.get('outlet_name', '?')}  ·  "
            f"**Mouth:** closed"
        )
        if breath:
            st.write(
                f"VT={breath.get('tidal_volume_L')} L · "
                f"RR={breath.get('respiratory_rate_per_min')}/min · "
                f"Ti={breath.get('Ti_s_effective', 0):.2f} s · "
                f"Q={flow.get('total_inflow_L_per_min', 0):.1f} L/min"
            )
        st.json(
            {
                "inlets": bc.get("inlet_names"),
                "outlet": bc.get("outlet_name"),
                "flow_split_L_per_min": breath.get("flow_split_L_per_min"),
            }
        )

    st.markdown("---")
    st.markdown(
        "**Future product modules** (not yet computed): ostium pathways · mucus "
        "clearance · CRS / NAO / NVC diagnosis · polyp ID · secure CT upload."
    )


if __name__ == "__main__":
    main()
