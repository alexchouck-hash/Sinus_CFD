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
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_CASE = "P001"
OUTPUTS = REPO_ROOT / "outputs"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading flow field…")
def load_case(case_id: str) -> dict:
    case_dir = OUTPUTS / case_id
    npz_path = case_dir / f"{case_id}_flow.npz"
    if not npz_path.is_file():
        return {"error": f"Missing {npz_path}. Run: py -3.12 scripts/compute_flow.py --case {case_id}"}

    data = np.load(npz_path)
    out = {
        "case_id": case_id,
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
            out["streamlines"] = json.load(f)["lines"]
    else:
        out["streamlines"] = []

    meta_path = case_dir / f"{case_id}_flow_meta.json"
    if meta_path.is_file():
        with meta_path.open(encoding="utf-8") as f:
            out["meta"] = json.load(f)
    else:
        out["meta"] = {}

    stl_path = case_dir / f"{case_id}_airway.stl"
    out["stl_path"] = str(stl_path) if stl_path.is_file() else None
    head_stl = case_dir / f"{case_id}_head.stl"
    out["head_stl_path"] = str(head_stl) if head_stl.is_file() else None
    skin_stl = case_dir / f"{case_id}_skin.stl"
    out["skin_stl_path"] = str(skin_stl) if skin_stl.is_file() else None
    bone_stl = case_dir / f"{case_id}_bone.stl"
    out["bone_stl_path"] = str(bone_stl) if bone_stl.is_file() else None

    bc_path = case_dir / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        with bc_path.open(encoding="utf-8") as f:
            out["bc"] = json.load(f)
    else:
        out["bc"] = {}

    stats_path = case_dir / f"{case_id}_stats.json"
    if stats_path.is_file():
        with stats_path.open(encoding="utf-8") as f:
            out["stats"] = json.load(f)
    else:
        out["stats"] = {}

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
) -> None:
    v = mesh.vertices
    f = mesh.faces
    fig.add_trace(
        go.Mesh3d(
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
            lighting=dict(
                ambient=0.55,
                diffuse=0.85,
                specular=0.35,
                roughness=0.45,
                fresnel=0.15,
            ),
            lightposition=dict(x=80, y=120, z=200),
            hoverinfo="skip",
            showlegend=True,
        )
    )
    if show_wireframe:
        wf = _mesh_wireframe_trace(mesh, max_edges=wireframe_max_edges, color=edge_color)
        if wf is not None:
            wf.name = f"{name} edges"
            fig.add_trace(wf)


def _fig_3d(
    mesh,
    head_mesh,
    skin_mesh,
    bone_mesh,
    streamlines: list,
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
    show_vectors: bool,
    vector_stride: int,
    max_vector_speed: float,
    ports: list[dict],
    bg_mode: str = "dark",
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
            show_wireframe=False,
            edge_color=head_edge,
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

    # --- Airway cavity (air space) ---
    if show_mesh and mesh is not None:
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

    # --- Streamlines (thinner by default so mesh stays primary) ---
    if show_streamlines:
        for li, line in enumerate(streamlines):
            arr = np.asarray(line, dtype=float)
            if arr.shape[0] < 2:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    z=arr[:, 2],
                    mode="lines",
                    line=dict(
                        width=stream_width,
                        color=np.linspace(0, 1, len(arr)),
                        colorscale="Turbo",
                    ),
                    name="Streamlines" if li == 0 else None,
                    showlegend=(li == 0),
                    hoverinfo="skip",
                    opacity=0.9,
                )
            )

    # --- Optional velocity cones (off by default) ---
    if show_vectors:
        zz, yy, xx = np.where(airway)
        step = max(int(vector_stride), 1)
        zz, yy, xx = zz[::step], yy[::step], xx[::step]
        if len(zz) > 1200:
            idx = np.linspace(0, len(zz) - 1, 1200, dtype=int)
            zz, yy, xx = zz[idx], yy[idx], xx[idx]

        sx, sy, sz = spacing
        ox, oy, oz = origin
        px = ox + xx * sx
        py = oy + yy * sy
        pz = oz + zz * sz
        vx = ux[zz, yy, xx]
        vy = uy[zz, yy, xx]
        vz = uz[zz, yy, xx]
        scale = 3.0 / max(max_vector_speed, 1e-9)
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
                sizeref=scale * 1.5,
                anchor="tail",
                name="Velocity",
                colorbar=dict(title="|u| m/s"),
                showscale=True,
                opacity=0.75,
            )
        )

    # Port markers
    for port in ports:
        c = port.get("center_mm", [0, 0, 0])
        color = "#3dff9a" if port.get("role") == "inlet" else "#ff5c7a"
        fig.add_trace(
            go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers+text",
                marker=dict(size=7, color=color, line=dict(width=1, color="white")),
                text=[port.get("name", "")],
                textposition="top center",
                name=port.get("name", "port"),
            )
        )

    fig.update_layout(
        height=700,
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
        "Potential-flow preview (not full Navier–Stokes CFD yet)"
    )

    cases = list_cases()
    with st.sidebar:
        st.header("Case")
        if not cases:
            st.error(
                "No flow fields found under `outputs/`. "
                "Run `process_case.py` then `compute_flow.py`."
            )
            st.code(
                "py -3.12 scripts/process_case.py --case P001\n"
                "py -3.12 scripts/compute_flow.py --case P001",
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
            ["Head + airway", "Cavity first", "Flow overlay", "Cavity only"],
            index=0,
            help="Head + airway: semi-transparent solid head with airway inside.",
        )
        show_skin = st.checkbox("Show skin surface", value=True)
        skin_opacity = st.slider("Skin opacity", 0.05, 0.9, 0.35, 0.01)
        show_head = st.checkbox("Show soft-tissue solid (if no skin)", value=False)
        head_opacity = st.slider("Soft-tissue opacity", 0.05, 0.85, 0.25, 0.01)
        show_bone = st.checkbox("Show bone", value=False)
        bone_opacity = st.slider("Bone opacity", 0.05, 1.0, 0.45, 0.01)
        show_mesh = st.checkbox("Show air space (airway)", value=True)
        mesh_opacity = st.slider("Air space opacity", 0.15, 1.0, 0.55, 0.01)
        show_wireframe = st.checkbox("Airway wireframe edges", value=False)
        show_streamlines = st.checkbox(
            "Streamlines (curved)",
            value=(preset not in ("Cavity only",)),
        )
        streamline_width = st.slider("Streamline width", 1.0, 8.0, 2.5, 0.5)
        show_vectors = st.checkbox(
            "Velocity cones",
            value=(preset == "Flow overlay"),
        )
        vector_stride = st.slider("Vector stride (higher = fewer)", 3, 16, 8)
        bg_mode = st.selectbox("Background", ["dark", "light"], index=0)

        if preset == "Cavity only":
            show_streamlines = False
            show_vectors = False
            show_head = False
            show_skin = False
            mesh_opacity = max(mesh_opacity, 0.7)
        elif preset == "Head + airway":
            show_skin = True
            skin_opacity = min(max(skin_opacity, 0.28), 0.45)
            mesh_opacity = max(mesh_opacity, 0.45)
            streamline_width = min(streamline_width, 3.0)
            show_vectors = False
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

    data = load_case(case_id)
    if "error" in data:
        st.error(data["error"])
        return

    speed = data["speed"]
    airway = data["airway"]
    nz, ny, nx = speed.shape
    meta = data.get("meta", {})
    bc = data.get("bc", {})

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max |u|", f"{meta.get('max_speed_m_s', float(speed[airway].max())):.3f} m/s")
    c2.metric("Mean |u|", f"{meta.get('mean_speed_m_s', float(speed[airway].mean())):.3f} m/s")
    c3.metric("Target Q", f"{meta.get('target_flow_L_per_min', 18):.1f} L/min")
    c4.metric("Method", meta.get("method", "potential_flow")[:22])

    if meta.get("notes"):
        with st.expander("Method notes / caveats", expanded=False):
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
            skin_mesh = _load_mesh_decimated(data["skin_stl_path"], target_faces=22000)
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

    streamlines = data["streamlines"] if show_streamlines else []
    ports = bc.get("ports", [])
    ports_lite = [
        {
            "name": p.get("name"),
            "role": p.get("role"),
            "center_mm": p.get("center_mm"),
        }
        for p in ports
    ]

    fig3d = _fig_3d(
        mesh=mesh,
        head_mesh=head_mesh,
        skin_mesh=skin_mesh,
        bone_mesh=bone_mesh,
        streamlines=streamlines if show_streamlines else [],
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
        show_vectors=show_vectors,
        vector_stride=vector_stride,
        max_vector_speed=vmax,
        ports=ports_lite,
        bg_mode=bg_mode,
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption(
        "Tip: preset **Head + airway** — solid head shell (semi-transparent) with "
        "airway inside. Raise head opacity to see the face/skull outline; lower it "
        "to see streamlines through the head."
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
