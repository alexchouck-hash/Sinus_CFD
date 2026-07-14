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

    bc_path = case_dir / f"{case_id}_boundary_conditions.json"
    if bc_path.is_file():
        with bc_path.open(encoding="utf-8") as f:
            out["bc"] = json.load(f)
    else:
        out["bc"] = {}

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


def _load_mesh_decimated(stl_path: str, target_faces: int = 8000):
    import trimesh

    mesh = trimesh.load(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(target_faces)
        except Exception:
            # Fallback: random face subset
            idx = np.linspace(0, len(mesh.faces) - 1, target_faces, dtype=int)
            mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces[idx], process=False)
    return mesh


def _fig_3d(
    mesh,
    streamlines: list,
    ux: np.ndarray,
    uy: np.ndarray,
    uz: np.ndarray,
    speed: np.ndarray,
    airway: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    show_vectors: bool,
    vector_stride: int,
    max_vector_speed: float,
    ports: list[dict],
) -> go.Figure:
    fig = go.Figure()

    # Semi-transparent cavity
    if mesh is not None:
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
                color="lightblue",
                opacity=0.18,
                name="Airway cavity",
                flatshading=True,
                hoverinfo="skip",
            )
        )

    # Curving streamlines colored by local speed (approx by segment index)
    for li, line in enumerate(streamlines):
        arr = np.asarray(line, dtype=float)
        if arr.shape[0] < 2:
            continue
        # Color by normalized arc length as stand-in for progression
        fig.add_trace(
            go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                mode="lines",
                line=dict(
                    width=4,
                    color=np.linspace(0, 1, len(arr)),
                    colorscale="Turbo",
                ),
                name="Streamline" if li == 0 else None,
                showlegend=(li == 0),
                hoverinfo="skip",
            )
        )

    # Optional quiver (subsampled airway)
    if show_vectors:
        zz, yy, xx = np.where(airway)
        # subsample
        step = max(int(vector_stride), 1)
        sel = slice(None, None, step)
        zz, yy, xx = zz[sel], yy[sel], xx[sel]
        # further cap
        if len(zz) > 2500:
            idx = np.linspace(0, len(zz) - 1, 2500, dtype=int)
            zz, yy, xx = zz[idx], yy[idx], xx[idx]

        sx, sy, sz = spacing
        ox, oy, oz = origin
        px = ox + xx * sx
        py = oy + yy * sy
        pz = oz + zz * sz
        vx = ux[zz, yy, xx]
        vy = uy[zz, yy, xx]
        vz = uz[zz, yy, xx]
        sp = speed[zz, yy, xx]
        # Scale arrow length in mm for visibility
        scale = 4.0 / max(max_vector_speed, 1e-9)
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
                sizeref=scale * 2.0,
                anchor="tail",
                name="Velocity",
                colorbar=dict(title="|u| m/s"),
                showscale=True,
            )
        )

    # Port markers
    for port in ports:
        c = port.get("center_mm", [0, 0, 0])
        color = "#00ff88" if port.get("role") == "inlet" else "#ff4466"
        fig.add_trace(
            go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers+text",
                marker=dict(size=6, color=color),
                text=[port.get("name", "")],
                textposition="top center",
                name=port.get("name", "port"),
            )
        )

    fig.update_layout(
        height=640,
        margin=dict(l=0, r=0, t=30, b=0),
        scene=dict(
            xaxis_title="X mm",
            yaxis_title="Y mm",
            zaxis_title="Z mm",
            aspectmode="data",
            bgcolor="#0e1117",
            xaxis=dict(backgroundcolor="#0e1117", gridcolor="#333"),
            yaxis=dict(backgroundcolor="#0e1117", gridcolor="#333"),
            zaxis=dict(backgroundcolor="#0e1117", gridcolor="#333"),
        ),
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
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

        case_id = st.selectbox("Case ID", cases, index=0)
        st.header("Display")
        show_vectors = st.checkbox("Velocity cones (3D)", value=False)
        vector_stride = st.slider("Vector stride (higher = fewer)", 2, 12, 5)
        show_streamlines = st.checkbox("Streamlines (curved)", value=True)
        st.header("Roadmap (future)")
        st.markdown(
            """
- Shortest path: naris → ostium → sinus  
- Mucus clearance / widened ostium  
- CRS · NAO · NVC diagnosis  
- Polyp detection  
- Patient CT upload  
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
    st.subheader("3D airway + airflow")
    mesh = None
    if data.get("stl_path"):
        try:
            mesh = _load_mesh_decimated(data["stl_path"])
        except Exception as exc:
            st.warning(f"Could not load STL: {exc}")

    streamlines = data["streamlines"] if show_streamlines else []
    ports = bc.get("ports", [])
    # Drop huge face_indices from hover noise — ports already stripped in json for centers
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
        streamlines=streamlines,
        ux=data["ux"],
        uy=data["uy"],
        uz=data["uz"],
        speed=speed,
        airway=airway,
        spacing=data["spacing"],
        origin=data["origin"],
        show_vectors=show_vectors,
        vector_stride=vector_stride,
        max_vector_speed=vmax,
        ports=ports_lite,
    )
    st.plotly_chart(fig3d, use_container_width=True)

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
