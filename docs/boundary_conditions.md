# Boundary conditions & breathing physiology

## Policy (project intent)

| Boundary | Role | CFD treatment |
|----------|------|----------------|
| **Both nostrils** | **Inlet** | Prescribed volumetric flow (split L/R) |
| **Trachea** | **Outlet** | Fixed gauge pressure (0 Pa) / outflow |
| **Mouth** | **Closed** | Not in fluid domain; if present → wall (no-slip) |
| **Airway mucosa** | Wall | No-slip |

Flow is **inspiratory** (air enters nostrils → exits trachea). Expiration can be added later as a second phase or reversed BCs.

## Typical resting breath → CFD flow

Defaults match a quiet adult nasal breath:

| Quantity | Default | Notes |
|----------|---------|--------|
| Tidal volume \(V_T\) | **0.50 L** | ~6–8 mL/kg ideal body weight long-term |
| Respiratory rate | **12 /min** | Breath period \(T = 5\) s |
| Inspiratory fraction | **1/3** | I:E ≈ 1:2 → \(T_i \approx 1.67\) s |
| **Mean inspiratory flow** | **\(Q = V_T / T_i \approx 18\) L/min** | Uniform delivery of \(V_T\) over \(T_i\) |
| CFD duration | **\(T_i\)** | Quasi-steady inspiration held for one typical inhale |

**Minute ventilation** \(V_T \times RR = 6\) L/min is *not* the same as mean inspiratory flow. During inspiration the flow is higher because volume is delivered only over \(T_i\).

### Why quasi-steady?

For early pipeline development we impose a **constant** total nasal flow equal to the mean inspiratory rate, sustained for \(T_i\). That matches:

- volume per breath ≈ typical inhale  
- timescale ≈ typical breath-in length  

Later: transient waveform (sinusoidal or measured) and patient-specific \(V_T\), RR, and L/R split.

## Nostril split

Default **50% / 50%** left/right. Override when patient data (rhinomanometry, acoustic rhinometry, or measured asymmetry) is available:

```powershell
py -3.12 scripts\process_case.py --case P001 --left-flow-fraction 0.6
```

## Mouth closed

With **label-based** masks (NasalSeg labels 1–3), the oral cavity is **never included**. The mouth is therefore closed by construction.

If using HU-only thresholding, oral air can leak into the domain — prefer labels, or explicitly seal the oral cavity before meshing.

## Trachea vs NasalSeg proxy

**Intent:** outlet = true trachea.

**NasalSeg reality:** FOV stops at nasopharynx; no tracheal segment.  
The pipeline places `trachea_outlet_proxy` on the **distal nasopharynx** tip (farthest from the nasal cavities). CFD is still useful for nasal resistance and local flow; full tracheobronchial outflow needs a head–neck CT that includes the trachea.

`outlet_is_proxy: true` is written into `*_boundary_conditions.json` so this is never ambiguous.

## Patient matching (long-term)

| Input | Use |
|-------|-----|
| Body weight | `PatientBreathing.from_weight_kg(weight)` → \(V_T \approx 7\) mL/kg |
| Height / sex | Ideal body weight → better \(V_T\) prior |
| Measured \(V_T\), RR, \(T_i\) | Override defaults directly |
| L/R nasal resistance | `--left-flow-fraction` or measured split |
| Activity level | Raise RR and/or \(V_T\) (e.g. light exercise) |

CLI examples:

```powershell
# Typical adult defaults (VT=0.5 L, RR=12, Ti≈1.67 s, Q≈18 L/min)
py -3.12 scripts\process_case.py --case P001

# Scale VT from body weight
py -3.12 scripts\process_case.py --case P001 --weight-kg 70

# Explicit patient-like set-point
py -3.12 scripts\process_case.py --case P001 `
  --tidal-volume-L 0.45 --respiratory-rate 14 --inspiratory-time-s 1.5
```

## Outputs

After `process_case.py`:

| File | Content |
|------|---------|
| `*_boundary_conditions.json` | Ports, normals, areas, flow split, physiology |
| `*_openfoam_bc_sketch.txt` | OpenFOAM-oriented `0/U` and `0/p` sketch |
| `*_port_markers.ply` | Spheres at port centers for MeshLab QC |
| `*_stats.json` | Includes BC summary |

## OpenFOAM sketch (concept)

```text
left_nostril  / right_nostril : flowRateInletVelocity  (m³/s each)
trachea_*                     : p = 0 gauge, pressureInletOutletVelocity
wall_airway / wall_mouth      : noSlip
```

Exact patch geometry is finalized at volume-mesh time; bulk speeds in the JSON use estimated port areas from label tips and should be replaced by patch-integrated flow rate BCs after meshing.
