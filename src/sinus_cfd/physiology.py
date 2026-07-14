"""
Resting breathing physiology for nasal CFD boundary conditions.

Default mode: quasi-steady inspiration at the mean inspiratory flow of a
typical adult tidal breath, held for a typical inspiratory duration.

Long-term: scale tidal volume / rate / inspiratory time to the patient.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# ---- literature-typical resting adult (quiet nasal breathing) ----
# Tidal volume ~500 mL; RR ~12 /min; I:E ≈ 1:2 → inspiratory fraction ~1/3
# Mean inspiratory flow Q = VT / Ti  →  ~15–18 L/min (common CFD set-point: 15 L/min)


@dataclass
class PatientBreathing:
    """
    Patient / scenario breathing parameters.

    Flow for steady (or quasi-steady) inspiratory CFD:
        Q_L_per_min = tidal_volume_L / inspiratory_time_s * 60

    Volume delivered during one inspiration:
        V = Q * Ti  (= tidal_volume_L by construction)
    """

    # Identity / scaling context (optional, for future patient matching)
    patient_id: str | None = None
    sex: str | None = None  # "F" | "M" | None
    height_cm: float | None = None
    weight_kg: float | None = None
    age_years: float | None = None

    # Core respiratory set-points (defaults = typical resting adult)
    tidal_volume_L: float = 0.50
    respiratory_rate_per_min: float = 12.0
    # Fraction of breath period that is inspiration (I:E=1:2 → 1/3)
    inspiratory_fraction: float = 1.0 / 3.0
    # If set, overrides inspiratory_fraction for Ti
    inspiratory_time_s: float | None = None

    # How total nasal flow is split between nostrils (sums to 1.0)
    left_nostril_flow_fraction: float = 0.5
    right_nostril_flow_fraction: float = 0.5

    # Fluid (air at ~37 °C, atmospheric) — SI for CFD templates
    density_kg_m3: float = 1.125
    dynamic_viscosity_Pa_s: float = 1.8e-5

    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        s = self.left_nostril_flow_fraction + self.right_nostril_flow_fraction
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"nostril flow fractions must sum to 1.0 (got {s})"
            )
        if self.tidal_volume_L <= 0:
            raise ValueError("tidal_volume_L must be > 0")
        if self.respiratory_rate_per_min <= 0:
            raise ValueError("respiratory_rate_per_min must be > 0")

    @property
    def breath_period_s(self) -> float:
        return 60.0 / self.respiratory_rate_per_min

    @property
    def Ti_s(self) -> float:
        if self.inspiratory_time_s is not None:
            return float(self.inspiratory_time_s)
        return self.breath_period_s * self.inspiratory_fraction

    @property
    def Te_s(self) -> float:
        return max(self.breath_period_s - self.Ti_s, 0.0)

    @property
    def minute_ventilation_L_per_min(self) -> float:
        """VT × RR (expired minute volume), not the same as mean inspiratory flow."""
        return self.tidal_volume_L * self.respiratory_rate_per_min

    @property
    def mean_inspiratory_flow_L_per_min(self) -> float:
        """
        Mean flow during inspiration if VT is delivered uniformly over Ti.

        This is the default quasi-steady CFD inlet flow rate.
        """
        return self.tidal_volume_L / self.Ti_s * 60.0

    @property
    def mean_inspiratory_flow_m3_s(self) -> float:
        """SI volumetric flow for OpenFOAM-style solvers."""
        return self.mean_inspiratory_flow_L_per_min / 1000.0 / 60.0

    def flow_split_L_per_min(self) -> dict[str, float]:
        q = self.mean_inspiratory_flow_L_per_min
        return {
            "total": q,
            "left_nostril": q * self.left_nostril_flow_fraction,
            "right_nostril": q * self.right_nostril_flow_fraction,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(
            {
                "breath_period_s": self.breath_period_s,
                "Ti_s_effective": self.Ti_s,
                "Te_s_effective": self.Te_s,
                "minute_ventilation_L_per_min": self.minute_ventilation_L_per_min,
                "mean_inspiratory_flow_L_per_min": self.mean_inspiratory_flow_L_per_min,
                "mean_inspiratory_flow_m3_s": self.mean_inspiratory_flow_m3_s,
                "flow_split_L_per_min": self.flow_split_L_per_min(),
                "cfd_mode": "quasi_steady_inspiration",
                "cfd_duration_s": self.Ti_s,
                "cfd_note": (
                    "Impose constant total nasal flow equal to mean inspiratory "
                    "flow for one typical inspiratory duration Ti. Mouth closed; "
                    "inlets = both nostrils; outlet = trachea (or distal airway proxy)."
                ),
            }
        )
        return d

    @classmethod
    def typical_resting_adult(cls, **overrides: Any) -> PatientBreathing:
        """Quiet breathing: VT=0.5 L, RR=12, I:E≈1:2 → Q≈18 L/min."""
        return cls(**overrides)

    @classmethod
    def from_weight_kg(
        cls,
        weight_kg: float,
        mL_per_kg: float = 7.0,
        respiratory_rate_per_min: float = 12.0,
        inspiratory_fraction: float = 1.0 / 3.0,
        **overrides: Any,
    ) -> PatientBreathing:
        """
        Scale tidal volume as ~6–8 mL/kg (default 7 mL/kg).

        Long-term patient matching starts here; refine with measured VT/RR when available.
        """
        vt_L = (mL_per_kg * weight_kg) / 1000.0
        return cls(
            weight_kg=weight_kg,
            tidal_volume_L=vt_L,
            respiratory_rate_per_min=respiratory_rate_per_min,
            inspiratory_fraction=inspiratory_fraction,
            notes=[f"VT scaled at {mL_per_kg} mL/kg from weight_kg={weight_kg}"],
            **overrides,
        )


def summary_text(params: PatientBreathing) -> str:
    split = params.flow_split_L_per_min()
    lines = [
        "Breathing / CFD flow set-point",
        f"  VT = {params.tidal_volume_L:.3f} L",
        f"  RR = {params.respiratory_rate_per_min:.1f} /min  (period {params.breath_period_s:.2f} s)",
        f"  Ti = {params.Ti_s:.2f} s   Te = {params.Te_s:.2f} s",
        f"  Minute ventilation VT×RR = {params.minute_ventilation_L_per_min:.2f} L/min",
        f"  Mean inspiratory flow (CFD) = {params.mean_inspiratory_flow_L_per_min:.2f} L/min",
        f"    left nostril  = {split['left_nostril']:.2f} L/min",
        f"    right nostril = {split['right_nostril']:.2f} L/min",
        f"  Hold quasi-steady inspiration for Ti = {params.Ti_s:.2f} s",
        "  Mouth: closed (not in domain / wall)",
        "  Outlet: trachea (distal airway pressure reference)",
    ]
    return "\n".join(lines)
