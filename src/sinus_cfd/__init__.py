"""Sinus_CFD: CT-based nasal airway segmentation, flow preview, and BCs."""

from .boundary_conditions import build_boundary_setup
from .flow_field import compute_flow_field
from .physiology import PatientBreathing
from .pipeline import process_case

__all__ = [
    "process_case",
    "PatientBreathing",
    "build_boundary_setup",
    "compute_flow_field",
]
