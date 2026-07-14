"""Sinus_CFD: CT-based nasal airway segmentation, flow preview, and BCs."""

from .boundary_conditions import build_boundary_setup
from .flow_field import compute_flow_field
from .nasal_passage import analyze_nasal_passage
from .physiology import PatientBreathing
from .pipeline import process_case
from .whole_head import process_whole_head

__all__ = [
    "process_case",
    "process_whole_head",
    "analyze_nasal_passage",
    "PatientBreathing",
    "build_boundary_setup",
    "compute_flow_field",
]
