"""Sinus_CFD: CT-based nasal airway segmentation and CFD boundary setup."""

from .boundary_conditions import build_boundary_setup
from .physiology import PatientBreathing
from .pipeline import process_case

__all__ = ["process_case", "PatientBreathing", "build_boundary_setup"]
