"""Interior air must be inside the head, and a naris seed must be on the lumen.

Two failures these cover, both found by rendering CQ500CT390 and THCA and looking
at the result rather than at the numbers:

1. ``interior_air_within_hull`` bounded recovered air by the per-slice CONVEX
   HULL of the body. The hull spans facial concavities, so the wedge of room air
   in front of the nose came back as interior air and was then named a maxillary
   sinus, and on THCA a band hugging the cheeks stayed connected to the lumen
   through the nostrils and became 6.8 mL of the CFD domain.

2. Removing that ambient air disconnected THCA's right vestibule pocket from the
   lumen, and the naris seed snapped onto the pocket. The competing flood then
   returned 55 voxels for the whole right cavity.
"""

from __future__ import annotations

import numpy as np

from sinus_cfd.auto_airway import main_air_component, snap_seed_to_air
from sinus_cfd.whole_head import interior_air_within_hull

ISO = (1.0, 1.0, 1.0)
VOX_ML = 1e-3


def _head_with_a_notch():
    """A block of tissue with a sealed cavity inside and a deep notch in its face.

    The notch is the nose: a concavity open to the outside. Its air is ambient.
    The cavity is a sinus. A convex hull cannot tell them apart; a hole fill can.
    """
    hu = np.full((24, 40, 40), -1000.0, dtype=np.float32)
    body = np.zeros((24, 40, 40), dtype=bool)
    body[4:20, 8:32, 8:32] = True                 # the head
    body[4:20, 8:16, 17:23] = False               # a notch cut into its front face
    hu[body] = 60.0                               # soft tissue
    cavity = np.zeros_like(body)
    cavity[9:15, 20:26, 12:18] = True             # sealed cavity inside the head
    body[cavity] = False
    hu[cavity] = -900.0                           # air, enclosed
    notch = np.zeros_like(body)
    notch[4:20, 8:16, 17:23] = True               # air in the notch, ambient
    return hu, body, cavity, notch


def test_ambient_air_in_a_facial_concavity_is_not_interior():
    hu, body, cavity, notch = _head_with_a_notch()
    got = interior_air_within_hull(hu, body, -300.0, VOX_ML, min_speck_ml=0.0)
    assert got[cavity].all(), "the sealed cavity is interior air and must be kept"
    assert not got[notch].any(), (
        f"{int(got[notch].sum())} voxels of ambient air in the facial notch were "
        "recovered as interior air -- this is the convex-hull leak"
    )


def test_interior_air_stays_inside_the_silhouette():
    hu, body, _cavity, _notch = _head_with_a_notch()
    got = interior_air_within_hull(hu, body, -300.0, VOX_ML, min_speck_ml=0.0)
    # Nothing outside the head block at all.
    outside = np.ones_like(body)
    outside[4:20, 8:32, 8:32] = False
    assert not (got & outside).any()


def test_no_interior_air_fails_rather_than_returning_ambient():
    """A body with no enclosed air returns empty, never the surrounding room."""
    hu = np.full((16, 32, 32), -1000.0, dtype=np.float32)
    body = np.zeros((16, 32, 32), dtype=bool)
    body[4:12, 8:24, 8:24] = True
    hu[body] = 60.0
    got = interior_air_within_hull(hu, body, -300.0, VOX_ML, min_speck_ml=0.0)
    assert not got.any()


def test_naris_seed_does_not_snap_onto_a_stranded_pocket():
    """The seed must land on the air the flood traverses, not a nearby island."""
    air = np.zeros((20, 40, 20), dtype=bool)
    air[6:14, 10:34, 6:14] = True          # the lumen
    pocket = np.zeros_like(air)
    pocket[9:11, 4:6, 9:11] = True         # a small island in front of it
    air |= pocket
    seed = (10, 3, 10)                     # a landmark in tissue, nearer the island

    naive = snap_seed_to_air(air, seed, ISO)
    assert pocket[naive], "precondition: the nearest air really is the island"

    snapped = snap_seed_to_air(main_air_component(air), seed, ISO)
    assert snapped is not None
    assert not pocket[snapped]
    assert air[snapped]


def test_snapping_to_the_main_component_cannot_reach_across_an_obstruction():
    """The radius still bounds the move, so a far lumen fails instead of joining."""
    air = np.zeros((20, 60, 20), dtype=bool)
    air[6:14, 40:56, 6:14] = True          # lumen, far from the landmark
    seed = (10, 2, 10)
    assert snap_seed_to_air(main_air_component(air), seed, ISO, radius_mm=10.0) is None
