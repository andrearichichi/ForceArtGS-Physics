"""Hoi! force CSV adapter; needs the (non-public) CSVs: FORCEARTGS_PHYSICS_HOI_FORCE_DIR=<.../force/v2a2>."""
import os
from pathlib import Path

import pytest

D = os.environ.get("FORCEARTGS_PHYSICS_HOI_FORCE_DIR")


@pytest.mark.skipif(not D, reason="FORCEARTGS_PHYSICS_HOI_FORCE_DIR not set")
def test_drawer_open_roundtrip():
    from forceartgs_physics.adapters.hoi import load_force_csv
    from forceartgs_physics.forces import generalized_effort
    from forceartgs_physics.validation.identifiability import diagnose
    x = load_force_csv(Path(D) / "drawer_open7" / "drawer_open7.csv", "prismatic")
    e = generalized_effort(x.joint, x.forces).levels
    assert len(e) == len(x.trajectory.t_s) and x.forces.calibration["source"].startswith("freeartgs-hoi")
    r = diagnose(x)
    assert "verdict" in r
