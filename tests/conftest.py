import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "tools"))
os.environ.setdefault("OMP_NUM_THREADS", "1")


@pytest.fixture(scope="session")
def bench():
    from forceartgs_physics.adapters.forcesapien import A3Benchmark
    bundle = os.environ.get("FORCEARTGS_PHYSICS_BENCHMARK")
    if not bundle:
        pytest.skip("external ForceSAPIEN bundle not configured (not redistributed)")
    return A3Benchmark(bundle)


@pytest.fixture(scope="session")
def fixtures_dir():
    return ROOT / "tests" / "fixtures"


def original_root():
    """Root that contains comparison/freeartgs_all_objects and AKF_PINN (the original research trees)."""
    p = os.environ.get("FORCEARTGS_PHYSICS_ORIGINAL_ROOT")
    return Path(p) if p else None
