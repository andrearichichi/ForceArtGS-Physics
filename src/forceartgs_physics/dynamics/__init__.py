"""1-DoF joint dynamics. ``viscous`` is the ForceSAPIEN model used by every validated estimator;
``extensions`` holds forward models for real data (Coulomb/static friction, gravity, bias), kept separate."""
from .viscous import propagate_from_rest, propagate_from_state

__all__ = ["propagate_from_rest", "propagate_from_state"]
