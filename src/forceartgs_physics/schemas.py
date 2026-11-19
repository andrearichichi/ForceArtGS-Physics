"""Typed, dataset-independent data model.

All arrays are ``numpy.float64``. Units follow :mod:`forceartgs_physics.units`; every geometric
quantity carries the name of the frame it is expressed in, and the library refuses to combine
quantities from different frames (no implicit transform).

Kinematics come from an upstream perception pipeline (ForceArtGS joint discovery); this library
never re-estimates the joint. Forces come from a robot / sensor / simulator record.
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .units import JOINT_TYPES, units_for


class KinematicStatus(str, enum.Enum):
    """Outcome of the upstream joint discovery for this interaction.

    Only ``IDENTIFIED`` kinematics can be used for physical identification. Every other status
    makes :meth:`Interaction.require_identifiable` raise :class:`NotIdentifiableError`, so an
    ambiguous or unobservable joint can never silently produce physical parameters.
    """

    IDENTIFIED = "identified"        # joint type, axis, origin and q(t) accepted upstream
    AMBIGUOUS = "ambiguous"          # several joint hypotheses remain (e.g. revolute vs prismatic)
    UNOBSERVABLE = "unobservable"    # motion too small / not visible to determine the joint
    STATIC = "static"                # no joint motion detected in the interaction
    REJECTED = "rejected"            # upstream quality gates failed
    UNKNOWN = "unknown"              # upstream did not report a status


class NotIdentifiableError(RuntimeError):
    """Raised when the inputs do not allow the requested identification."""


def _arr(x, shape_tail=None, name="array") -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if shape_tail is not None and a.shape[1:] != shape_tail:
        raise ValueError(f"{name} must have shape [N, {', '.join(map(str, shape_tail))}], got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _strictly_increasing(t: np.ndarray, name: str) -> None:
    if t.ndim != 1 or len(t) < 2 or not np.all(np.diff(t) > 0):
        raise ValueError(f"{name} must be a 1-D strictly increasing array with at least two samples")


@dataclass
class JointKinematics:
    """Joint geometry from the kinematic pipeline.

    axis   : direction, unit-normalised on construction, in ``frame``; positive q is a right-hand rotation
             about +axis (revolute) or a translation along +axis (prismatic).
    origin : a point on the joint axis [m] in ``frame`` (required for revolute joints; for prismatic joints
             it is informative only).
    limits : (q_min, q_max) in rad or m if known, else None.
    axis_std_rad / origin_std_m : 1-sigma uncertainties if the upstream pipeline provides them.
    """

    joint_type: str
    axis: np.ndarray
    origin: np.ndarray
    frame: str
    status: KinematicStatus = KinematicStatus.IDENTIFIED
    limits: Optional[tuple] = None
    axis_std_rad: Optional[float] = None
    origin_std_m: Optional[float] = None
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.joint_type not in JOINT_TYPES:
            raise ValueError(f"joint_type must be one of {JOINT_TYPES}, got {self.joint_type!r}")
        self.status = KinematicStatus(self.status)
        self.axis = np.asarray(self.axis, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(self.axis))
        if not np.isfinite(n) or n <= 0:
            raise ValueError("joint axis must be a finite non-zero vector")
        self.axis = self.axis / n
        self.origin = np.asarray(self.origin, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(self.origin)):
            raise ValueError("joint origin must be finite")

    @property
    def units(self) -> dict:
        return units_for(self.joint_type)


@dataclass
class InitialState:
    """A DECLARED initial condition (e.g. 'at rest before the push'). Estimators of the validated
    benchmark do not use it (offset and initial velocity are nuisance unknowns there)."""

    t_s: float
    q: float
    qdot: float
    source: str


@dataclass
class Trajectory:
    """Observed joint coordinate q(t).

    t_s : observation times [s] in the SAME clock as the force record.
    q   : joint coordinate [rad | m]; the offset is arbitrary unless ``zero_reference`` says otherwise.
    q_std : optional per-sample 1-sigma uncertainty [rad | m].
    """

    t_s: np.ndarray
    q: np.ndarray
    q_std: Optional[np.ndarray] = None
    zero_reference: str = "arbitrary offset"
    initial_state: Optional[InitialState] = None
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_s = _arr(self.t_s, name="t_s")
        self.q = _arr(self.q, name="q")
        _strictly_increasing(self.t_s, "trajectory t_s")
        if self.q.shape != self.t_s.shape:
            raise ValueError("q and t_s must have the same shape")
        if self.q_std is not None:
            self.q_std = _arr(self.q_std, name="q_std")
            if self.q_std.shape != self.t_s.shape:
                raise ValueError("q_std must match t_s")


@dataclass
class ForceRecord:
    """Applied interaction forces (force ON the object), sampled as a forward zero-order hold:
    sample k is held on [t_k, t_{k+1}); the first sample is also held back to t = 0 (the convention of
    every validated estimator).

    force_N   : [N, 3] force applied to the moving part, in ``frame`` [N].
    point_m   : [N, 3] application point in ``frame`` [m] (required for revolute joints).
    torque_Nm : optional [N, 3] pure couple applied in addition to the force (e.g. a measured wrench
                transported to ``point_m``), in ``frame`` [N*m]. It adds ``axis . torque`` to the revolute effort.
    contact_valid : optional [N] bool; samples marked False are treated as NO applied force only if the
                caller applies :func:`forceartgs_physics.forces.preprocessing.apply_contact_mask` (explicit).
    force_std_N / point_std_m : optional 1-sigma uncertainties (scalars or [N]).
    calibration : free-form record of every preprocessing step applied (bias removal, lag, frame chain).
    """

    t_s: np.ndarray
    force_N: np.ndarray
    frame: str
    point_m: Optional[np.ndarray] = None
    torque_Nm: Optional[np.ndarray] = None
    contact_valid: Optional[np.ndarray] = None
    force_std_N: Optional[Any] = None
    point_std_m: Optional[Any] = None
    wrench_frame: Optional[str] = None
    tcp: Optional[dict] = None
    calibration: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t_s = _arr(self.t_s, name="force t_s")
        _strictly_increasing(self.t_s, "force t_s")
        self.force_N = _arr(self.force_N, (3,), "force_N")
        n = len(self.t_s)
        if len(self.force_N) != n:
            raise ValueError("force_N must have one row per force sample")
        for name in ("point_m", "torque_Nm"):
            v = getattr(self, name)
            if v is not None:
                v = _arr(v, (3,), name)
                if len(v) != n:
                    raise ValueError(f"{name} must have one row per force sample")
                setattr(self, name, v)
        if self.contact_valid is not None:
            self.contact_valid = np.asarray(self.contact_valid, dtype=bool).reshape(n)


@dataclass
class Interaction:
    """One observed interaction: joint + q(t) + forces, all in explicit frames and SI units."""

    name: str
    joint: JointKinematics
    trajectory: Trajectory
    forces: ForceRecord
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.joint.frame != self.forces.frame:
            raise ValueError(f"joint frame {self.joint.frame!r} differs from force frame {self.forces.frame!r}: "
                             "transform one of them explicitly (forces.frames) before building the interaction")
        if self.joint.joint_type == "revolute" and self.forces.point_m is None:
            raise ValueError("a revolute joint needs force application points (point_m)")

    @property
    def joint_type(self) -> str:
        return self.joint.joint_type

    @property
    def units(self) -> dict:
        return self.joint.units

    def require_identifiable(self) -> None:
        if self.joint.status is not KinematicStatus.IDENTIFIED:
            raise NotIdentifiableError(
                f"{self.name}: upstream kinematic status is {self.joint.status.value!r}; physical parameters are "
                "not estimated for ambiguous / unobservable / static / rejected joints")

    def fingerprint(self) -> str:
        """sha256 of the numerical content (inputs provenance of a result)."""
        h = hashlib.sha256()
        for a in (self.trajectory.t_s, self.trajectory.q, self.forces.t_s, self.forces.force_N,
                  self.forces.point_m if self.forces.point_m is not None else np.zeros(0),
                  self.forces.torque_Nm if self.forces.torque_Nm is not None else np.zeros(0),
                  self.joint.axis, self.joint.origin):
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
        h.update(self.joint.joint_type.encode())
        return h.hexdigest()


# ---------------------------------------------------------------------------------------------- outputs
@dataclass
class PhysicalParameters:
    """Identified parameters of ``I q'' + c q' = tau`` (model 'viscous_1dof', the ForceSAPIEN model).

    inertia : kg*m^2 (revolute) or kg (prismatic, effective mass).
    damping : N*m*s/rad (revolute) or N*s/m (prismatic).
    identifiable : None if not assessed, else the verdict of the identifiability diagnostic.
    uncertainty : optional 1-sigma relative std (fraction) of I and c, with the method used.
    """

    joint_type: str
    inertia: float
    damping: float
    model: str = "viscous_1dof"
    identifiable: Optional[bool] = None
    uncertainty: Optional[dict] = None
    extra: dict = field(default_factory=dict)

    @property
    def units(self) -> dict:
        u = units_for(self.joint_type)
        return {"inertia": u["inertia"], "damping": u["damping"]}

    @property
    def decay_time_s(self) -> float | None:
        try:
            return self.inertia / self.damping if self.damping > 0 else None
        except (TypeError, ZeroDivisionError):
            return None

    def require_exportable(self) -> None:
        if self.identifiable is not True:
            raise NotIdentifiableError("physical export requires a positive identifiability verdict")
        try:
            valid = (self.joint_type in JOINT_TYPES and np.isfinite(self.inertia)
                     and np.isfinite(self.damping) and self.inertia > 0 and self.damping > 0)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise NotIdentifiableError("physical export requires finite positive inertia/effective mass and damping")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["units"] = self.units
        d["decay_time_s"] = self.decay_time_s
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PhysicalParameters":
        keys = {"joint_type", "inertia", "damping", "model", "identifiable", "uncertainty", "extra"}
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class IdentificationResult:
    """Result of one estimator on one (or several) interactions."""

    estimator: str
    status: str                                  # ok | numerical_failure | failed | not_identifiable
    parameters: Optional[PhysicalParameters]
    interactions: list
    config: dict = field(default_factory=dict)
    q_fit: Optional[dict] = None                 # {"t_s": [...], "q": [...]} estimator's state estimate on the training data
    fit_metrics: dict = field(default_factory=dict)
    identifiability: Optional[dict] = None
    per_seed: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    runtime_s: Optional[float] = None
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["parameters"] = None if self.parameters is None else self.parameters.to_dict()
        return _jsonable(d)

    def require_exportable(self) -> None:
        ident = self.identifiability or {}
        if (self.status != "ok" or self.parameters is None or ident.get("identifiable") is not True
                or ident.get("kinematic_status") != "identified"
                or any(ident.get(k) is False for k in ("identifiable_inertia", "identifiable_damping"))):
            raise NotIdentifiableError("physical export requires successful identification and a positive diagnostic")
        self.parameters.require_exportable()

    def save(self, path) -> Path:
        """Save the full diagnostic record, including unsuccessful/unidentifiable fit candidates.

        This is a research report, not an export of deployable physical parameters.
        Use api.save_parameters for a guarded physical export.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))
        return path

    @classmethod
    def load(cls, path) -> "IdentificationResult":
        d = json.loads(Path(path).read_text())
        p = d.get("parameters")
        d["parameters"] = None if p is None else PhysicalParameters.from_dict(p)
        keys = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in keys})


@dataclass
class Prediction:
    """Open-loop prediction with frozen parameters. q is the displacement from the start state unless
    ``initial_state`` is given."""

    t_s: np.ndarray
    q: np.ndarray
    parameters: PhysicalParameters
    effort_model: str
    initial_state: dict
    effort_t_s: Optional[np.ndarray] = None
    effort: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        return _jsonable(dict(t_s=self.t_s, q=self.q, parameters=self.parameters.to_dict(), effort_model=self.effort_model,
                              initial_state=self.initial_state, effort_t_s=self.effort_t_s, effort=self.effort,
                              units=dict(q=units_for(self.parameters.joint_type)["q"], t_s="s",
                                         effort=units_for(self.parameters.joint_type)["effort"])))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, enum.Enum):
        return x.value
    if isinstance(x, float) and not np.isfinite(x):
        return str(x)
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):     # torch tensors
        return _jsonable(x.tolist())
    return x
