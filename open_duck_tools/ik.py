"""Deterministic, bounded inverse kinematics for Microduck's physical foot sites."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


POSITION_TOLERANCE_M = 1e-5
PITCH_TOLERANCE_RAD = 1e-5
PITCH_WEIGHT_M_PER_RAD = 0.05


@dataclass(frozen=True)
class LegPose:
    position: np.ndarray
    pitch: float


@dataclass(frozen=True)
class LegIKResult:
    angles: np.ndarray
    reached: bool
    clamped: bool
    objective: float
    position_error_m: float
    pitch_error_rad: float


def _rotation_matrix(quaternion_wxyz) -> np.ndarray:
    w, x, y, z = (float(value) for value in quaternion_wxyz)
    norm = math.hypot(w, x, y, z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("kinematic quaternion must be finite and nonzero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    sine = math.sin(angle)
    cosine = math.cos(angle)
    return np.eye(3) + sine * skew + (1.0 - cosine) * (skew @ skew)


@dataclass(frozen=True)
class LegKinematics:
    side: str
    body_positions: tuple[np.ndarray, ...]
    body_rotations: tuple[np.ndarray, ...]
    axes: tuple[np.ndarray, ...]
    limits: tuple[tuple[float, float], ...]
    home_angles: tuple[float, ...]
    site_position: np.ndarray
    pitch_coefficients: np.ndarray

    def _evaluate(self, angles) -> tuple[np.ndarray, np.ndarray]:
        angles = np.asarray(angles, dtype=np.float64)
        if angles.shape != (4,) or not np.isfinite(angles).all():
            raise ValueError("leg angles must contain four finite values")
        position = np.zeros(3, dtype=np.float64)
        rotation = np.eye(3, dtype=np.float64)
        pivots = []
        world_axes = []
        for offset, rest_rotation, axis, angle in zip(
            self.body_positions,
            self.body_rotations,
            self.axes,
            angles,
            strict=True,
        ):
            position = position + rotation @ offset
            rotation_before_hinge = rotation @ rest_rotation
            pivots.append(position.copy())
            world_axes.append(rotation_before_hinge @ axis)
            rotation = rotation_before_hinge @ _axis_rotation(axis, float(angle))
        site = position + rotation @ self.site_position
        jacobian = np.column_stack(
            [np.cross(axis, site - pivot) for axis, pivot in zip(world_axes, pivots)]
        )
        return site, jacobian

    def forward(self, angles) -> LegPose:
        values = np.asarray(angles, dtype=np.float64)
        position, _ = self._evaluate(values)
        pitch = float(self.pitch_coefficients @ values)
        return LegPose(position, pitch)

    def position_jacobian(self, angles) -> np.ndarray:
        return self._evaluate(angles)[1]


def leg_kinematics(profile, side: str) -> LegKinematics:
    if side not in {"left", "right"}:
        raise ValueError("leg side must be 'left' or 'right'")
    joint_names = tuple(
        f"{side}_{suffix}" for suffix in ("hip_roll", "hip_pitch", "knee", "ankle")
    )
    joints_by_name = {joint.name: joint for joint in profile.joints}
    bodies_by_name = {body.name: body for body in profile.bodies}
    sites_by_name = {site.name: site for site in profile.sites}
    missing = [name for name in joint_names if name not in joints_by_name]
    if missing:
        raise ValueError(f"profile is missing leg joints: {', '.join(missing)}")
    site = sites_by_name.get(f"{side}_foot")
    if site is None:
        raise ValueError(f"profile is missing physical site {side}_foot")
    joints = tuple(joints_by_name[name] for name in joint_names)
    if any(joints[index].parent_body != joints[index - 1].child_body for index in range(1, 4)):
        raise ValueError(f"{side} leg joints do not form a direct body chain")
    if site.parent_body != joints[-1].child_body:
        raise ValueError(f"{side}_foot is not attached to the ankle body")
    home_by_name = dict(zip(profile.joint_names, profile.home_positions))
    pitch_coefficients = (
        np.asarray((0.0, 1.0, -1.0, 1.0), dtype=np.float64)
        if side == "left"
        else np.asarray((0.0, -1.0, 1.0, -1.0), dtype=np.float64)
    )
    return LegKinematics(
        side=side,
        body_positions=tuple(
            np.asarray(bodies_by_name[joint.child_body].position, dtype=np.float64)
            for joint in joints
        ),
        body_rotations=tuple(
            _rotation_matrix(bodies_by_name[joint.child_body].quaternion_wxyz)
            for joint in joints
        ),
        axes=tuple(np.asarray(joint.axis, dtype=np.float64) for joint in joints),
        limits=tuple(tuple(float(value) for value in joint.range_rad) for joint in joints),
        home_angles=tuple(float(home_by_name[name]) for name in joint_names),
        site_position=np.asarray(site.position, dtype=np.float64),
        pitch_coefficients=pitch_coefficients,
    )


def _pitch_error(actual: float, target: float) -> float:
    return math.remainder(actual - target, 2.0 * math.pi)


def _residual(model: LegKinematics, angles: np.ndarray, target, pitch: float):
    pose = model.forward(angles)
    position_error = pose.position - target
    pitch_error = _pitch_error(pose.pitch, pitch)
    weighted = np.concatenate(
        (position_error, np.asarray((PITCH_WEIGHT_M_PER_RAD * pitch_error,)))
    )
    return pose, position_error, pitch_error, weighted


def solve_leg_ik(
    model: LegKinematics,
    target_position,
    target_pitch: float,
    *,
    initial_angles=None,
    pole_sign: float = 1.0,
    max_iterations: int = 200,
) -> LegIKResult:
    """Return the minimum finite weighted objective found across fixed seeds."""
    target = np.asarray(target_position, dtype=np.float64)
    pitch = float(target_pitch)
    if target.shape != (3,) or not np.isfinite(target).all() or not math.isfinite(pitch):
        raise ValueError("IK target position and pitch must be finite")
    lower = np.asarray([limit[0] for limit in model.limits], dtype=np.float64)
    upper = np.asarray([limit[1] for limit in model.limits], dtype=np.float64)
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
        raise ValueError("IK joint limits must be finite and ordered")
    home = np.clip(np.asarray(model.home_angles, dtype=np.float64), lower, upper)
    initial = home if initial_angles is None else np.asarray(initial_angles, dtype=np.float64)
    if initial.shape != (4,) or not np.isfinite(initial).all():
        raise ValueError("initial leg angles must contain four finite values")
    midpoint = 0.5 * (lower + upper)
    branch = 1.0 if float(pole_sign) >= 0.0 else -1.0
    quarter = lower + 0.25 * (upper - lower)
    three_quarter = lower + 0.75 * (upper - lower)
    bent = midpoint.copy()
    bent[2] = three_quarter[2] if branch > 0.0 else quarter[2]
    opposite = midpoint.copy()
    opposite[2] = quarter[2] if branch > 0.0 else three_quarter[2]
    seeds = (np.clip(initial, lower, upper), home, bent, opposite, midpoint)
    best = None
    for seed in seeds:
        angles = seed.copy()
        damping = 1e-4
        for _ in range(max_iterations):
            pose, position_error, pitch_error, weighted = _residual(
                model, angles, target, pitch
            )
            objective = float(weighted @ weighted)
            if not math.isfinite(objective):
                break
            if np.linalg.norm(position_error) <= POSITION_TOLERANCE_M and abs(pitch_error) <= PITCH_TOLERANCE_RAD:
                break
            jacobian = np.vstack(
                (
                    model.position_jacobian(angles),
                    PITCH_WEIGHT_M_PER_RAD * model.pitch_coefficients,
                )
            )
            normal = jacobian.T @ jacobian + damping * np.eye(4)
            try:
                step = np.linalg.solve(normal, -(jacobian.T @ weighted))
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            accepted = False
            scale = 1.0
            for _ in range(12):
                candidate = np.clip(angles + scale * step, lower, upper)
                candidate_weighted = _residual(model, candidate, target, pitch)[3]
                candidate_objective = float(candidate_weighted @ candidate_weighted)
                if math.isfinite(candidate_objective) and candidate_objective < objective:
                    angles = candidate
                    damping = max(1e-10, damping * 0.5)
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                damping = min(1e8, damping * 10.0)
                if damping >= 1e8:
                    break
        pose, position_error, pitch_error, weighted = _residual(model, angles, target, pitch)
        objective = float(weighted @ weighted)
        if math.isfinite(objective) and (best is None or objective < best[0]):
            best = (objective, angles.copy(), float(np.linalg.norm(position_error)), abs(pitch_error))
    if best is None:
        raise ValueError("IK solver could not produce a finite bounded solution")
    objective, angles, position_error_m, pitch_error_rad = best
    reached = position_error_m <= POSITION_TOLERANCE_M and pitch_error_rad <= PITCH_TOLERANCE_RAD
    on_limit = bool(np.any(np.isclose(angles, lower, atol=1e-10)) or np.any(np.isclose(angles, upper, atol=1e-10)))
    return LegIKResult(
        angles=angles,
        reached=reached,
        clamped=(not reached) or on_limit,
        objective=objective,
        position_error_m=position_error_m,
        pitch_error_rad=pitch_error_rad,
    )
