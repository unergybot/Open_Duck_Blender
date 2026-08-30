"""Build the native motion archive consumed by mjlab's MotionLoader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class MotionError(ValueError):
    """Animation samples violate the Microduck motion contract."""


def _finite_array(value, name: str, shape_tail: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != len(shape_tail) + 1 or result.shape[1:] != shape_tail:
        raise MotionError(f"{name} must have shape [T,{','.join(map(str, shape_tail))}]")
    if result.shape[0] == 0:
        raise MotionError(f"{name} must contain at least one frame")
    if not np.isfinite(result).all():
        index = tuple(int(item) for item in np.argwhere(~np.isfinite(result))[0])
        raise MotionError(f"{name} contains a non-finite value at index {index}")
    return result


def _derivative(values: np.ndarray, fps: int) -> np.ndarray:
    if values.shape[0] == 1:
        return np.zeros_like(values)
    return np.gradient(values, 1.0 / fps, axis=0, edge_order=1)


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_conjugate(value: np.ndarray) -> np.ndarray:
    result = value.copy()
    result[..., 1:] *= -1
    return result


def _relative_rotvec(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    relative = _quat_multiply(end, _quat_conjugate(start))
    negative = relative[..., 0] < 0
    relative[negative] *= -1
    vector = relative[..., 1:]
    length = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(length, np.clip(relative[..., 0], -1.0, 1.0))
    scale = np.divide(angle, length, out=np.full_like(angle, 2.0), where=length > 1e-12)
    return vector * scale[..., None]


def _angular_velocity(quaternions: np.ndarray, fps: int) -> np.ndarray:
    count = quaternions.shape[0]
    result = np.zeros(quaternions.shape[:-1] + (3,), dtype=np.float64)
    if count == 1:
        return result
    result[0] = _relative_rotvec(quaternions[0], quaternions[1]) * fps
    result[-1] = _relative_rotvec(quaternions[-2], quaternions[-1]) * fps
    if count > 2:
        result[1:-1] = _relative_rotvec(quaternions[:-2], quaternions[2:]) * (fps / 2.0)
    return result


def _normalized_continuous_quaternions(value: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(value, axis=-1)
    if np.any(norms < 1e-12):
        frame, body = (int(item) for item in np.argwhere(norms < 1e-12)[0])
        raise MotionError(f"body_quat_w has a zero quaternion at frame {frame}, body {body}")
    result = value / norms[..., None]
    for frame in range(1, result.shape[0]):
        flip = np.sum(result[frame - 1] * result[frame], axis=-1) < 0
        result[frame, flip] *= -1
    return result


def build_motion_archive(
    joint_pos,
    body_pos_w,
    body_quat_w,
    *,
    fps: int,
    joint_names: Sequence[str],
    body_names: Sequence[str],
    joint_ranges: Sequence[tuple[float, float]],
    source_hashes: Mapping[str, str],
) -> dict[str, np.ndarray]:
    """Validate evaluated frames and derive a native mjlab archive."""
    if fps != 50:
        raise MotionError(f"Microduck motion export requires 50 Hz, got {fps} Hz")
    joint_names = tuple(joint_names)
    body_names = tuple(body_names)
    if len(joint_names) != len(joint_ranges):
        raise MotionError("joint_ranges must match joint_names")
    joints = _finite_array(joint_pos, "joint_pos", (len(joint_names),))
    positions = _finite_array(body_pos_w, "body_pos_w", (len(body_names), 3))
    quaternions = _finite_array(body_quat_w, "body_quat_w", (len(body_names), 4))
    if not (joints.shape[0] == positions.shape[0] == quaternions.shape[0]):
        raise MotionError("joint and body arrays must contain the same frame count")
    for joint, (lower, upper) in enumerate(joint_ranges):
        outside = np.flatnonzero((joints[:, joint] < lower - 1e-6) | (joints[:, joint] > upper + 1e-6))
        if outside.size:
            frame = int(outside[0])
            raise MotionError(
                f"joint {joint_names[joint]!r} exceeds [{lower}, {upper}] at frame {frame}: "
                f"{joints[frame, joint]}"
            )
    quaternions = _normalized_continuous_quaternions(quaternions)
    archive = {
        "joint_pos": joints.astype(np.float32),
        "joint_vel": _derivative(joints, fps).astype(np.float32),
        "body_pos_w": positions.astype(np.float32),
        "body_quat_w": quaternions.astype(np.float32),
        "body_lin_vel_w": _derivative(positions, fps).astype(np.float32),
        "body_ang_vel_w": _angular_velocity(quaternions, fps).astype(np.float32),
        "fps": np.array([fps], dtype=np.int32),
        "schema_version": np.array([1], dtype=np.int32),
        "joint_names": np.asarray(joint_names, dtype=np.str_),
        "body_names": np.asarray(body_names, dtype=np.str_),
        "source_hashes_json": np.array(
            [json.dumps(dict(source_hashes), sort_keys=True, separators=(",", ":"))],
            dtype=np.str_,
        ),
    }
    return archive


def save_motion_npz(path: str | Path, archive: Mapping[str, np.ndarray]) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".npz":
        path = path.with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **archive)
    return path
