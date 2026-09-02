"""Convert robot-keyframe-kit dense motion into the native Open Duck archive."""

from __future__ import annotations

from typing import Any

import numpy as np

from .motion import MotionError, build_motion_archive


def archive_from_keyframe_data(data: dict[str, Any], profile) -> dict[str, np.ndarray]:
    try:
        time = np.asarray(data["time"], dtype=np.float64)
        joint_pos = np.asarray(data["action"], dtype=np.float64)
        body_pos = np.asarray(data["body_pos"], dtype=np.float64)
        body_quat = np.asarray(data["body_quat"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise MotionError(f"invalid keyframe-kit motion: {exc}") from exc
    if time.ndim != 1 or time.size < 1:
        raise MotionError("keyframe-kit time must contain at least one frame")
    if time.size > 1 and not np.allclose(np.diff(time), 0.02, rtol=0.0, atol=1e-7):
        raise MotionError("keyframe-kit motion must be sampled at exactly 50 Hz")
    names = tuple(str(name) for name in data.get("joint_names", ()))
    if names != profile.joint_names:
        raise MotionError(
            f"joint_names must match canonical order: got {names!r}, expected {profile.joint_names!r}"
        )
    if not (joint_pos.shape[0] == body_pos.shape[0] == body_quat.shape[0] == time.size):
        raise MotionError("keyframe-kit arrays must share their frame count")
    source_hashes = {
        name: str(data[name])
        for name in ("source_video_sha256", "retarget_config_sha256")
        if name in data
    }
    return build_motion_archive(
        joint_pos,
        body_pos,
        body_quat,
        fps=50,
        joint_names=profile.joint_names,
        body_names=profile.body_names,
        joint_ranges=tuple(joint.range_rad for joint in profile.joints),
        source_hashes=source_hashes,
    )
