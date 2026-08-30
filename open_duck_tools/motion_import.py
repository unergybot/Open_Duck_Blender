"""Strict loading and Blender import of native mjlab motion archives."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .motion import MotionError
from .profile import RobotProfile


_NATIVE_KEYS = frozenset(
    {
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "fps",
        "schema_version",
        "joint_names",
        "body_names",
        "source_hashes_json",
    }
)


@dataclass(frozen=True)
class ImportedMotion:
    joint_pos: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    fps: int
    frames: int


def _array(value, field: str, shape: tuple[int, ...]) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MotionError(f"{field} must contain numeric values") from exc
    if result.shape != shape:
        raise MotionError(f"{field} must have shape {shape}, got {result.shape}")
    bad = np.argwhere(~np.isfinite(result))
    if bad.size:
        index = tuple(int(value) for value in bad[0])
        frame = index[0]
        item = index[1] if len(index) > 1 else 0
        raise MotionError(f"{field} contains a non-finite value at frame {frame}, index {item}")
    return result.copy()


def _names(value, field: str, expected: tuple[str, ...]) -> None:
    try:
        actual = tuple(str(item) for item in np.asarray(value).tolist())
    except (TypeError, ValueError) as exc:
        raise MotionError(f"{field} must be a one-dimensional name array") from exc
    if actual == expected:
        return
    first = next(
        (
            index
            for index, (got, want) in enumerate(zip(actual, expected))
            if got != want
        ),
        min(len(actual), len(expected)),
    )
    got = actual[first] if first < len(actual) else "<missing>"
    want = expected[first] if first < len(expected) else "<missing>"
    raise MotionError(f"{field} differs at index {first}: got {got!r}, expected {want!r}")


def _scalar(value, field: str, expected: int) -> None:
    array = np.asarray(value)
    if array.shape != (1,):
        raise MotionError(f"{field} must have shape (1,)")
    try:
        number = float(array[0])
    except (TypeError, ValueError) as exc:
        raise MotionError(f"{field} must equal {expected}") from exc
    if not np.isfinite(number) or number != expected:
        raise MotionError(f"{field} must equal {expected}")


def _validate_source_hashes(value) -> None:
    array = np.asarray(value)
    if array.shape != (1,):
        raise MotionError("source_hashes_json must have shape (1,)")
    try:
        payload = json.loads(str(array[0]))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MotionError("source_hashes_json must contain JSON") from exc
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in payload.items()
    ):
        raise MotionError("source_hashes_json must contain a string mapping")


def load_motion(path: str | Path, profile: RobotProfile) -> ImportedMotion:
    """Load a complete native archive without altering the Blender scene."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys != _NATIVE_KEYS:
                missing = sorted(_NATIVE_KEYS - keys)
                extra = sorted(keys - _NATIVE_KEYS)
                details = []
                if missing:
                    details.append(f"missing {', '.join(missing)}")
                if extra:
                    details.append(f"extra {', '.join(extra)}")
                raise MotionError(f"archive keys must match native schema ({'; '.join(details)})")
            loaded = {name: archive[name].copy() for name in _NATIVE_KEYS}
    except MotionError:
        raise
    except (OSError, ValueError, KeyError) as exc:
        raise MotionError(f"could not load motion archive: {exc}") from exc

    _scalar(loaded["fps"], "fps", 50)
    _scalar(loaded["schema_version"], "schema_version", 1)
    _names(loaded["joint_names"], "joint_names", profile.joint_names)
    _names(loaded["body_names"], "body_names", profile.body_names)
    _validate_source_hashes(loaded["source_hashes_json"])
    frames = int(np.asarray(loaded["joint_pos"]).shape[0])
    if frames < 1:
        raise MotionError("joint_pos must contain at least one frame")
    joints = _array(loaded["joint_pos"], "joint_pos", (frames, len(profile.joint_names)))
    _array(loaded["joint_vel"], "joint_vel", (frames, len(profile.joint_names)))
    positions = _array(loaded["body_pos_w"], "body_pos_w", (frames, len(profile.body_names), 3))
    quaternions = _array(loaded["body_quat_w"], "body_quat_w", (frames, len(profile.body_names), 4))
    _array(loaded["body_lin_vel_w"], "body_lin_vel_w", (frames, len(profile.body_names), 3))
    _array(loaded["body_ang_vel_w"], "body_ang_vel_w", (frames, len(profile.body_names), 3))
    for joint_index, joint in enumerate(profile.joints):
        lower, upper = joint.range_rad
        violations = np.argwhere((joints[:, joint_index] < lower) | (joints[:, joint_index] > upper))
        if violations.size:
            frame = int(violations[0, 0])
            raise MotionError(
                f"joint_pos exceeds limit at frame {frame}, index {joint_index} "
                f"({joint.name!r}): {joints[frame, joint_index]} not in [{lower}, {upper}]"
            )
    norms = np.linalg.norm(quaternions, axis=-1)
    zeroes = np.argwhere(norms < 1e-12)
    if zeroes.size:
        frame, body = (int(value) for value in zeroes[0])
        raise MotionError(f"body_quat_w has a zero quaternion at frame {frame}, index {body}")
    root_quaternions = quaternions[:, 0] / norms[:, 0, None]
    for frame in range(1, frames):
        if float(np.dot(root_quaternions[frame - 1], root_quaternions[frame])) < 0:
            root_quaternions[frame] *= -1
    return ImportedMotion(
        joint_pos=joints,
        root_pos_w=positions[:, 0].copy(),
        root_quat_wxyz=root_quaternions,
        fps=50,
        frames=frames,
    )


def import_motion_action(armature, profile: RobotProfile, path: str | Path, *, action_name: str):
    """Import a validated motion as a root-moving action transactionally."""
    from mathutils import Matrix, Quaternion
    import bpy

    motion = load_motion(path, profile)
    joint_bones = []
    for joint in profile.joints:
        pose_bone = armature.pose.bones.get(joint.child_body)
        if pose_bone is None:
            raise MotionError(
                f"armature is missing pose bone for joint {joint.name!r}: {joint.child_body!r}"
            )
        joint_bones.append(pose_bone)
    root = next((body for body in profile.bodies if body.parent is None), None)
    if root is None:
        raise MotionError("profile has no root body")
    root_mjcf_rest = (
        Matrix.Translation(root.position)
        @ Quaternion(root.quaternion_wxyz).to_matrix().to_4x4()
    )

    scene = bpy.context.scene
    animation_data = armature.animation_data_create()
    previous_action = animation_data.action
    previous_matrix = armature.matrix_world.copy()
    previous_rotation_mode = armature.rotation_mode
    previous_pose = {
        bone.name: (bone.rotation_mode, bone.rotation_euler.copy()) for bone in joint_bones
    }
    previous_scene = (scene.frame_start, scene.frame_end, scene.frame_current, scene.render.fps)
    action = None
    try:
        action = bpy.data.actions.new(action_name)
        action.use_fake_user = True
        animation_data.action = action
        armature.rotation_mode = "QUATERNION"
        scene.render.fps = motion.fps
        scene.frame_start = 1
        scene.frame_end = motion.frames
        for index in range(motion.frames):
            frame = index + 1
            scene.frame_set(frame)
            desired_root_world = (
                Matrix.Translation(motion.root_pos_w[index])
                @ Quaternion(motion.root_quat_wxyz[index]).to_matrix().to_4x4()
            )
            armature.matrix_world = desired_root_world @ root_mjcf_rest.inverted_safe()
            armature.keyframe_insert(data_path="location", frame=frame, group="root")
            armature.keyframe_insert(
                data_path="rotation_quaternion", frame=frame, group="root"
            )
            for joint, bone, angle in zip(profile.joints, joint_bones, motion.joint_pos[index]):
                bone.rotation_mode = "XYZ"
                bone.rotation_euler.z = float(angle)
                bone.keyframe_insert(
                    data_path="rotation_euler", index=2, frame=frame, group=joint.name
                )
        scene.frame_set(scene.frame_start)
        return action
    except Exception as exc:
        animation_data.action = previous_action
        armature.matrix_world = previous_matrix
        armature.rotation_mode = previous_rotation_mode
        for bone in joint_bones:
            rotation_mode, rotation_euler = previous_pose[bone.name]
            bone.rotation_mode = rotation_mode
            bone.rotation_euler = rotation_euler
        scene.frame_start, scene.frame_end, current_frame, scene.render.fps = previous_scene
        scene.frame_set(current_frame)
        if action is not None and action.name in bpy.data.actions:
            bpy.data.actions.remove(action)
        if isinstance(exc, MotionError):
            raise
        raise MotionError(f"could not import motion action: {exc}") from exc
