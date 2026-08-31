"""Strict loading and Blender import of native mjlab motion archives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
import zipfile

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
_QUATERNION_NORM_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ImportedMotion:
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    root_body_index: int
    source_sha256: str
    fps: int
    frames: int

    @property
    def root_pos_w(self) -> np.ndarray:
        return self.body_pos_w[:, self.root_body_index]

    @property
    def root_quat_wxyz(self) -> np.ndarray:
        return self.body_quat_w[:, self.root_body_index]


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
    array = np.asarray(value)
    if array.ndim != 1:
        raise MotionError(f"{field} must be a one-dimensional name array")
    try:
        actual = tuple(str(item) for item in array.tolist())
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
        raw = Path(path).read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
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
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise MotionError(f"could not load motion archive: {exc}") from exc

    _scalar(loaded["fps"], "fps", 50)
    _scalar(loaded["schema_version"], "schema_version", 1)
    _names(loaded["joint_names"], "joint_names", profile.joint_names)
    _names(loaded["body_names"], "body_names", profile.body_names)
    _validate_source_hashes(loaded["source_hashes_json"])
    joint_pos_shape = np.asarray(loaded["joint_pos"]).shape
    if len(joint_pos_shape) != 2:
        raise MotionError(
            "joint_pos must have shape "
            f"[T,{len(profile.joint_names)}], got {joint_pos_shape}"
        )
    frames = int(joint_pos_shape[0])
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
    nonunit = np.argwhere(np.abs(norms - 1.0) > _QUATERNION_NORM_TOLERANCE)
    if nonunit.size:
        frame, body = (int(value) for value in nonunit[0])
        raise MotionError(
            f"body_quat_w at frame {frame}, index {body} must have a unit quaternion"
        )
    quaternions = quaternions / norms[..., None]
    for frame in range(1, frames):
        flip = np.sum(quaternions[frame - 1] * quaternions[frame], axis=1) < 0
        quaternions[frame, flip] *= -1

    roots = tuple(body for body in profile.bodies if body.parent is None)
    if len(roots) != 1:
        raise MotionError(f"profile must contain exactly one root body, got {len(roots)}")
    root_name = roots[0].name
    try:
        root_body_index = profile.body_names.index(root_name)
    except ValueError as exc:
        raise MotionError(f"profile root body {root_name!r} is missing from body_names") from exc

    from mathutils import Matrix, Quaternion

    from .blender_bridge import (
        MATRIX_RESIDUAL_TOLERANCE,
        canonical_body_matrices,
        matrix_residual,
    )

    for frame in range(frames):
        root_world = (
            Matrix.Translation(positions[frame, root_body_index])
            @ Quaternion(quaternions[frame, root_body_index]).to_matrix().to_4x4()
        )
        canonical = canonical_body_matrices(root_world, joints[frame], profile)
        for body_index, body_name in enumerate(profile.body_names):
            archived = (
                Matrix.Translation(positions[frame, body_index])
                @ Quaternion(quaternions[frame, body_index]).to_matrix().to_4x4()
            )
            position_m, rotation_rad, affine = matrix_residual(
                archived, canonical[body_name]
            )
            if (
                position_m > MATRIX_RESIDUAL_TOLERANCE
                or rotation_rad > MATRIX_RESIDUAL_TOLERANCE
                or affine > MATRIX_RESIDUAL_TOLERANCE
            ):
                raise MotionError(
                    f"body transform differs from canonical FK at frame {frame}, "
                    f"body {body_name!r}: position residual {position_m:.6g} m, "
                    f"rotation residual {rotation_rad:.6g} rad"
                )
    return ImportedMotion(
        joint_pos=joints,
        body_pos_w=positions,
        body_quat_w=quaternions,
        root_body_index=root_body_index,
        source_sha256=source_sha256,
        fps=50,
        frames=frames,
    )


def _action_fcurves(action):
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return tuple(legacy)
    return tuple(
        fcurve
        for layer in action.layers
        for strip in layer.strips
        if strip.type == "KEYFRAME"
        for channelbag in strip.channelbags
        for fcurve in channelbag.fcurves
    )


def import_motion_action(
    armature,
    profile: RobotProfile,
    path: str | Path,
    *,
    action_name: str,
    motion_kind: str = "mjlab_import",
    before_mutation=None,
):
    """Import a validated motion as a root-moving action transactionally."""
    from mathutils import Matrix, Quaternion
    import bpy

    from .blender_bridge import force_fk, reset_canonical_pose

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
    animation_data = armature.animation_data
    had_animation_data = animation_data is not None
    previous_action = animation_data.action if animation_data is not None else None
    previous_matrix = armature.matrix_world.copy()
    previous_rotation_mode = armature.rotation_mode
    previous_pose = {
        bone.name: (bone.rotation_mode, bone.matrix_basis.copy())
        for bone in armature.pose.bones
    }
    has_mouth_state = hasattr(armature, "duck_mouth_open")
    previous_mouth_open = (
        float(armature.duck_mouth_open) if has_mouth_state else None
    )
    previous_scene = (
        scene.frame_start,
        scene.frame_end,
        scene.frame_current,
        scene.frame_subframe,
        scene.render.fps,
        scene.render.fps_base,
    )
    constraint_influences = tuple(
        (constraint, float(constraint.influence))
        for pose_bone in armature.pose.bones
        for constraint in pose_bone.constraints
        if constraint.name.startswith("DUCK_IK")
    )
    had_fk_ik = "fk_ik" in armature.keys()
    previous_fk_ik = armature.get("fk_ik")
    action = None
    try:
        if before_mutation is not None:
            before_mutation()
        if animation_data is None:
            animation_data = armature.animation_data_create()
        action = bpy.data.actions.new(action_name)
        action.use_fake_user = True
        action.use_frame_range = True
        action.frame_start = 1.0
        action.frame_end = float(motion.frames)
        action["duck_motion_kind"] = motion_kind
        action["duck_source_sha256"] = motion.source_sha256
        action["duck_loopable"] = False
        force_fk(armature)
        reset_canonical_pose(armature, profile)
        animation_data.action = action
        armature.rotation_mode = "QUATERNION"
        scene.render.fps = motion.fps
        scene.render.fps_base = 1.0
        scene.frame_start = 1
        scene.frame_end = motion.frames
        joint_by_child = {joint.child_body: joint for joint in profile.joints}
        previous_root_quaternion = None
        for index in range(motion.frames):
            frame = index + 1
            scene.frame_set(frame)
            desired_root_world = (
                Matrix.Translation(motion.root_pos_w[index])
                @ Quaternion(motion.root_quat_wxyz[index]).to_matrix().to_4x4()
            )
            armature.matrix_world = desired_root_world @ root_mjcf_rest.inverted_safe()
            root_quaternion = armature.rotation_quaternion.copy()
            root_quaternion.normalize()
            if (
                previous_root_quaternion is not None
                and previous_root_quaternion.dot(root_quaternion) < 0.0
            ):
                root_quaternion.negate()
            armature.rotation_quaternion = root_quaternion
            previous_root_quaternion = root_quaternion.copy()
            armature.keyframe_insert(data_path="location", frame=frame, group="root")
            armature.keyframe_insert(
                data_path="rotation_quaternion", frame=frame, group="root"
            )
            reset_canonical_pose(armature, profile)
            for joint, bone, angle in zip(
                profile.joints, joint_bones, motion.joint_pos[index]
            ):
                bone.rotation_euler.z = float(angle)
            for body_name in profile.body_names:
                bone = armature.pose.bones[body_name]
                group = joint_by_child.get(body_name)
                group_name = group.name if group is not None else body_name
                bone.keyframe_insert(data_path="location", frame=frame, group=group_name)
                bone.keyframe_insert(data_path="scale", frame=frame, group=group_name)
                bone.keyframe_insert(
                    data_path=(
                        "rotation_euler"
                        if body_name in joint_by_child
                        else "rotation_quaternion"
                    ),
                    frame=frame,
                    group=group_name,
                )
        for fcurve in _action_fcurves(action):
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"
        scene.frame_set(scene.frame_start)
        return action
    except Exception as exc:
        if had_animation_data:
            animation_data.action = previous_action
        elif armature.animation_data is not None:
            armature.animation_data_clear()
        (
            scene.frame_start,
            scene.frame_end,
            current_frame,
            current_subframe,
            scene.render.fps,
            scene.render.fps_base,
        ) = previous_scene
        scene.frame_set(current_frame, subframe=current_subframe)
        armature.rotation_mode = previous_rotation_mode
        armature.matrix_world = previous_matrix
        if has_mouth_state:
            armature.duck_mouth_open = previous_mouth_open
        for bone in armature.pose.bones:
            rotation_mode, matrix_basis = previous_pose[bone.name]
            bone.rotation_mode = rotation_mode
            bone.matrix_basis = matrix_basis
        for constraint, influence in constraint_influences:
            constraint.influence = influence
        if had_fk_ik:
            armature["fk_ik"] = previous_fk_ik
        elif "fk_ik" in armature.keys():
            del armature["fk_ik"]
        if action is not None and action.name in bpy.data.actions:
            bpy.data.actions.remove(action)
        if isinstance(exc, MotionError):
            raise
        raise MotionError(f"could not import motion action: {exc}") from exc
