"""Blender UI and operators shared by generated duck projects."""

from __future__ import annotations

import math
from pathlib import Path
import re

import bpy
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Matrix, Quaternion
import numpy as np

from .blender_bridge import (
    MATRIX_RESIDUAL_TOLERANCE,
    body_samples,
    canonical_body_matrices,
    force_fk,
    joint_angles_from_body_matrices,
    matrix_residual,
    reset_canonical_pose,
)
from .ik import leg_kinematics, solve_leg_ik
from .motion import MotionError, build_motion_archive, save_motion_npz
from .motion_import import import_motion_action
from .profile import ProfileError, profile_from_json


COLORWAYS = {
    "CREAM": ("Cream", "#f7e6cb", "#f59e0b"),
    "GRAPHITE": ("Graphite", "#6c6a68", "#f7c948"),
    "LAVENDER": ("Lavender", "#bfa9cf", "#f7c948"),
    "SKY": ("Sky", "#a9dbe8", "#f59e0b"),
}
BEGINNER_ACTION_PRESETS = (("Policy_alpha_walking_forward", "Walk"),)
_IK_UPDATE_GUARD: set[int] = set()


def _srgb_channel(value: int) -> float:
    value /= 255.0
    return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4


def _hex_rgba(value: str) -> tuple[float, float, float, float]:
    rgb = tuple(_srgb_channel(int(value[index : index + 2], 16)) for index in (1, 3, 5))
    return (*rgb, 1.0)


def profile_from_armature(armature: bpy.types.Object):
    encoded = armature.data.get("duck_robot_profile_json")
    if not encoded:
        raise ProfileError("active armature has no embedded duck_robot_profile_json")
    return profile_from_json(str(encoded))


def _evaluated_body_matrices(armature: bpy.types.Object, body_names) -> dict:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = armature.evaluated_get(depsgraph)
    missing = [name for name in body_names if evaluated.pose.bones.get(name) is None]
    if missing:
        raise MotionError(f"armature is missing body bones: {', '.join(missing)}")
    return {
        name: evaluated.matrix_world @ evaluated.pose.bones[name].matrix
        for name in body_names
    }


def _rest_body_matrices(armature: bpy.types.Object, body_names) -> dict:
    missing = [name for name in body_names if armature.data.bones.get(name) is None]
    if missing:
        raise MotionError(f"armature is missing rest bones: {', '.join(missing)}")
    return {name: armature.data.bones[name].matrix_local.copy() for name in body_names}


def collect_armature_motion(
    armature: bpy.types.Object,
    profile,
    frame_start: int,
    frame_end: int,
) -> dict[str, np.ndarray]:
    if frame_end < frame_start:
        raise MotionError("export frame end precedes frame start")
    scene = bpy.context.scene
    fps_base = float(scene.render.fps_base)
    effective_fps = (
        float(scene.render.fps) / fps_base
        if math.isfinite(fps_base) and fps_base > 0.0
        else math.inf
    )
    if not math.isclose(effective_fps, 50.0, rel_tol=0.0, abs_tol=1e-9):
        raise MotionError(
            f"effective frame rate is {effective_fps:.12g} Hz; "
            "Microduck motion export requires 50 Hz"
        )
    original_frame = scene.frame_current
    original_subframe = scene.frame_subframe
    joint_frames = []
    body_positions = []
    body_quaternions = []
    rest_matrices = _rest_body_matrices(armature, profile.body_names)
    roots = tuple(body for body in profile.bodies if body.parent is None)
    if len(roots) != 1:
        raise MotionError(f"profile must contain exactly one root body, got {len(roots)}")
    root_spec = roots[0]
    root_name = root_spec.name
    root_mjcf_rest = Matrix.Translation(root_spec.position) @ Quaternion(
        root_spec.quaternion_wxyz
    ).to_matrix().to_4x4()
    try:
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            matrices = _evaluated_body_matrices(armature, profile.body_names)
            angles = joint_angles_from_body_matrices(matrices, profile, rest_matrices)
            joint_frames.append(angles)
            root_world = (
                matrices[root_name]
                @ rest_matrices[root_name].inverted_safe()
                @ root_mjcf_rest
            )
            canonical = canonical_body_matrices(root_world, angles, profile)
            for body_name in profile.body_names:
                position_m, rotation_rad, affine = matrix_residual(
                    matrices[body_name], canonical[body_name]
                )
                if (
                    position_m > MATRIX_RESIDUAL_TOLERANCE
                    or rotation_rad > MATRIX_RESIDUAL_TOLERANCE
                    or affine > MATRIX_RESIDUAL_TOLERANCE
                ):
                    raise MotionError(
                        "evaluated body transform differs from canonical FK at "
                        f"frame {frame}, body {body_name!r}: "
                        f"position residual {position_m:.6g} m, "
                        f"rotation residual {rotation_rad:.6g} rad, "
                        f"affine residual {affine:.6g}"
                    )
            positions, quaternions = body_samples(canonical, profile)
            body_positions.append(positions)
            body_quaternions.append(quaternions)
    finally:
        scene.frame_set(original_frame, subframe=original_subframe)
    return build_motion_archive(
        np.asarray(joint_frames),
        np.asarray(body_positions),
        np.asarray(body_quaternions),
        fps=50,
        joint_names=profile.joint_names,
        body_names=profile.body_names,
        joint_ranges=tuple(joint.range_rad for joint in profile.joints),
        source_hashes=profile.source_sha256,
    )


def _apply_colorway(armature: bpy.types.Object, key: str) -> None:
    _, shell_hex, trim_hex = COLORWAYS[key]
    colors = {"shell": _hex_rgba(shell_hex), "trim": _hex_rgba(trim_hex)}
    for material in bpy.data.materials:
        role = material.get("duck_material_role")
        if role in colors:
            material.diffuse_color = colors[role]
            if material.use_nodes and material.node_tree:
                node = material.node_tree.nodes.get("Principled BSDF")
                if node is not None:
                    node.inputs["Base Color"].default_value = colors[role]


def _colorway_updated(self, _context):
    if self.type == "ARMATURE" and self.get("duck_robot_id") == "microduck-alpha":
        _apply_colorway(self, self.duck_colorway)


def _pose_matrix(pose):
    return Matrix.Translation(pose.position) @ Quaternion(
        pose.quaternion_wxyz
    ).to_matrix().to_4x4()


def _apply_mouth_pose(armature: bpy.types.Object, open_fraction: float) -> None:
    profile = profile_from_armature(armature)
    linkage = profile.mouth
    servo_rad = linkage.closed_rad + float(open_fraction) * (
        linkage.open_rad - linkage.closed_rad
    )
    upper = next(
        (index for index, sample in enumerate(linkage.samples) if sample.servo_rad >= servo_rad),
        len(linkage.samples) - 1,
    )
    lower = max(0, upper - 1)
    first = linkage.samples[lower]
    second = linkage.samples[upper]
    span = second.servo_rad - first.servo_rad
    blend = 0.0 if abs(span) < 1e-12 else (servo_rad - first.servo_rad) / span
    closed = linkage.samples[0]
    for link in linkage.links:
        first_pose = first.poses[link.name]
        second_pose = second.poses[link.name]
        position = tuple(
            a + blend * (b - a)
            for a, b in zip(first_pose.position, second_pose.position)
        )
        rotation = Quaternion(first_pose.quaternion_wxyz).slerp(
            Quaternion(second_pose.quaternion_wxyz), blend
        )
        desired = Matrix.Translation(position) @ rotation.to_matrix().to_4x4()
        rest = _pose_matrix(closed.poses[link.name])
        armature.pose.bones[f"mouth::{link.name}"].matrix_basis = (
            rest.inverted_safe() @ desired
        )


def _mouth_updated(self, _context):
    if self.type == "ARMATURE" and self.get("duck_robot_id") == "microduck-alpha":
        _apply_mouth_pose(self, self.duck_mouth_open)


def _leg_joint_specs(profile, side: str):
    by_name = {joint.name: joint for joint in profile.joints}
    names = tuple(
        f"{side}_{suffix}"
        for suffix in ("hip_roll", "hip_pitch", "knee", "ankle")
    )
    try:
        return tuple(by_name[name] for name in names)
    except KeyError as exc:
        raise MotionError(f"profile is missing complete {side} leg chain") from exc


def _physical_site_matrix(armature, profile, side: str) -> Matrix:
    site = next(
        (site for site in profile.sites if site.name == f"{side}_foot"),
        None,
    )
    if site is None:
        raise MotionError(f"profile is missing physical site {side}_foot")
    ankle = armature.pose.bones.get(site.parent_body)
    if ankle is None:
        raise MotionError(f"armature is missing ankle body {site.parent_body}")
    return ankle.matrix @ Matrix.Translation(site.position) @ Quaternion(
        site.quaternion_wxyz
    ).to_matrix().to_4x4()


def update_physical_ik(armature) -> None:
    """Solve both physical foot controls onto canonical local-Z hinges."""
    if float(armature.get("fk_ik", 0.0)) < 0.5:
        return
    bpy.context.view_layer.update()
    profile = profile_from_armature(armature)
    solutions = []
    for side in ("left", "right"):
        joints = _leg_joint_specs(profile, side)
        foot = armature.pose.bones.get(f"IK_FOOT_{side}")
        pole = armature.pose.bones.get(f"IK_POLE_{side}")
        hip_yaw = armature.pose.bones.get(joints[0].parent_body)
        if foot is None or pole is None or hip_yaw is None:
            raise MotionError(f"armature is missing {side} physical IK controls")
        model = leg_kinematics(profile, side)
        hip_inverse = hip_yaw.matrix.inverted_safe()
        target = hip_inverse @ foot.matrix.translation
        pole_local = hip_inverse @ pole.matrix.translation
        initial = np.asarray(
            [armature.pose.bones[joint.child_body].rotation_euler.z for joint in joints],
            dtype=np.float64,
        )
        result = solve_leg_ik(
            model,
            tuple(target),
            float(foot.get("duck_sagittal_pitch", 0.0)),
            initial_angles=initial,
            pole_sign=float(pole_local.y),
        )
        solutions.append((joints, foot, result))
    for joints, foot, result in solutions:
        for joint, angle in zip(joints, result.angles, strict=True):
            pose_bone = armature.pose.bones[joint.child_body]
            pose_bone.rotation_mode = "XYZ"
            pose_bone.location = (0.0, 0.0, 0.0)
            pose_bone.scale = (1.0, 1.0, 1.0)
            pose_bone.rotation_euler = (0.0, 0.0, float(angle))
        foot["duck_ik_clamped"] = bool(result.clamped)


def _clear_physical_ik_handlers() -> None:
    for handlers in (
        bpy.app.handlers.frame_change_post,
        bpy.app.handlers.depsgraph_update_post,
    ):
        for handler in tuple(handlers):
            if getattr(handler, "_duck_physical_ik_handler", False):
                handlers.remove(handler)


@bpy.app.handlers.persistent
def _physical_ik_update_handler(_scene, *_args) -> None:
    for armature in tuple(bpy.data.objects):
        if (
            armature.type != "ARMATURE"
            or armature.get("duck_robot_id") != "microduck-alpha"
        ):
            continue
        pointer = armature.as_pointer()
        if pointer in _IK_UPDATE_GUARD:
            continue
        _IK_UPDATE_GUARD.add(pointer)
        try:
            if hasattr(armature, "duck_mouth_open"):
                _apply_mouth_pose(armature, float(armature.duck_mouth_open))
            if float(armature.get("fk_ik", 0.0)) >= 0.5:
                update_physical_ik(armature)
        except (MotionError, ProfileError, ValueError):
            armature["duck_ik_update_error"] = True
        finally:
            _IK_UPDATE_GUARD.discard(pointer)


_physical_ik_update_handler._duck_physical_ik_handler = True


def _install_physical_ik_handlers() -> None:
    _clear_physical_ik_handlers()
    bpy.app.handlers.frame_change_post.append(_physical_ik_update_handler)
    bpy.app.handlers.depsgraph_update_post.append(_physical_ik_update_handler)


class DUCK_OT_switch_ik(bpy.types.Operator):
    bl_idname = "duck.switch_ik"
    bl_label = "Switch to IK"
    bl_description = "Move foot targets to the current feet and enable IK"

    def execute(self, context):
        armature = context.object
        profile = profile_from_armature(armature)
        original_flag = float(armature.get("fk_ik", 0.0))
        targets = {}
        try:
            for side in ("left", "right"):
                joints = _leg_joint_specs(profile, side)
                target = armature.pose.bones.get(f"IK_FOOT_{side}")
                if target is None:
                    raise MotionError(f"armature is missing {side} foot control")
                targets[side] = target.matrix.copy()
                angles = np.asarray(
                    [
                        armature.pose.bones[joint.child_body].rotation_euler.z
                        for joint in joints
                    ],
                    dtype=np.float64,
                )
                target.matrix = _physical_site_matrix(armature, profile, side)
                target["duck_sagittal_pitch"] = leg_kinematics(
                    profile, side
                ).forward(angles).pitch
            armature["fk_ik"] = 1.0
            context.view_layer.update()
            update_physical_ik(armature)
            context.view_layer.update()
            return {"FINISHED"}
        except (MotionError, ValueError) as exc:
            armature["fk_ik"] = original_flag
            for side, matrix in targets.items():
                armature.pose.bones[f"IK_FOOT_{side}"].matrix = matrix
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}


class DUCK_OT_switch_fk(bpy.types.Operator):
    bl_idname = "duck.switch_fk"
    bl_label = "Switch to FK"
    bl_description = "Bake the evaluated IK chain onto FK bones and disable IK"

    def execute(self, context):
        armature = context.object
        armature["fk_ik"] = 0.0
        context.view_layer.update()
        return {"FINISHED"}


def _default_motion_action_name(filepath: str) -> str:
    stem = Path(filepath).stem
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return sanitized or "ImportedMotion"


def _clear_play_once_handlers(scene=None) -> None:
    scene_pointer = scene.as_pointer() if scene is not None else None
    for handler in tuple(bpy.app.handlers.frame_change_post):
        if not getattr(handler, "_duck_play_once_handler", False):
            continue
        if scene_pointer is not None and getattr(
            handler, "_duck_scene_pointer", None
        ) != scene_pointer:
            continue
        bpy.app.handlers.frame_change_post.remove(handler)


def _clear_native_playback_handlers(scene=None) -> None:
    scene_pointer = scene.as_pointer() if scene is not None else None
    for handler in tuple(bpy.app.handlers.animation_playback_post):
        if not getattr(handler, "_duck_native_playback_handler", False):
            continue
        if scene_pointer is not None and getattr(
            handler, "_duck_scene_pointer", None
        ) != scene_pointer:
            continue
        bpy.app.handlers.animation_playback_post.remove(handler)
        target_scene = getattr(handler, "_duck_scene", None)
        previous_mode = getattr(handler, "_duck_previous_loop_mode", None)
        if (
            target_scene is not None
            and previous_mode is not None
            and hasattr(target_scene, "playback_loop_mode")
        ):
            target_scene.playback_loop_mode = previous_mode


def _stop_playback(context) -> None:
    screen = getattr(context, "screen", None)
    if screen is not None and screen.is_animation_playing:
        bpy.ops.screen.animation_cancel(restore_frame=False)
    scene = getattr(context, "scene", None)
    _clear_play_once_handlers(scene)
    _clear_native_playback_handlers(scene)


def _action_range(action) -> tuple[int, int]:
    if action.use_frame_range:
        start, end = action.frame_start, action.frame_end
    else:
        start, end = action.frame_range
    return math.floor(start), math.ceil(end)


def _set_action_scene_range(scene, action) -> tuple[int, int]:
    start, end = _action_range(action)
    scene.frame_start = start
    scene.frame_end = end
    scene.use_preview_range = False
    return start, end


def _is_loopable(action) -> bool:
    return bool(action.get("duck_loopable", True))


def _install_play_once_handler(scene) -> None:
    _clear_play_once_handlers(scene)
    scene_pointer = scene.as_pointer()
    end_frame = scene.frame_end

    def stop_at_end(changed_scene, *_args):
        if changed_scene.as_pointer() != scene_pointer:
            return
        if changed_scene.frame_current < end_frame:
            return
        screen = bpy.context.screen
        if screen is not None and screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        _clear_play_once_handlers(changed_scene)

    stop_at_end._duck_play_once_handler = True
    stop_at_end._duck_scene_pointer = scene_pointer
    bpy.app.handlers.frame_change_post.append(stop_at_end)


def _install_native_playback_cleanup(scene) -> None:
    _clear_native_playback_handlers(scene)
    scene_pointer = scene.as_pointer()
    previous_mode = scene.playback_loop_mode

    def restore_after_playback(changed_scene, *_args):
        if changed_scene.as_pointer() == scene_pointer:
            _clear_native_playback_handlers(changed_scene)

    restore_after_playback._duck_native_playback_handler = True
    restore_after_playback._duck_scene_pointer = scene_pointer
    restore_after_playback._duck_scene = scene
    restore_after_playback._duck_previous_loop_mode = previous_mode
    bpy.app.handlers.animation_playback_post.append(restore_after_playback)


def _configure_action_playback(scene, action) -> None:
    _clear_play_once_handlers(scene)
    if hasattr(scene, "playback_loop_mode"):
        _install_native_playback_cleanup(scene)
        scene.playback_loop_mode = (
            "INFINITE" if _is_loopable(action) else "STOP_END_FRAME"
        )
    elif not _is_loopable(action):
        _install_play_once_handler(scene)


def _activate_action(armature, action, scene) -> None:
    force_fk(armature)
    _stop_playback(bpy.context)
    reset_canonical_pose(armature, profile_from_armature(armature))
    armature.animation_data_create()
    armature.animation_data.action = action
    start, _end = _set_action_scene_range(scene, action)
    scene.frame_set(start)


def _action_name_get(armature) -> str:
    animation_data = armature.animation_data
    return animation_data.action.name if animation_data and animation_data.action else ""


def _action_name_set(armature, action_name: str) -> None:
    action = bpy.data.actions.get(action_name)
    if action is not None:
        _activate_action(armature, action, bpy.context.scene)


class DUCK_OT_select_action(bpy.types.Operator):
    bl_idname = "duck.select_action"
    bl_label = "Select Duck Action"
    bl_description = "Activate an animation and use its complete frame range"

    action_name: bpy.props.StringProperty()

    def execute(self, context):
        armature = context.object
        action = bpy.data.actions.get(self.action_name)
        if armature is None or armature.type != "ARMATURE" or action is None:
            self.report(
                {"WARNING"},
                f"Animation action is unavailable: {self.action_name}",
            )
            return {"CANCELLED"}
        armature.duck_action_name = action.name
        return {"FINISHED"}


class DUCK_OT_toggle_animation(bpy.types.Operator):
    bl_idname = "duck.toggle_animation"
    bl_label = "Play Once/Pause"
    bl_description = "Play a non-looping action once, or pause the active animation"

    def execute(self, context):
        armature = context.object
        if (
            armature is None
            or armature.animation_data is None
            or armature.animation_data.action is None
        ):
            self.report({"WARNING"}, "Select an animation action first")
            return {"CANCELLED"}
        if context.screen.is_animation_playing:
            _stop_playback(context)
        else:
            action = armature.animation_data.action
            force_fk(armature)
            reset_canonical_pose(armature, profile_from_armature(armature))
            start, end = _set_action_scene_range(context.scene, action)
            if context.scene.frame_current < start or context.scene.frame_current >= end:
                context.scene.frame_set(start)
            else:
                context.scene.frame_set(
                    context.scene.frame_current,
                    subframe=context.scene.frame_subframe,
                )
            _configure_action_playback(context.scene, action)
            bpy.ops.screen.animation_play()
        return {"FINISHED"}


class DUCK_OT_reset_animation(bpy.types.Operator):
    bl_idname = "duck.reset_animation"
    bl_label = "Reset"
    bl_description = "Pause and return to the first animation frame"

    def execute(self, context):
        armature = context.object
        action = (
            armature.animation_data.action
            if armature is not None and armature.animation_data is not None
            else None
        )
        if action is not None:
            force_fk(armature)
            reset_canonical_pose(armature, profile_from_armature(armature))
            start, _end = _set_action_scene_range(context.scene, action)
        else:
            start = context.scene.frame_start
        _stop_playback(context)
        context.scene.frame_set(start)
        return {"FINISHED"}


class DUCK_OT_import_motion(bpy.types.Operator, ImportHelper):
    bl_idname = "duck.import_motion"
    bl_label = "Import mjlab Motion"
    bl_description = "Import a native mjlab motion archive as a new action"
    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(default="*.npz", options={"HIDDEN"})
    action_name: bpy.props.StringProperty(name="Action Name", default="")

    def execute(self, context):
        armature = context.object
        if armature is None or armature.type != "ARMATURE":
            self.report({"WARNING"}, "Select a duck armature before importing motion")
            return {"CANCELLED"}
        try:
            profile = profile_from_armature(armature)
            action_name = self.action_name.strip() or _default_motion_action_name(self.filepath)
            action = import_motion_action(
                armature,
                profile,
                Path(self.filepath),
                action_name=action_name,
                before_mutation=lambda: _stop_playback(context),
            )
        except (MotionError, ProfileError, OSError, ValueError) as exc:
            _stop_playback(context)
            self.report({"WARNING"}, str(exc))
            return {"CANCELLED"}
        frame_count = context.scene.frame_end - context.scene.frame_start + 1
        self.report({"INFO"}, f"Imported {action.name} ({frame_count} frames)")
        return {"FINISHED"}


class DUCK_OT_export_motion(bpy.types.Operator, ExportHelper):
    bl_idname = "duck.export_motion"
    bl_label = "Export mjlab Motion"
    filename_ext = ".npz"
    filter_glob: bpy.props.StringProperty(default="*.npz", options={"HIDDEN"})

    def execute(self, context):
        armature = context.object
        try:
            profile = profile_from_armature(armature)
            archive = collect_armature_motion(
                armature,
                profile,
                context.scene.frame_start,
                context.scene.frame_end,
            )
            destination = save_motion_npz(Path(self.filepath), archive)
        except (MotionError, ProfileError, OSError, ValueError) as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        self.report({"INFO"}, f"Saved {destination}")
        return {"FINISHED"}


class DUCK_PT_tools(bpy.types.Panel):
    bl_label = "Open Duck Tools"
    bl_idname = "DUCK_PT_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Open Duck"

    @classmethod
    def poll(cls, context):
        return context.object is not None and context.object.type == "ARMATURE" and bool(
            context.object.get("duck_robot_id")
        )

    def draw(self, context):
        layout = self.layout
        armature = context.object
        layout.label(text=f"Robot: {armature.get('duck_robot_id')}")
        if armature.get("duck_robot_id") == "microduck-alpha":
            layout.prop(armature, "duck_colorway", text="Colourway")
            layout.prop(armature, "duck_mouth_open")
        layout.label(
            text="Mode: IK" if float(armature.get("fk_ik", 0.0)) >= 0.5 else "Mode: FK",
            icon="CON_KINEMATIC" if float(armature.get("fk_ik", 0.0)) >= 0.5 else "BONE_DATA",
        )
        row = layout.row(align=True)
        row.operator("duck.switch_fk")
        row.operator("duck.switch_ik")
        animation = layout.box()
        animation.label(text="Animation", icon="ACTION")
        animation.prop_search(
            armature,
            "duck_action_name",
            bpy.data,
            "actions",
            text="Action",
        )
        presets = animation.row(align=True)
        for action_name, label in BEGINNER_ACTION_PRESETS:
            if bpy.data.actions.get(action_name) is not None:
                operator = presets.operator("duck.select_action", text=label)
                operator.action_name = action_name
        active_action = armature.animation_data and armature.animation_data.action
        if active_action and not bool(active_action.get("duck_contact_valid", True)):
            animation.label(
                text="Kinematic test only — ground contact is not guaranteed",
                icon="ERROR",
            )
        controls = animation.row(align=True)
        controls.enabled = bool(active_action)
        controls.operator(
            "duck.toggle_animation",
            text=(
                "Pause"
                if context.screen.is_animation_playing
                else (
                    "Play"
                    if active_action and _is_loopable(active_action)
                    else "Play Once"
                )
            ),
            icon="PAUSE" if context.screen.is_animation_playing else "PLAY",
        )
        controls.operator("duck.reset_animation", text="Reset", icon="LOOP_BACK")
        if active_action:
            animation.label(
                text=(
                    f"Range {context.scene.frame_start}"
                    f"–{context.scene.frame_end}"
                )
            )
        else:
            animation.label(text="No action selected", icon="INFO")
        layout.operator("duck.import_motion", icon="IMPORT")
        layout.operator("duck.export_motion", icon="EXPORT")


CLASSES = (
    DUCK_OT_switch_ik,
    DUCK_OT_switch_fk,
    DUCK_OT_select_action,
    DUCK_OT_toggle_animation,
    DUCK_OT_reset_animation,
    DUCK_OT_import_motion,
    DUCK_OT_export_motion,
    DUCK_PT_tools,
)


def register():
    _clear_play_once_handlers()
    _clear_native_playback_handlers()
    _install_physical_ik_handlers()
    for cls in CLASSES:
        if not hasattr(bpy.types, cls.__name__):
            bpy.utils.register_class(cls)
    if not hasattr(bpy.types.Object, "duck_colorway"):
        bpy.types.Object.duck_colorway = bpy.props.EnumProperty(
            name="Microduck Colourway",
            items=[(key, values[0], "") for key, values in COLORWAYS.items()],
            default="CREAM",
            update=_colorway_updated,
        )
    if not hasattr(bpy.types.Object, "duck_mouth_open"):
        bpy.types.Object.duck_mouth_open = bpy.props.FloatProperty(
            name="Mouth (visual approximation)",
            min=0.0,
            max=1.0,
            default=0.0,
            update=_mouth_updated,
        )
    if not hasattr(bpy.types.Object, "duck_action_name"):
        bpy.types.Object.duck_action_name = bpy.props.StringProperty(
            name="Animation Action",
            get=_action_name_get,
            set=_action_name_set,
        )


def unregister():
    _stop_playback(bpy.context)
    _clear_play_once_handlers()
    _clear_native_playback_handlers()
    _clear_physical_ik_handlers()
    _IK_UPDATE_GUARD.clear()
    if hasattr(bpy.types.Object, "duck_action_name"):
        del bpy.types.Object.duck_action_name
    if hasattr(bpy.types.Object, "duck_mouth_open"):
        del bpy.types.Object.duck_mouth_open
    if hasattr(bpy.types.Object, "duck_colorway"):
        del bpy.types.Object.duck_colorway
    for cls in reversed(CLASSES):
        if hasattr(bpy.types, cls.__name__):
            bpy.utils.unregister_class(cls)
