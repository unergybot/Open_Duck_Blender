"""Blender UI and operators shared by generated duck projects."""

from __future__ import annotations

from pathlib import Path

import bpy
from bpy_extras.io_utils import ExportHelper
from mathutils import Matrix, Quaternion
import numpy as np

from .blender_bridge import body_samples, joint_angles_from_body_matrices
from .motion import MotionError, build_motion_archive, save_motion_npz
from .profile import ProfileError, profile_from_json


COLORWAYS = {
    "CREAM": ("Cream", "#f7e6cb", "#f59e0b"),
    "GRAPHITE": ("Graphite", "#6c6a68", "#f7c948"),
    "LAVENDER": ("Lavender", "#bfa9cf", "#f7c948"),
    "SKY": ("Sky", "#a9dbe8", "#f59e0b"),
}


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


def collect_armature_motion(
    armature: bpy.types.Object,
    profile,
    frame_start: int,
    frame_end: int,
) -> dict[str, np.ndarray]:
    if frame_end < frame_start:
        raise MotionError("export frame end precedes frame start")
    scene = bpy.context.scene
    original_frame = scene.frame_current
    joint_frames = []
    body_positions = []
    body_quaternions = []
    try:
        for frame in range(frame_start, frame_end + 1):
            scene.frame_set(frame)
            matrices = _evaluated_body_matrices(armature, profile.body_names)
            joint_frames.append(joint_angles_from_body_matrices(matrices, profile))
            positions, quaternions = body_samples(matrices, profile)
            body_positions.append(positions)
            body_quaternions.append(quaternions)
    finally:
        scene.frame_set(original_frame)
    return build_motion_archive(
        np.asarray(joint_frames),
        np.asarray(body_positions),
        np.asarray(body_quaternions),
        fps=scene.render.fps,
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
    armature["duck_colorway"] = key


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


def _leg_body_chain(profile, side: str) -> list[str]:
    by_joint = {joint.name: joint.child_body for joint in profile.joints}
    names = [
        f"{side}_hip_roll",
        f"{side}_hip_pitch",
        f"{side}_knee",
        f"{side}_ankle",
    ]
    return [by_joint[name] for name in names if name in by_joint]


class DUCK_OT_switch_ik(bpy.types.Operator):
    bl_idname = "duck.switch_ik"
    bl_label = "Switch to IK"
    bl_description = "Move foot targets to the current feet and enable IK"

    def execute(self, context):
        armature = context.object
        profile = profile_from_armature(armature)
        for side in ("left", "right"):
            chain = _leg_body_chain(profile, side)
            ankle = armature.pose.bones.get(chain[-1]) if len(chain) == 4 else None
            target = bpy.data.objects.get(f"IK_FOOT_{side}")
            if ankle is None or target is None:
                self.report({"ERROR"}, f"Missing {side} IK controls")
                return {"CANCELLED"}
            target.matrix_world = armature.matrix_world @ ankle.matrix
            for constraint in ankle.constraints:
                if constraint.name.startswith("DUCK_IK"):
                    constraint.influence = 1.0
        armature["fk_ik"] = 1.0
        context.view_layer.update()
        return {"FINISHED"}


class DUCK_OT_switch_fk(bpy.types.Operator):
    bl_idname = "duck.switch_fk"
    bl_label = "Switch to FK"
    bl_description = "Bake the evaluated IK chain onto FK bones and disable IK"

    def execute(self, context):
        armature = context.object
        profile = profile_from_armature(armature)
        chains = {side: _leg_body_chain(profile, side) for side in ("left", "right")}
        if any(len(chain) != 4 for chain in chains.values()):
            self.report({"ERROR"}, "Profile is missing a complete leg chain")
            return {"CANCELLED"}
        evaluated = armature.evaluated_get(context.evaluated_depsgraph_get())
        solved = {
            name: evaluated.pose.bones[name].matrix.copy()
            for chain in chains.values()
            for name in chain
        }
        for side, chain in chains.items():
            ankle = armature.pose.bones.get(chain[-1])
            if ankle is None:
                self.report({"ERROR"}, f"Missing {side} FK controls")
                return {"CANCELLED"}
            for constraint in ankle.constraints:
                if constraint.name.startswith("DUCK_IK"):
                    constraint.influence = 0.0
            context.view_layer.update()
            for name in chain:
                armature.pose.bones[name].matrix = solved[name]
                context.view_layer.update()
        armature["fk_ik"] = 0.0
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
            layout.prop(armature, "duck_mouth_open", text="Mouth")
        row = layout.row(align=True)
        row.operator("duck.switch_fk")
        row.operator("duck.switch_ik")
        layout.operator("duck.export_motion", icon="EXPORT")


CLASSES = (DUCK_OT_switch_ik, DUCK_OT_switch_fk, DUCK_OT_export_motion, DUCK_PT_tools)


def register():
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
            name="Mouth",
            min=0.0,
            max=1.0,
            default=0.0,
            update=_mouth_updated,
        )


def unregister():
    if hasattr(bpy.types.Object, "duck_mouth_open"):
        del bpy.types.Object.duck_mouth_open
    if hasattr(bpy.types.Object, "duck_colorway"):
        del bpy.types.Object.duck_colorway
    for cls in reversed(CLASSES):
        if hasattr(bpy.types, cls.__name__):
            bpy.utils.unregister_class(cls)
