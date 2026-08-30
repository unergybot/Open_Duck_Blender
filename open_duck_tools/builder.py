"""Deterministically generate the Microduck Blender scene from MJCF."""

from __future__ import annotations

from pathlib import Path
import struct
import types
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Quaternion, Vector
import numpy as np

from .profile import ProfileError, profile_to_json


def _transform(position, quaternion) -> Matrix:
    return Matrix.Translation(position) @ Quaternion(quaternion).to_matrix().to_4x4()


def _world_rest_matrices(profile) -> dict[str, Matrix]:
    result = {}
    for body in profile.bodies:
        local = _transform(body.position, body.quaternion_wxyz)
        result[body.name] = local if body.parent is None else result[body.parent] @ local
    return result


def _binary_stl(path: Path, name: str):
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ProfileError(f"{path}: truncated binary STL")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ProfileError(f"{path}: expected binary STL data")
    vertices = []
    faces = []
    offset = 84
    for _ in range(count):
        triangle = struct.unpack_from("<12fH", raw, offset)
        base = len(vertices)
        vertices.extend((triangle[3:6], triangle[6:9], triangle[9:12]))
        faces.append((base, base + 1, base + 2))
        offset += 50
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    return mesh


def _material(name: str, rgba: tuple[float, float, float, float]):
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.diffuse_color = rgba
    material.use_nodes = True
    node = material.node_tree.nodes.get("Principled BSDF")
    if node is not None:
        node.inputs["Base Color"].default_value = rgba
        node.inputs["Roughness"].default_value = 0.55
    lowered = name.lower()
    if "bottom_head_shell" in lowered or any(
        token in lowered for token in ("jaw", "foot", "ankle", "trim")
    ):
        material["duck_material_role"] = "trim"
    elif "shell" in lowered:
        material["duck_material_role"] = "shell"
    return material


def _embed_addon(addon_source_root: Path) -> None:
    module_names = ("profile", "motion", "blender_bridge", "addon")
    for name in module_names:
        path = addon_source_root / f"{name}.py"
        if not path.exists():
            raise ProfileError(f"add-on source is missing {path}")
        text = bpy.data.texts.get(f"open_duck_tools.{name}") or bpy.data.texts.new(
            f"open_duck_tools.{name}"
        )
        text.clear()
        text.write(path.read_text())
    bootstrap_source = '''import bpy, sys, types

MODULES = ("profile", "motion", "blender_bridge", "addon")
PACKAGE = "open_duck_tools_embedded"

def register():
    package = sys.modules.get(PACKAGE)
    if package is None:
        package = types.ModuleType(PACKAGE)
        package.__path__ = []
        sys.modules[PACKAGE] = package
    for short_name in MODULES:
        full_name = f"{PACKAGE}.{short_name}"
        module = types.ModuleType(full_name)
        module.__file__ = f"<blender-text:{short_name}>"
        module.__package__ = PACKAGE
        sys.modules[full_name] = module
        source = bpy.data.texts[f"open_duck_tools.{short_name}"].as_string()
        exec(compile(source, module.__file__, "exec"), module.__dict__)
    sys.modules[f"{PACKAGE}.addon"].register()

def unregister():
    module = sys.modules.get(f"{PACKAGE}.addon")
    if module is not None:
        module.unregister()

register()
'''
    bootstrap = bpy.data.texts.get("open_duck_bootstrap.py") or bpy.data.texts.new(
        "open_duck_bootstrap.py"
    )
    bootstrap.clear()
    bootstrap.write(bootstrap_source)
    bootstrap.use_module = True


def _create_armature(profile, world_rest: dict[str, Matrix]):
    data = bpy.data.armatures.new("MicroduckRig")
    armature = bpy.data.objects.new("MicroduckRig", data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    body_bones = {}
    for body in profile.bodies:
        bone = data.edit_bones.new(body.name)
        bone.matrix = world_rest[body.name]
        bone.length = 0.015
        if body.parent is not None:
            bone.parent = body_bones[body.parent]
            bone.use_connect = False
        body_bones[body.name] = bone
    for link in profile.mouth.links:
        parent = body_bones.get(link.parent)
        if parent is None:
            raise ProfileError(f"mouth link {link.name!r} has unknown parent {link.parent!r}")
        pose = profile.mouth.samples[0].poses[link.name]
        bone = data.edit_bones.new(f"mouth::{link.name}")
        bone.matrix = world_rest[link.parent] @ _transform(
            pose.position, pose.quaternion_wxyz
        )
        bone.length = 0.01
        bone.parent = parent
        bone.use_connect = False
    bpy.ops.object.mode_set(mode="POSE")
    home = dict(zip(profile.joint_names, profile.home_positions))
    joint_by_name = {joint.name: joint for joint in profile.joints}
    for name, angle in home.items():
        joint = joint_by_name[name]
        pose_bone = armature.pose.bones[joint.child_body]
        pose_bone.rotation_mode = "XYZ"
        pose_bone.rotation_euler = (0.0, 0.0, angle)
        pose_bone.lock_rotation = (True, True, False)
        pose_bone["duck_joint_name"] = name
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature


def _create_ik(profile, armature, world_rest):
    by_joint = {joint.name: joint.child_body for joint in profile.joints}
    for side in ("left", "right"):
        names = [
            f"{side}_hip_roll",
            f"{side}_hip_pitch",
            f"{side}_knee",
            f"{side}_ankle",
        ]
        if any(name not in by_joint for name in names):
            continue
        ankle_body = by_joint[names[-1]]
        target = bpy.data.objects.new(f"IK_FOOT_{side}", None)
        target.empty_display_type = "CUBE"
        target.empty_display_size = 0.025
        target.matrix_world = world_rest[ankle_body]
        bpy.context.scene.collection.objects.link(target)
        ankle = armature.pose.bones[ankle_body]
        ik = ankle.constraints.new("IK")
        ik.name = "DUCK_IK"
        ik.target = target
        ik.chain_count = 4
        ik.influence = 0.0
        rotation = ankle.constraints.new("COPY_ROTATION")
        rotation.name = "DUCK_IK_ROTATION"
        rotation.target = target
        rotation.target_space = "WORLD"
        rotation.owner_space = "WORLD"
        rotation.influence = 0.0


def _create_visuals(profile, mjcf_path: Path, armature, world_rest):
    root = ET.parse(mjcf_path).getroot()
    compiler = root.find("compiler")
    mesh_dir = mjcf_path.parent / (compiler.get("meshdir", ".") if compiler is not None else ".")
    asset = root.find("asset")
    mesh_files = {}
    materials = {}
    if asset is not None:
        for mesh in asset.findall("mesh"):
            filename = mesh.get("file") or f"{mesh.get('name')}.stl"
            mesh_files[mesh.get("name") or Path(filename).stem] = mesh_dir / filename
        for item in asset.findall("material"):
            rgba = tuple(float(component) for component in item.get("rgba", "0.5 0.5 0.5 1").split())
            materials[item.get("name")] = _material(item.get("name"), rgba)
    mouth_meshes = {
        mesh: f"mouth::{link.name}" for link in profile.mouth.links for mesh in link.meshes
    }
    used_mouth_meshes = set()
    mesh_data = {}

    def walk(body_element, parent_world: Matrix):
        name = body_element.get("name")
        body_world = world_rest[name]
        for index, geom in enumerate(body_element.findall("geom")):
            mesh_name = geom.get("mesh")
            if not mesh_name or geom.get("class") not in (None, "visual"):
                continue
            if mesh_name not in mesh_files:
                raise ProfileError(f"MJCF geom references unknown mesh {mesh_name!r}")
            if mesh_name not in mesh_data:
                mesh_data[mesh_name] = _binary_stl(mesh_files[mesh_name], f"mesh::{mesh_name}")
            object_name = f"visual::{mesh_name}"
            if bpy.data.objects.get(object_name) is not None:
                object_name = f"{object_name}::{name}::{index}"
            obj = bpy.data.objects.new(object_name, mesh_data[mesh_name])
            bpy.context.scene.collection.objects.link(obj)
            material = materials.get(geom.get("material"))
            if material is not None:
                obj.data.materials.append(material)
            local = _transform(
                tuple(float(value) for value in geom.get("pos", "0 0 0").split()),
                tuple(float(value) for value in geom.get("quat", "1 0 0 0").split()),
            )
            obj.matrix_world = body_world @ local
            bone_name = mouth_meshes.get(mesh_name, name)
            if mesh_name in mouth_meshes:
                used_mouth_meshes.add(mesh_name)
            obj.parent = armature
            obj.parent_type = "BONE"
            obj.parent_bone = bone_name
            obj.matrix_parent_inverse = (
                armature.matrix_world @ armature.data.bones[bone_name].matrix_local
            ).inverted_safe()
        for child in body_element.findall("body"):
            walk(child, body_world)

    worldbody = root.find("worldbody")
    for top in worldbody.findall("body"):
        walk(top, Matrix.Identity(4))
    missing = set(mouth_meshes) - used_mouth_meshes
    if missing:
        raise ProfileError(f"mouth linkage references meshes absent from visual MJCF: {sorted(missing)}")


def _create_demo_action(armature, profile, motion_path: Path) -> None:
    with np.load(motion_path, allow_pickle=False) as archive:
        fps = np.asarray(archive["fps"])
        joint_names = tuple(str(name) for name in archive["joint_names"])
        joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
    if fps.shape != (1,) or int(fps[0]) != 50:
        raise ProfileError("demo motion must declare fps=[50]")
    if joint_names != tuple(profile.joint_names):
        raise ProfileError("demo motion joint_names do not match the robot profile")
    expected_shape = (joint_pos.shape[0], len(profile.joint_names))
    if joint_pos.ndim != 2 or joint_pos.shape != expected_shape or not joint_pos.shape[0]:
        raise ProfileError("demo motion joint_pos must have shape [T, joint_count]")
    if not np.isfinite(joint_pos).all():
        raise ProfileError("demo motion joint_pos contains a non-finite value")

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = joint_pos.shape[0]
    joint_by_name = {joint.name: joint for joint in profile.joints}
    frame_count = joint_pos.shape[0]
    for index, positions in enumerate(joint_pos):
        frame = index + 1
        for joint_name, angle in zip(profile.joint_names, positions):
            pose_bone = armature.pose.bones[joint_by_name[joint_name].child_body]
            pose_bone.rotation_euler.z = float(angle)
            pose_bone.keyframe_insert(
                data_path="rotation_euler",
                index=2,
                frame=frame,
                group=joint_name,
            )
        progress = 0.0 if frame_count == 1 else index / (frame_count - 1)
        armature.duck_mouth_open = 1.0 - abs(2.0 * progress - 1.0)
        armature.keyframe_insert(data_path="duck_mouth_open", frame=frame)
    armature.animation_data.action.name = "MicroduckCrouchTest"
    scene.frame_set(scene.frame_start)


def generate_microduck_scene(
    profile,
    mjcf_path,
    addon_source_root,
    *,
    demo_motion_path: Path | None = None,
):
    """Replace the current scene with a profiled, animatable Microduck."""
    mjcf_path = Path(mjcf_path)
    addon_source_root = Path(addon_source_root)
    for collection in (bpy.data.objects, bpy.data.armatures, bpy.data.meshes, bpy.data.materials):
        for item in list(collection):
            collection.remove(item, do_unlink=True)
    scene = bpy.context.scene
    scene.render.fps = 50
    scene.frame_start = 1
    scene.frame_end = 100
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    world_rest = _world_rest_matrices(profile)
    armature = _create_armature(profile, world_rest)
    armature["duck_robot_id"] = profile.robot_id
    armature.data["duck_robot_profile_json"] = profile_to_json(profile)
    armature["fk_ik"] = 0.0
    from . import addon

    addon.register()
    armature.duck_mouth_open = 0.0
    armature.duck_colorway = "CREAM"
    _create_ik(profile, armature, world_rest)
    _create_visuals(profile, mjcf_path, armature, world_rest)
    _embed_addon(addon_source_root)
    if demo_motion_path is not None:
        _create_demo_action(armature, profile, Path(demo_motion_path))
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    return armature
