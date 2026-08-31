#!/usr/bin/env python3
"""Headless acceptance check for the self-contained Microduck release blend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import bpy
from mathutils import Matrix, Quaternion
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_duck_tools import addon
from open_duck_tools.addon import collect_armature_motion, profile_from_armature
from open_duck_tools.blender_bridge import force_fk, reset_canonical_pose
from open_duck_tools.motion_import import load_motion
from open_duck_tools.profile import build_microduck_profile, profile_to_json


MJCF = Path.home() / "MyCode/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml"
RUNTIME = Path.home() / "MyCode/microduck/duck-control/src/model.rs"
CONTRACT = Path.home() / "MyCode/microduck/duck-ipc-proto/src/lib.rs"


def _matrix_error(actual, expected) -> float:
    return max(
        abs(actual[row][column] - expected[row][column])
        for row in range(4)
        for column in range(4)
    )


def _canonical_visual_matrices(profile, armature, mouth_poses=None):
    root = ET.parse(MJCF).getroot()
    expected = {}

    def walk(body):
        body_name = body.get("name")
        for geom in body.findall("geom"):
            mesh = geom.get("mesh")
            if mesh and geom.get("class") in (None, "visual"):
                position = tuple(float(value) for value in geom.get("pos", "0 0 0").split())
                quaternion = tuple(float(value) for value in geom.get("quat", "1 0 0 0").split())
                norm = np.linalg.norm(quaternion)
                local = Matrix.Translation(position) @ Quaternion(
                    tuple(value / norm for value in quaternion)
                ).to_matrix().to_4x4()
                if mouth_poses is not None and mesh in mouth_poses:
                    closed, desired = mouth_poses[mesh]
                    local = desired @ closed.inverted_safe() @ local
                expected.setdefault(mesh, []).append(
                    armature.matrix_world @ armature.pose.bones[body_name].matrix @ local
                )
        for child in body.findall("body"):
            walk(child)

    for body in root.find("worldbody").findall("body"):
        walk(body)
    return expected


def _check_visual_matrices(profile, armature, visuals, mouth_poses=None) -> None:
    expected_visuals = _canonical_visual_matrices(profile, armature, mouth_poses)
    for obj in visuals:
        mesh_name = obj.data.name.removeprefix("mesh::").split("::", 1)[0]
        candidates = expected_visuals.get(mesh_name, [])
        if not candidates:
            raise AssertionError(f"unexpected visual mesh {mesh_name}")
        match = min(
            range(len(candidates)),
            key=lambda index: _matrix_error(obj.matrix_world, candidates[index]),
        )
        if _matrix_error(obj.matrix_world, candidates[match]) > 1e-5:
            raise AssertionError(f"visual matrix differs for {obj.name}")
        candidates.pop(match)
    if any(expected_visuals.values()):
        raise AssertionError("canonical visual instances are missing")


def _mouth_visual_poses(profile, open_fraction: float):
    linkage = profile.mouth
    servo = linkage.closed_rad + float(open_fraction) * (
        linkage.open_rad - linkage.closed_rad
    )
    upper = next(
        (index for index, sample in enumerate(linkage.samples) if sample.servo_rad >= servo),
        len(linkage.samples) - 1,
    )
    lower = max(0, upper - 1)
    first, second = linkage.samples[lower], linkage.samples[upper]
    span = second.servo_rad - first.servo_rad
    blend = 0.0 if abs(span) < 1e-12 else (servo - first.servo_rad) / span
    result = {}
    for link in linkage.links:
        closed_pose = linkage.samples[0].poses[link.name]
        first_pose, second_pose = first.poses[link.name], second.poses[link.name]
        position = tuple(
            a + blend * (b - a)
            for a, b in zip(first_pose.position, second_pose.position)
        )
        rotation = Quaternion(first_pose.quaternion_wxyz).slerp(
            Quaternion(second_pose.quaternion_wxyz), blend
        )
        closed = Matrix.Translation(closed_pose.position) @ Quaternion(
            closed_pose.quaternion_wxyz
        ).to_matrix().to_4x4()
        desired = Matrix.Translation(position) @ rotation.to_matrix().to_4x4()
        for mesh in link.meshes:
            result[mesh] = (closed, desired)
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_release() -> dict[str, int]:
    scene = bpy.context.scene
    armature = bpy.data.objects.get("MicroduckRig")
    if armature is None or armature.type != "ARMATURE":
        raise AssertionError("release is missing MicroduckRig armature")
    profile = profile_from_armature(armature)
    canonical_profile = build_microduck_profile(
        MJCF, RUNTIME, None, joint_contract_path=CONTRACT
    )
    if profile_to_json(profile) != profile_to_json(canonical_profile):
        raise AssertionError("embedded profile differs from canonical sources")
    if len(profile.body_names) != 15 or len(profile.joint_names) != 14:
        raise AssertionError("release does not contain the canonical 15-body/14-joint rig")
    if {collection.name for collection in scene.collection.children} != {
        "Microduck",
        "Presentation",
    }:
        raise AssertionError("release top-level collection hierarchy is invalid")
    microduck = bpy.data.collections["Microduck"]
    if {collection.name for collection in microduck.children} != {
        "Rig",
        "Visuals",
        "Controls",
    }:
        raise AssertionError("Microduck child collections are invalid")
    if {obj.name for obj in bpy.data.collections["Presentation"].objects} != {
        "Ground",
        "MicroduckCamera",
        "KeyLight",
        "FillLight",
    }:
        raise AssertionError("presentation collection is incomplete")
    visuals = tuple(bpy.data.collections["Visuals"].objects)
    if len(visuals) != 70 or any(not obj.name.startswith("visual::") for obj in visuals):
        raise AssertionError("release must contain exactly 70 canonical visuals")
    if any(len(obj.data.materials) > 1 for obj in visuals):
        raise AssertionError("a canonical visual has multiple material slots")
    world_rest = {}
    for body in profile.bodies:
        local = Matrix.Translation(body.position) @ Quaternion(
            body.quaternion_wxyz
        ).to_matrix().to_4x4()
        world_rest[body.name] = (
            local if body.parent is None else world_rest[body.parent] @ local
        )
        if _matrix_error(armature.data.bones[body.name].matrix_local, world_rest[body.name]) > 1e-6:
            raise AssertionError(f"rest matrix differs for {body.name}")
    _check_visual_matrices(profile, armature, visuals)
    expected_texts = {
        "profile",
        "motion",
        "blender_bridge",
        "motion_import",
        "ik",
        "addon",
    }
    text_names = {text.name for text in bpy.data.texts}
    missing_texts = {
        f"open_duck_tools.{name}" for name in expected_texts
    } - text_names
    if missing_texts:
        raise AssertionError(f"release is missing embedded modules: {sorted(missing_texts)}")
    manifest = json.loads(bpy.data.texts["microduck-build-manifest.json"].as_string())
    if manifest.get("schema_version") != 2:
        raise AssertionError("release manifest is not schema 2")
    if manifest.get("source_sha256") != profile.source_sha256:
        raise AssertionError("release manifest/profile source hashes differ")
    if not str(manifest.get("build_blender_version", "")).startswith("4.3.2"):
        raise AssertionError("release was not built by Blender 4.3.2")
    motion_paths = {
        "KinematicCrouchTest": REPO_ROOT / "assets/motions/microduck-crouch-test.npz",
        "Policy_alpha_walking_forward": REPO_ROOT / "assets/motions/alpha-walking-forward.npz",
    }
    expected_frames = {"KinematicCrouchTest": 51, "Policy_alpha_walking_forward": 200}
    original_action = armature.animation_data.action
    original_frame = scene.frame_current
    addon.register()
    try:
        for name, path in motion_paths.items():
            action = bpy.data.actions.get(name)
            if action is None:
                raise AssertionError(f"release is missing action {name}")
            digest = _sha256(path)
            if manifest["motion_sha256"].get(name) != digest:
                raise AssertionError(f"manifest hash differs for {name}")
            if action.get("duck_source_sha256") != digest:
                raise AssertionError(f"action source hash differs for {name}")
            expected_kind = "kinematic_test" if name == "KinematicCrouchTest" else "policy_rollout"
            if action.get("duck_motion_kind") != expected_kind or bool(action.get("duck_loopable", True)):
                raise AssertionError(f"action metadata differs for {name}")
            if name == "KinematicCrouchTest" and bool(action.get("duck_contact_valid", True)):
                raise AssertionError("crouch action must be marked non-contact-valid")
            armature.animation_data.action = action
            force_fk(armature)
            reset_canonical_pose(armature, profile)
            frame_count = expected_frames[name]
            archive = collect_armature_motion(armature, profile, 1, frame_count)
            authoritative = load_motion(path, profile)
            comparisons = (
                ("joint_pos", archive["joint_pos"], authoritative.joint_pos, 1e-5),
                ("body_pos_w", archive["body_pos_w"], authoritative.body_pos_w, 1e-5),
            )
            for field, actual, expected, tolerance in comparisons:
                if actual.shape != expected.shape or np.max(np.abs(actual - expected)) > tolerance:
                    raise AssertionError(f"{name} differs from authoritative {field}")
            actual_quat = archive["body_quat_w"].astype(np.float64)
            expected_quat = authoritative.body_quat_w.astype(np.float64)
            actual_quat /= np.linalg.norm(actual_quat, axis=-1, keepdims=True)
            expected_quat /= np.linalg.norm(expected_quat, axis=-1, keepdims=True)
            dots = np.abs(np.sum(actual_quat * expected_quat, axis=-1))
            if np.max(2.0 * np.arccos(np.clip(dots, -1.0, 1.0))) > 1e-5:
                raise AssertionError(f"{name} differs from authoritative body_quat_w")
            for frame in range(1, frame_count + 1):
                scene.frame_set(frame)
                _check_visual_matrices(
                    profile,
                    armature,
                    visuals,
                    _mouth_visual_poses(profile, armature.duck_mouth_open),
                )
        scene.frame_set(1)
        mouth_bone_name = f"mouth::{profile.mouth.links[0].name}"
        armature.duck_mouth_open = 0.0
        bpy.context.view_layer.update()
        closed = tuple(
            tuple(row) for row in armature.pose.bones[mouth_bone_name].matrix_basis
        )
        _check_visual_matrices(
            profile, armature, visuals, _mouth_visual_poses(profile, 0.0)
        )
        armature.duck_mouth_open = 1.0
        bpy.context.view_layer.update()
        opened = tuple(
            tuple(row) for row in armature.pose.bones[mouth_bone_name].matrix_basis
        )
        _check_visual_matrices(
            profile, armature, visuals, _mouth_visual_poses(profile, 1.0)
        )
        if closed == opened:
            raise AssertionError("mouth control does not change the mouth helper pose")
        for key, (_label, shell_hex, trim_hex) in addon.COLORWAYS.items():
            addon._apply_colorway(armature, key)
            expected_colors = {
                "shell": addon._hex_rgba(shell_hex),
                "trim": addon._hex_rgba(trim_hex),
            }
            for material in bpy.data.materials:
                role = material.get("duck_material_role")
                if role in expected_colors and max(
                    abs(a - b) for a, b in zip(material.diffuse_color, expected_colors[role])
                ) > 1e-6:
                    raise AssertionError(f"{key} colorway differs for {material.name}")
        addon._apply_colorway(armature, "CREAM")
    finally:
        armature.duck_mouth_open = 0.0
        armature.animation_data.action = original_action
        scene.frame_set(original_frame)
        force_fk(armature)
        addon.unregister()
    if bpy.data.libraries:
        raise AssertionError("release contains linked external libraries")
    view_spaces = [
        area.spaces.active
        for screen in bpy.data.screens
        for area in screen.areas
        if area.type == "VIEW_3D"
    ]
    if not view_spaces or any(not space.show_region_ui for space in view_spaces):
        raise AssertionError("release sidebar is not open")
    if any(space.shading.type != "MATERIAL" for space in view_spaces):
        raise AssertionError("release viewport is not material-visible")
    bootstrap = bpy.data.texts["open_duck_bootstrap.py"]
    namespace = {}
    exec(compile(bootstrap.as_string(), "<release-bootstrap>", "exec"), namespace)
    embedded_addon = sys.modules.get("open_duck_tools_embedded.addon")
    if embedded_addon is None or not hasattr(bpy.types, "DUCK_PT_tools"):
        raise AssertionError("embedded bootstrap did not register the add-on")
    embedded_addon.unregister()
    return {
        "walk_frames": 200,
        "crouch_frames": 51,
        "canonical_bodies": len(profile.body_names),
        "canonical_visuals": len(visuals),
        "external_libraries": len(bpy.data.libraries),
        "viewports": len(view_spaces),
        "open_sidebars": sum(space.show_region_ui for space in view_spaces),
        "material_viewports": sum(space.shading.type == "MATERIAL" for space in view_spaces),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blend", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()), load_ui=True)
    print(json.dumps(check_release(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
