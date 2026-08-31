#!/usr/bin/env python3
"""Build the generated Microduck Blender project inside Blender."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_duck_tools.builder import generate_microduck_scene
from open_duck_tools.motion import MotionError
from open_duck_tools.motion_import import import_motion_action
from open_duck_tools.profile import ProfileError, build_microduck_profile


def _new_collection(name: str, parent) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if collection.name not in {child.name for child in parent.children}:
        parent.children.link(collection)
    return collection


def _move_object(obj, collection) -> None:
    for current in tuple(obj.users_collection):
        current.objects.unlink(obj)
    collection.objects.link(obj)


def _look_at(obj, target) -> None:
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat(
        "-Z", "Y"
    ).to_euler()


def _organize_release_scene(armature, policy_motion: Path) -> None:
    scene = bpy.context.scene
    root = scene.collection
    for child in tuple(root.children):
        root.children.unlink(child)
    microduck = _new_collection("Microduck", root)
    rig = _new_collection("Rig", microduck)
    visuals = _new_collection("Visuals", microduck)
    _new_collection("Controls", microduck)
    presentation = _new_collection("Presentation", root)
    _move_object(armature, rig)
    for obj in tuple(bpy.data.objects):
        if obj.name.startswith("visual::"):
            _move_object(obj, visuals)

    with np.load(policy_motion, allow_pickle=False) as archive:
        positions = np.asarray(archive["body_pos_w"], dtype=np.float64)
    minimum = positions.min(axis=(0, 1))
    maximum = positions.max(axis=(0, 1))
    center = 0.5 * (minimum + maximum)
    x_min, x_max = minimum[0] - 0.075, maximum[0] + 0.075
    y_center = center[1]
    y_half = max(0.15, 0.5 * (maximum[1] - minimum[1]) + 0.075)
    mesh = bpy.data.meshes.new("GroundMesh")
    mesh.from_pydata(
        (
            (x_min, y_center - y_half, 0.0),
            (x_max, y_center - y_half, 0.0),
            (x_max, y_center + y_half, 0.0),
            (x_min, y_center + y_half, 0.0),
        ),
        (),
        ((0, 1, 2, 3),),
    )
    ground = bpy.data.objects.new("Ground", mesh)
    presentation.objects.link(ground)

    target = Vector((center[0], center[1], max(0.12, center[2])))
    camera_data = bpy.data.cameras.new("MicroduckCamera")
    camera_data.lens = 50.0
    camera = bpy.data.objects.new("MicroduckCamera", camera_data)
    camera.location = target + Vector((-0.82, -0.99, 0.66))
    _look_at(camera, target)
    presentation.objects.link(camera)
    scene.camera = camera
    for name, offset, energy, size in (
        ("KeyLight", (-0.35, -0.55, 0.85), 700.0, 0.45),
        ("FillLight", (0.65, 0.30, 0.50), 350.0, 0.35),
    ):
        light_data = bpy.data.lights.new(name, "AREA")
        light_data.energy = energy
        light_data.shape = "DISK"
        light_data.size = size
        light = bpy.data.objects.new(name, light_data)
        light.location = target + Vector(offset)
        _look_at(light, target)
        presentation.objects.link(light)

    armature.data.display_type = "STICK"
    armature.show_in_front = True
    bpy.ops.object.mode_set(mode="OBJECT") if armature.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate microduck-alpha.blend from canonical robot sources."
    )
    default_code = Path.home() / "MyCode"
    parser.add_argument(
        "--rl-root",
        type=Path,
        default=default_code / "microduck_rl",
        help="microduck_rl checkout containing the canonical MJCF and STL assets",
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=default_code / "microduck",
        help="Microduck runtime checkout containing joint order and home pose",
    )
    parser.add_argument(
        "--mouth-linkage",
        type=Path,
        help="authorized CAD/mates linkage JSON; omit for the image-derived approximate hinge",
    )
    parser.add_argument(
        "--demo-motion",
        type=Path,
        default=REPO_ROOT / "assets/motions/microduck-crouch-test.npz",
        help="native mjlab motion archive to embed as the demonstration action",
    )
    parser.add_argument(
        "--policy-motion",
        type=Path,
        default=REPO_ROOT / "assets/motions/alpha-walking-forward.npz",
        help="native mjlab policy rollout to embed as the active walking action",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "microduck-alpha.blend",
    )
    return parser.parse_args(argv)


def build(args: argparse.Namespace) -> Path:
    mjcf = (
        args.rl_root
        / "src/mjlab_microduck/robot/microduck/robot_walk.xml"
    )
    runtime = args.runtime_root / "duck-control/src/model.rs"
    contract = args.runtime_root / "duck-ipc-proto/src/lib.rs"
    required = (mjcf, runtime, contract, args.demo_motion, args.policy_motion) + (
        (args.mouth_linkage,) if args.mouth_linkage is not None else ()
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProfileError("required source file(s) missing: " + ", ".join(missing))
    profile = build_microduck_profile(
        mjcf,
        runtime,
        args.mouth_linkage,
        joint_contract_path=contract,
    )
    armature = generate_microduck_scene(
        profile,
        mjcf,
        REPO_ROOT / "open_duck_tools",
        demo_motion_path=args.demo_motion,
    )
    import_motion_action(
        armature,
        profile,
        args.policy_motion,
        action_name="Policy_alpha_walking_forward",
        motion_kind="policy_rollout",
    )
    _organize_release_scene(armature, args.policy_motion)
    manifest = bpy.data.texts.get("microduck-build-manifest.json") or bpy.data.texts.new(
        "microduck-build-manifest.json"
    )
    manifest.clear()
    manifest.write(
        json.dumps(
            {
                "schema_version": 2,
                "robot_id": profile.robot_id,
                "mouth_mode": (
                    "authorized-cad" if args.mouth_linkage is not None else "image-derived-approximation"
                ),
                "motion_sha256": {
                    "KinematicCrouchTest": hashlib.sha256(
                        args.demo_motion.read_bytes()
                    ).hexdigest(),
                    "Policy_alpha_walking_forward": hashlib.sha256(
                        args.policy_motion.read_bytes()
                    ).hexdigest(),
                },
                "source_sha256": profile.source_sha256,
                "build_blender_version": bpy.app.version_string,
            },
            indent=2,
            sort_keys=True,
        )
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    armature["fk_ik"] = 0.0
    bpy.context.scene.frame_set(1)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    return output


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    try:
        output = build(_arguments(argv))
    except (MotionError, ProfileError, OSError) as exc:
        print(f"Microduck build failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
