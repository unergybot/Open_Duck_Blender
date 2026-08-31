#!/usr/bin/env python3
"""Run the real one-click policy preview workflow headlessly."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import bpy


DEFAULT_ROUNDTRIP_OUTPUT = Path("/tmp/microduck-policy-preview-roundtrip.npz")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_duck_tools import addon
from open_duck_tools.addon import collect_armature_motion, profile_from_armature
from open_duck_tools.motion import save_motion_npz


def _blend_path(value: str) -> Path:
    path = Path(value)
    if path.suffix != ".blend":
        raise argparse.ArgumentTypeError("input must use the .blend suffix")
    return path


def _npz_path(value: str) -> Path:
    path = Path(value)
    if path.suffix != ".npz":
        raise argparse.ArgumentTypeError("round-trip output must use the .npz suffix")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("blend", type=_blend_path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--roundtrip-output",
        type=_npz_path,
        default=DEFAULT_ROUNDTRIP_OUTPUT,
    )
    return parser.parse_args(argv)


def _select_microduck_rig():
    armature = bpy.data.objects.get("MicroduckRig")
    if armature is None or armature.type != "ARMATURE":
        raise AssertionError("blend is missing the MicroduckRig armature")
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    return armature


def _set_default_preview_fields(armature) -> None:
    workspace = Path.home() / "MyCode"
    armature.duck_microduck_root = str(workspace / "microduck")
    armature.duck_microduck_rl_root = str(workspace / "microduck_rl")
    armature.duck_policy_path = str(
        workspace / "microduck/policies/alpha_walking.onnx"
    )
    armature.duck_policy_forward = 0.30
    armature.duck_policy_lateral = 0.0
    armature.duck_policy_yaw = 0.0
    armature.duck_policy_duration = 4.0
    armature.duck_policy_seed = 0


def check_policy_preview(args: argparse.Namespace) -> dict[str, object]:
    blend_path = args.blend.expanduser().resolve()
    roundtrip_output = args.roundtrip_output.expanduser().resolve()
    bpy.ops.wm.open_mainfile(filepath=str(blend_path), load_ui=True)
    addon.register()
    armature = _select_microduck_rig()
    _set_default_preview_fields(armature)
    existing_actions = set(bpy.data.actions.keys())

    try:
        result = bpy.ops.duck.generate_policy_preview()
        if result != {"FINISHED"}:
            raise AssertionError(
                "duck.generate_policy_preview returned "
                f"{result!r}, expected {{'FINISHED'}}"
            )

        deadline = time.monotonic() + args.timeout
        while addon._POLICY_PREVIEW_SESSION is not None:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"policy preview did not finish within {args.timeout:g} seconds"
                )
            addon._poll_policy_preview_job()
            if addon._POLICY_PREVIEW_SESSION is not None:
                time.sleep(0.05)

        if not armature.duck_policy_status.startswith("Imported"):
            raise AssertionError(
                "policy preview did not import successfully: "
                f"{armature.duck_policy_status}: {armature.duck_policy_details}"
            )
        added_actions = set(bpy.data.actions.keys()) - existing_actions
        if len(added_actions) != 1:
            raise AssertionError(
                f"policy preview added {len(added_actions)} actions, "
                "expected exactly one"
            )
        action = armature.animation_data.action if armature.animation_data else None
        if action is None or action.name not in added_actions:
            raise AssertionError("the newly added policy preview action is not active")
        if action.get("duck_motion_kind") != "policy_preview":
            raise AssertionError("the new action is not marked as a policy preview")

        scene = bpy.context.scene
        action_range = tuple(float(value) for value in action.frame_range)
        if action_range != (1.0, 200.0):
            raise AssertionError(
                f"policy preview action range is {action_range}, expected (1.0, 200.0)"
            )
        if (scene.frame_start, scene.frame_end) != (1, 200):
            raise AssertionError(
                "policy preview scene range is "
                f"{(scene.frame_start, scene.frame_end)}, expected (1, 200)"
            )

        profile = profile_from_armature(armature)
        archive = collect_armature_motion(armature, profile, 1, 200)
        root_travel = float(
            archive["body_pos_w"][-1, 0, 0] - archive["body_pos_w"][0, 0, 0]
        )
        if root_travel <= 0.1:
            raise AssertionError(
                f"policy preview root travel is {root_travel:.6g} m, expected > 0.1 m"
            )
        destination = save_motion_npz(roundtrip_output, archive)
        return {
            "action_name": action.name,
            "frames": 200,
            "cache_key": str(action.get("duck_policy_preview_cache_key", "")),
            "root_travel_m": root_travel,
            "roundtrip_archive": str(destination),
        }
    finally:
        addon._clear_policy_preview_job(force=True)


def main() -> int:
    args = parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    print(json.dumps(check_policy_preview(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
