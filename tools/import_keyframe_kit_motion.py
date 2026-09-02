#!/usr/bin/env python3
"""Import a keyframe-kit motion as a versioned Blender Action."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_duck_tools.motion import MotionError
from open_duck_tools.motion_import import import_motion_action
from open_duck_tools.profile import ProfileError, profile_from_json


def _arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--action", required=True)
    parser.add_argument("--rkk-root", type=Path, required=True)
    parser.add_argument("--output-blend", type=Path)
    return parser.parse_args(argv)


def _armature():
    active = bpy.context.view_layer.objects.active
    if active is not None and active.type == "ARMATURE":
        return active
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1:
        raise MotionError(f"expected one active MicroDuck armature, got {len(armatures)}")
    return armatures[0]


def run(args: argparse.Namespace) -> Path:
    armature = _armature()
    encoded = armature.data.get("duck_robot_profile_json")
    if not encoded:
        raise ProfileError("active armature has no embedded duck_robot_profile_json")
    profile = profile_from_json(str(encoded))
    uv = Path.home() / ".local/bin/uv"
    if not uv.is_file():
        raise MotionError(f"uv executable is missing: {uv}")
    with tempfile.TemporaryDirectory(prefix="open-duck-keyframe-kit-") as directory:
        root = Path(directory)
        profile_path = root / "profile.json"
        native_path = root / "motion.npz"
        profile_path.write_text(str(encoded))
        command = [
            str(uv),
            "run",
            "--project",
            str(args.rkk_root),
            "python",
            str(REPO_ROOT / "tools/convert_keyframe_kit_motion.py"),
            str(args.input),
            "--profile-json",
            str(profile_path),
            "--output",
            str(native_path),
        ]
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise MotionError(f"keyframe-kit conversion process failed: {detail}")
        action = import_motion_action(
            armature,
            profile,
            native_path,
            action_name=args.action,
            motion_kind="keyframe_kit_reference",
        )
        action["duck_keyframe_kit_source"] = str(args.input.resolve())
        action["duck_review_phases_json"] = json.dumps(
            ["standing", "descending", "bottom", "ascending", "stable"]
        )
    output = (args.output_blend or Path(bpy.data.filepath)).expanduser().resolve()
    if not str(output):
        raise MotionError("--output-blend is required for an unsaved Blender project")
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    return output


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    try:
        output = run(_arguments(argv))
    except (MotionError, ProfileError, OSError) as exc:
        print(f"keyframe-kit import failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
