#!/usr/bin/env python3
"""Build the generated Microduck Blender project inside Blender."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bpy


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_duck_tools.builder import generate_microduck_scene
from open_duck_tools.profile import ProfileError, build_microduck_profile


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
    required = (mjcf, runtime, contract) + (
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
    generate_microduck_scene(profile, mjcf, REPO_ROOT / "open_duck_tools")
    manifest = bpy.data.texts.get("microduck-build-manifest.json") or bpy.data.texts.new(
        "microduck-build-manifest.json"
    )
    manifest.clear()
    manifest.write(
        json.dumps(
            {
                "schema_version": 1,
                "robot_id": profile.robot_id,
                "mouth_mode": (
                    "authorized-cad" if args.mouth_linkage is not None else "image-derived-approximation"
                ),
                "source_sha256": profile.source_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), check_existing=False)
    return output


def main() -> int:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    try:
        output = build(_arguments(argv))
    except (ProfileError, OSError) as exc:
        print(f"Microduck build failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
