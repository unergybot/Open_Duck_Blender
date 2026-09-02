#!/usr/bin/env python3
"""Convert a keyframe-kit LZ4 file to a temporary native Open Duck NPZ."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from open_duck_tools.keyframe_kit_bridge import archive_from_keyframe_data
from open_duck_tools.motion import MotionError, save_motion_npz
from open_duck_tools.profile import ProfileError, profile_from_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        profile = profile_from_json(args.profile_json.read_text())
        archive = archive_from_keyframe_data(joblib.load(args.input), profile)
        save_motion_npz(args.output, archive)
    except (OSError, MotionError, ProfileError, ValueError) as exc:
        print(f"keyframe-kit conversion failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
