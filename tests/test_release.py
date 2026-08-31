import json
from pathlib import Path
import subprocess
import unittest

import bpy

RELEASE = Path(__file__).parents[1] / "microduck-alpha.blend"
CHECKER = Path(__file__).parents[1] / "tools/check_microduck_release.py"


class ReleaseArtifactTests(unittest.TestCase):
    def test_tracked_release_passes_headless_acceptance(self):
        result = subprocess.run(
            (
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(CHECKER),
                "--",
                str(RELEASE),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(
            next(line for line in result.stdout.splitlines() if line.startswith("{"))
        )

        self.assertEqual(summary["walk_frames"], 200)
        self.assertEqual(summary["crouch_frames"], 51)
        self.assertEqual(summary["canonical_bodies"], 15)
        self.assertEqual(summary["canonical_visuals"], 70)
        self.assertEqual(summary["external_libraries"], 0)
        self.assertEqual(summary["material_viewports"], summary["viewports"])
        self.assertEqual(summary["open_sidebars"], summary["viewports"])


if __name__ == "__main__":
    unittest.main()
