import json
from pathlib import Path
import subprocess
import unittest

import bpy

RELEASE = Path(__file__).parents[1] / "microduck-alpha.blend"
CHECKER = Path(__file__).parents[1] / "tools/check_microduck_release.py"
POLICY_PREVIEW_CHECKER = Path(__file__).parents[1] / "tools/check_policy_preview.py"


class ReleaseArtifactTests(unittest.TestCase):
    def test_policy_preview_checker_reports_execution_source(self):
        expression = (
            "import json,sys;"
            f"sys.path.insert(0,{str(POLICY_PREVIEW_CHECKER.parents[1])!r});"
            "from tools.check_policy_preview import _execution_source;"
            "print(json.dumps([_execution_source(object()),"
            "_execution_source(None)]))"
        )
        result = subprocess.run(
            (
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python-expr",
                expression,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        sources = json.loads(
            next(line for line in result.stdout.splitlines() if line.startswith("["))
        )
        self.assertEqual(sources, ["exporter_child", "validated_cache"])

    def test_policy_preview_checker_uses_stable_defaults(self):
        expression = (
            "import json,sys;"
            f"sys.path.insert(0,{str(POLICY_PREVIEW_CHECKER.parents[1])!r});"
            "from tools.check_policy_preview import parse_args;"
            "args=parse_args(['example.blend']);"
            "print(json.dumps({'timeout':args.timeout,"
            "'output':str(args.roundtrip_output)}))"
        )
        result = subprocess.run(
            (
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python-expr",
                expression,
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        summary = json.loads(
            next(line for line in result.stdout.splitlines() if line.startswith("{"))
        )
        self.assertEqual(
            summary,
            {
                "timeout": 120.0,
                "output": "/tmp/microduck-policy-preview-roundtrip.npz",
            },
        )

    def test_policy_preview_checker_rejects_non_npz_output(self):
        for output in ("/tmp/not-an-archive.txt", "/tmp/not-an-archive.NPZ"):
            with self.subTest(output=output):
                result = subprocess.run(
                    (
                        bpy.app.binary_path,
                        "--background",
                        "--factory-startup",
                        "--python-exit-code",
                        "1",
                        "--python",
                        str(POLICY_PREVIEW_CHECKER),
                        "--",
                        str(RELEASE),
                        "--roundtrip-output",
                        output,
                    ),
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
                self.assertIn(".npz", result.stdout + result.stderr)

    def test_policy_preview_checker_rejects_non_blend_input(self):
        result = subprocess.run(
            (
                bpy.app.binary_path,
                "--background",
                "--factory-startup",
                "--python-exit-code",
                "1",
                "--python",
                str(POLICY_PREVIEW_CHECKER),
                "--",
                "/tmp/not-a-blender-file.txt",
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(".blend", result.stdout + result.stderr)

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
