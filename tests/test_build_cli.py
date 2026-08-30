import importlib.util
import tempfile
import unittest
from pathlib import Path

from open_duck_tools.profile import ProfileError


SCRIPT = Path(__file__).parents[1] / "tools" / "build_microduck_blend.py"
SPEC = importlib.util.spec_from_file_location("build_microduck_blend", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildCliTests(unittest.TestCase):
    def test_defaults_to_explicit_approximate_mouth_mode(self):
        args = MODULE._arguments([])
        self.assertIsNone(args.mouth_linkage)

    def test_reports_all_missing_canonical_sources_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = MODULE._arguments(
                [
                    "--rl-root", str(root / "rl"),
                    "--runtime-root", str(root / "runtime"),
                ]
            )
            with self.assertRaisesRegex(ProfileError, "required source file"):
                MODULE.build(args)


if __name__ == "__main__":
    unittest.main()
