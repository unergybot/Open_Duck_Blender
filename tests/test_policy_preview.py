import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from open_duck_tools.policy_preview import (
    PolicyPreviewError,
    PreviewConfig,
    preview_action_name,
    temporary_output_path,
    validate_preview_config,
)


class PreviewConfigurationTests(unittest.TestCase):
    def write_sources(self, root: Path):
        runtime = root / "microduck"
        rollout = root / "microduck_rl"
        policy = runtime / "policies/alpha walking;safe.onnx"
        exporter = rollout / "scripts/export_policy_rollout.py"
        policy.parent.mkdir(parents=True)
        exporter.parent.mkdir(parents=True)
        policy.write_bytes(b"policy-fixture")
        exporter.write_text("raise SystemExit(0)\n")
        return runtime, rollout, policy

    def test_validates_exact_frames_and_builds_safe_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = self.write_sources(root)
            cache = root / "cache"
            config = PreviewConfig(
                microduck_root=runtime,
                microduck_rl_root=rollout,
                policy_path=policy,
                command=(0.2, -0.1, 0.3),
                duration_s=4.0,
                seed=7,
                cache_root=cache,
            )
            validated = validate_preview_config(
                config, which=lambda name: "/opt/uv bin/uv" if name == "uv" else None
            )

        self.assertEqual(validated.frames, 200)
        self.assertEqual(validated.policy_sha256, hashlib.sha256(b"policy-fixture").hexdigest())
        self.assertEqual(validated.action_name, "PolicyWalk_x0.20_y-0.10_yaw0.30")
        self.assertEqual(validated.argv[0], "/opt/uv bin/uv")
        self.assertEqual(validated.argv[1:3], ("run", "scripts/export_policy_rollout.py"))
        self.assertIn(str(policy.resolve()), validated.argv)
        self.assertNotIn("shell", " ".join(validated.argv))
        self.assertEqual(validated.cwd, rollout.resolve())
        self.assertEqual(validated.cache_path.parent, cache.resolve())

    def test_cache_key_is_canonical_and_policy_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = self.write_sources(root)
            base = PreviewConfig(runtime, rollout, policy, (0.3, 0.0, 0.0), 4.0, 0, root / "cache")
            first = validate_preview_config(base, which=lambda _name: "/usr/bin/uv")
            second = validate_preview_config(base, which=lambda _name: "/usr/bin/uv")
            policy.write_bytes(b"changed-policy")
            changed = validate_preview_config(base, which=lambda _name: "/usr/bin/uv")

        self.assertEqual(first.cache_key, second.cache_key)
        self.assertNotEqual(first.cache_key, changed.cache_key)
        self.assertEqual(
            first.rollout_config_sha256,
            "eb7e3697bc1f166a458a080867f9fcf02f5c8005a404430a06b1437eb7187298",
        )
        payload = json.loads(first.canonical_config_json)
        self.assertEqual(payload["command"], ["0.29999999999999999", "0", "0"])
        self.assertEqual(payload["exporter_contract_version"], 1)

    def test_rejects_nonintegral_duration_and_nonfinite_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = self.write_sources(root)
            for command, duration, message in (
                ((math.nan, 0.0, 0.0), 4.0, "command must be finite"),
                ((0.3, 0.0, 0.0), 0.031, "integral number of 50 Hz frames"),
                ((0.3, 0.0, 0.0), 0.0, "duration must be positive"),
            ):
                config = PreviewConfig(runtime, rollout, policy, command, duration, 0, root / "cache")
                with self.subTest(message=message), self.assertRaisesRegex(PolicyPreviewError, message):
                    validate_preview_config(config, which=lambda _name: "/usr/bin/uv")

    def test_reports_each_missing_dependency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = self.write_sources(root)
            config = PreviewConfig(runtime, rollout, policy, (0.3, 0.0, 0.0), 4.0, 0, root / "cache")
            with self.assertRaisesRegex(PolicyPreviewError, "uv executable"):
                validate_preview_config(config, which=lambda _name: None)
            policy.unlink()
            with self.assertRaisesRegex(PolicyPreviewError, "policy file"):
                validate_preview_config(config, which=lambda _name: "/usr/bin/uv")

    def test_temporary_output_is_hidden_unique_npz_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = self.write_sources(root)
            config = PreviewConfig(runtime, rollout, policy, (0.3, 0.0, 0.0), 4.0, 0, root / "cache")
            validated = validate_preview_config(config, which=lambda _name: "/usr/bin/uv")
            first = temporary_output_path(validated)
            second = temporary_output_path(validated)

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, validated.cache_path.parent)
        self.assertEqual(first.suffix, ".npz")
        self.assertTrue(first.name.startswith(f".{validated.cache_key}."))

    def test_action_name_normalizes_negative_zero(self):
        self.assertEqual(
            preview_action_name((-0.0, 0.0001, -0.0001)),
            "PolicyWalk_x0.00_y0.00_yaw0.00",
        )
