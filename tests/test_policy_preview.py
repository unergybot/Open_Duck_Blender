import io
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

from open_duck_tools.policy_preview import (
    PolicyPreviewError,
    PreviewConfig,
    PreviewProcess,
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

    def test_normalizes_blender_float32_durations_for_every_downstream_identity(self):
        cases = (
            (
                0.019999999552965164,
                0.02,
                1,
                "0.02",
                "165a857f3ffc3259ff6b914f2ccd1eba84973b6d42d646a5258a26b8af8d9f9c",
            ),
            (
                0.05999999865889549,
                0.06,
                3,
                "0.059999999999999998",
                "d5921fd5d7d846e4f04ac106374fa0c972f1aa6554793fcc6092d4f9b32982da",
            ),
            (
                0.10000000149011612,
                0.10,
                5,
                "0.10000000000000001",
                "9be3c4ba1eee2326dcf8b9dace73131bc86a942eb654288e233fb51f55c6d8e8",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = self.write_sources(root)
            for raw_duration, normalized, frames, cli_duration, provenance in cases:
                with self.subTest(raw_duration=raw_duration):
                    config = PreviewConfig(
                        runtime,
                        rollout,
                        policy,
                        (0.3, 0.0, 0.0),
                        raw_duration,
                        0,
                        root / f"cache-{frames}",
                    )
                    validated = validate_preview_config(
                        config, which=lambda _name: "/usr/bin/uv"
                    )
                    exact = validate_preview_config(
                        PreviewConfig(
                            runtime,
                            rollout,
                            policy,
                            (0.3, 0.0, 0.0),
                            normalized,
                            0,
                            root / f"cache-{frames}",
                        ),
                        which=lambda _name: "/usr/bin/uv",
                    )

                    duration_index = validated.argv.index("--duration") + 1
                    self.assertEqual(validated.frames, frames)
                    self.assertEqual(validated.config.duration_s, normalized)
                    self.assertEqual(validated.argv[duration_index], cli_duration)
                    self.assertEqual(
                        json.loads(validated.canonical_config_json)["duration_s"],
                        cli_duration,
                    )
                    self.assertEqual(validated.rollout_config_sha256, provenance)
                    self.assertEqual(validated.cache_key, exact.cache_key)

    def test_rejects_blender_float32_nonintegral_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = self.write_sources(root)
            config = PreviewConfig(
                runtime,
                rollout,
                policy,
                (0.3, 0.0, 0.0),
                0.03099999949336052,
                0,
                root / "cache",
            )

            with self.assertRaisesRegex(
                PolicyPreviewError, "integral number of 50 Hz frames"
            ):
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


class FakeProcess:
    def __init__(self, output=b"", returncode=None, pid=43210):
        self.stdout = io.BytesIO(output)
        self.returncode = returncode
        self.pid = pid
        self.terminated = 0
        self.killed = 0
        self.waits = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1
        self.returncode = -9

    def wait(self, timeout=None):
        self.waits.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class PreviewProcessTests(unittest.TestCase):
    def validated(self, root: Path):
        runtime, rollout, policy = PreviewConfigurationTests().write_sources(root)
        config = PreviewConfig(runtime, rollout, policy, (0.3, 0.0, 0.0), 4.0, 0, root / "cache")
        return validate_preview_config(config, which=lambda _name: sys.executable)

    def test_reader_drains_large_output_and_keeps_only_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            fake = FakeProcess(output=b"x" * 200000 + b"TAIL-MARKER", returncode=0)
            job = PreviewProcess.start(
                validated,
                output,
                popen_factory=lambda *args, **kwargs: fake,
                log_limit_bytes=32768,
            )
            deadline = time.monotonic() + 10.0
            outcome = None
            while outcome is None and time.monotonic() < deadline:
                outcome = job.poll()
                time.sleep(0.01)
            job.close()

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.returncode, 0)
        self.assertIn("TAIL-MARKER", outcome.log_tail)
        self.assertLessEqual(len(outcome.log_tail.encode()), 32768)

    def test_launch_arguments_are_exact_and_shell_is_false(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            fake = FakeProcess(returncode=0)
            factory = mock.Mock(return_value=fake)
            job = PreviewProcess.start(validated, output, popen_factory=factory)
            while job.poll() is None:
                time.sleep(0.001)
            job.close()

        args, kwargs = factory.call_args
        self.assertEqual(args[0], validated.argv_for(output))
        self.assertEqual(kwargs["cwd"], str(validated.cwd))
        self.assertFalse(kwargs["shell"])
        self.assertEqual(kwargs["stderr"], __import__("subprocess").STDOUT)
        self.assertTrue(kwargs["start_new_session"])

    def test_cancel_terms_then_kills_the_dedicated_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            fake = FakeProcess()
            now = [10.0]
            signals = []

            def signal_group(process_group, requested_signal):
                signals.append((process_group, requested_signal))
                if requested_signal == signal.SIGKILL:
                    fake.returncode = -signal.SIGKILL

            with mock.patch("os.killpg", side_effect=signal_group):
                job = PreviewProcess.start(
                    validated,
                    output,
                    popen_factory=lambda *args, **kwargs: fake,
                    clock=lambda: now[0],
                    cancel_grace_s=0.5,
                )
                job.request_cancel()
                self.assertIsNone(job.poll())
                now[0] = 10.6
                outcome = job.poll()
                job.close()

        self.assertEqual(
            signals,
            [(fake.pid, signal.SIGTERM), (fake.pid, signal.SIGKILL)],
        )
        self.assertTrue(outcome.cancelled)
        self.assertEqual(outcome.returncode, -9)

    def test_cancel_escalates_group_after_wrapper_already_exited(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            fake = FakeProcess(returncode=-signal.SIGKILL)
            now = [10.0]
            with mock.patch("os.killpg") as killpg:
                job = PreviewProcess.start(
                    validated,
                    output,
                    popen_factory=lambda *args, **kwargs: fake,
                    clock=lambda: now[0],
                    cancel_grace_s=0.5,
                )
                job.request_cancel()
                now[0] = 10.6
                outcome = job.poll()
                job.close()

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(fake.pid, signal.SIGTERM),
                mock.call(fake.pid, signal.SIGKILL),
            ],
        )
        self.assertTrue(outcome.cancelled)

    def test_force_close_kills_live_child_and_removes_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            output.write_bytes(b"partial")
            fake = FakeProcess()
            job = PreviewProcess.start(validated, output, popen_factory=lambda *a, **k: fake)
            with mock.patch(
                "os.killpg",
                side_effect=lambda _group, _signal: setattr(
                    fake, "returncode", -signal.SIGKILL
                ),
            ) as killpg:
                job.close(force=True)
            killpg.assert_called_once_with(fake.pid, signal.SIGKILL)
            self.assertEqual(fake.waits, [0.25])
            self.assertFalse(output.exists())

    def test_force_close_bounds_cleanup_terminates_reader_and_reaps_live_subprocess(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            child = None

            def start_sleeping_child(*_args, **kwargs):
                nonlocal child
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-c",
                        "import sys,time; sys.stdout.write('started\\n'); sys.stdout.flush(); time.sleep(30)",
                    ],
                    stdout=kwargs["stdout"],
                    stderr=kwargs["stderr"],
                    start_new_session=kwargs["start_new_session"],
                )
                return child

            job = PreviewProcess.start(validated, output, popen_factory=start_sleeping_child)
            try:
                deadline = time.monotonic() + 2.0
                while not job._reader.is_alive() and time.monotonic() < deadline:
                    time.sleep(0.001)
                self.assertTrue(job._reader.is_alive())
                started = time.monotonic()
                job.close(force=True)
                elapsed = time.monotonic() - started

                self.assertLess(elapsed, 0.75)
                self.assertFalse(job._reader.is_alive())
                self.assertIsNotNone(child.returncode)
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=1.0)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process groups only")
    def test_real_uv_orphan_descendant_is_killed_and_reaped_after_wrapper_exit(self):
        uv = shutil.which("uv")
        self.assertIsNotNone(uv)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, rollout, policy = PreviewConfigurationTests().write_sources(root)
            exporter = rollout / "scripts/export_policy_rollout.py"
            exporter.write_text(
                """\
import os
from pathlib import Path
import signal
import sys
import time

output = Path(sys.argv[sys.argv.index(\"--output\") + 1])
Path(str(output) + \".exporter-pid\").write_text(str(os.getpid()))
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print(\"EXPORTER_READY\", flush=True)
while True:
    time.sleep(0.05)
"""
            )
            validated = validate_preview_config(
                PreviewConfig(
                    runtime,
                    rollout,
                    policy,
                    (0.3, 0.0, 0.0),
                    0.02,
                    0,
                    root / "cache",
                ),
                which=lambda _name: uv,
            )
            output = temporary_output_path(validated)
            pid_path = Path(str(output) + ".exporter-pid")
            wrapper = None
            exporter_pid = None
            exporter_process_group = None

            def capture_wrapper(*args, **kwargs):
                nonlocal wrapper
                wrapper = subprocess.Popen(*args, **kwargs)
                return wrapper

            job = PreviewProcess.start(
                validated,
                output,
                popen_factory=capture_wrapper,
                cancel_grace_s=0.1,
            )
            try:
                deadline = time.monotonic() + 10.0
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(pid_path.exists(), "real uv exporter never became ready")
                exporter_pid = int(pid_path.read_text())
                self.assertNotEqual(wrapper.pid, exporter_pid)

                os.kill(wrapper.pid, signal.SIGKILL)
                while wrapper.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertIsNotNone(wrapper.returncode)
                self.assertTrue(Path(f"/proc/{exporter_pid}").exists())
                exporter_process_group = os.getpgid(exporter_pid)

                job.request_cancel()
                outcome = None
                deadline = time.monotonic() + 5.0
                while outcome is None and time.monotonic() < deadline:
                    outcome = job.poll()
                    time.sleep(0.01)

                self.assertIsNotNone(outcome)
                self.assertTrue(outcome.cancelled)
            finally:
                if exporter_pid is not None and Path(f"/proc/{exporter_pid}").exists():
                    try:
                        os.kill(exporter_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                job.close(force=True)
                deadline = time.monotonic() + 2.0
                while (
                    exporter_pid is not None
                    and Path(f"/proc/{exporter_pid}").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)

            self.assertFalse(Path(f"/proc/{wrapper.pid}").exists())
            self.assertFalse(Path(f"/proc/{exporter_pid}").exists())
            self.assertEqual(exporter_process_group, wrapper.pid)
