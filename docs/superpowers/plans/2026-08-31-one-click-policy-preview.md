# One-Click Walking Policy Preview Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a responsive Open Duck sidebar workflow that runs the existing `microduck_rl` walking-policy exporter in the background and imports each validated rollout as a new Blender action.

**Architecture:** Blender owns configuration, safe subprocess lifecycle, cancellation, caching, and action import. The new `open_duck_tools.policy_preview` module is independent of `bpy`; `open_duck_tools.addon` adapts it to Blender timers and the existing transactional motion importer. ONNX Runtime and MuJoCo remain exclusively in the sibling `microduck_rl` process.

**Tech Stack:** Python standard library (`dataclasses`, `hashlib`, `json`, `pathlib`, `subprocess`, `threading`, `time`), Blender 4.3.2/5.2.1 Python API, `unittest`, existing `microduck_rl` `uv` CLI, NumPy motion archives.

**Spec:** `docs/superpowers/specs/2026-08-31-one-click-policy-preview-design.md`

## Global Constraints

- Support Linux for this milestone; do not add Windows or macOS process-tree behavior.
- Support walking policies using the established 61-dimensional command contract only.
- Default policy is `~/MyCode/microduck/policies/alpha_walking.onnx`.
- Default command is forward `0.30 m/s`, lateral `0.00 m/s`, yaw `0.00 rad/s`, duration `4.0 s`, seed `0`.
- Do not import ONNX Runtime, MuJoCo, or `microduck_rl` into Blender's Python environment.
- Launch `uv` with an argument list and `shell=False`.
- Only one policy-preview child process may run per Blender process.
- Failed or cancelled generation must not mutate Blender animation state.
- Successful imports must preserve all existing actions and create a collision-safe new action.
- Cache identity must include policy SHA-256, resolved policy path, canonical command/duration/seed, and exporter contract version.
- Build the release artifact with Blender 4.3.2 and verify it with Blender 4.3.2 and 5.2.1.
- Do not modify `open-duck-mini.blend` or its backup.

---

### Task 1: Preview Configuration, Naming, and Cache Identity

**Files:**
- Create: `open_duck_tools/policy_preview.py`
- Create: `tests/test_policy_preview.py`

**Interfaces:**
- Produces: `PreviewConfig`, `ValidatedPreview`, `PolicyPreviewError`.
- Produces: `validate_preview_config(config, *, which=shutil.which) -> ValidatedPreview`.
- Produces: `preview_action_name(command) -> str`.
- Produces: `temporary_output_path(validated) -> Path`.
- Consumes: no Blender API and no project-local modules.

- [ ] **Step 1: Write failing configuration and naming tests**

Create `tests/test_policy_preview.py` with literal behavior checks:

```python
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
```

- [ ] **Step 2: Run the tests and verify the module is missing**

Run:

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_policy_preview.PreviewConfigurationTests')); raise SystemExit(not r.wasSuccessful())"
```

Expected: FAIL with `ModuleNotFoundError: No module named 'open_duck_tools.policy_preview'`.

- [ ] **Step 3: Implement immutable configuration validation**

Create `open_duck_tools/policy_preview.py` with these public types and constants:

```python
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid

CONTROL_HZ = 50
EXPORTER_CONTRACT_VERSION = 1


class PolicyPreviewError(ValueError):
    pass


@dataclass(frozen=True)
class PreviewConfig:
    microduck_root: Path
    microduck_rl_root: Path
    policy_path: Path
    command: tuple[float, float, float]
    duration_s: float
    seed: int
    cache_root: Path


@dataclass(frozen=True)
class ValidatedPreview:
    config: PreviewConfig
    uv_executable: Path
    exporter_path: Path
    cwd: Path
    frames: int
    policy_sha256: str
    rollout_config_sha256: str
    canonical_config_json: str
    cache_key: str
    cache_path: Path
    action_name: str

    def argv_for(self, output_path: Path) -> tuple[str, ...]:
        x, y, yaw = self.config.command
        return (
            str(self.uv_executable), "run", "scripts/export_policy_rollout.py",
            str(self.config.policy_path.resolve()), "--output", str(output_path),
            "--duration", _canonical_float(self.config.duration_s),
            "--lin-vel-x", _canonical_float(x),
            "--lin-vel-y", _canonical_float(y),
            "--ang-vel-z", _canonical_float(yaw),
            "--seed", str(self.config.seed),
        )

    @property
    def argv(self) -> tuple[str, ...]:
        return self.argv_for(self.cache_path)
```

Implement `_canonical_float()` with `format(float(value), ".17g")`; normalize
values whose magnitude is below `0.0005` only in the two-decimal action label,
not in the cache key or CLI. `validate_preview_config()` must resolve paths,
check dependencies in the order `uv`, roots, exporter, policy, numeric values,
create the cache directory, and compute:

```python
payload = {
    "command": [_canonical_float(value) for value in config.command],
    "duration_s": _canonical_float(config.duration_s),
    "exporter_contract_version": EXPORTER_CONTRACT_VERSION,
    "policy_path": str(policy_path),
    "policy_sha256": policy_sha256,
    "seed": int(config.seed),
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
cache_key = hashlib.sha256(canonical.encode()).hexdigest()
cache_path = cache_root / f"{cache_key}.npz"
```

Also compute the exporter-contract digest independently so cached archives can
be checked against their embedded provenance:

```python
rollout_payload = {
    "command": [float(value) for value in config.command],
    "control_decimation": 4,
    "control_hz": 50,
    "duration_s": float(config.duration_s),
    "seed": int(config.seed),
    "timestep_s": 0.005,
}
rollout_config_sha256 = hashlib.sha256(
    json.dumps(
        rollout_payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
).hexdigest()
```

For the default `(0.30, 0.0, 0.0)`, `4.0 s`, seed `0` configuration, assert
the literal digest is
`eb7e3697bc1f166a458a080867f9fcf02f5c8005a404430a06b1437eb7187298`.

The frame test is:

```python
exact_frames = float(config.duration_s) * CONTROL_HZ
frames = round(exact_frames)
if not math.isclose(exact_frames, frames, rel_tol=0.0, abs_tol=1e-9):
    raise PolicyPreviewError("duration must resolve to an integral number of 50 Hz frames")
```

`temporary_output_path()` returns
`cache_path.with_name(f".{cache_key}.{uuid.uuid4().hex}.npz")`.

- [ ] **Step 4: Run configuration tests**

Run the command from Step 2.

Expected: all `PreviewConfigurationTests` pass.

- [ ] **Step 5: Commit the configuration boundary**

```bash
git add open_duck_tools/policy_preview.py tests/test_policy_preview.py
git commit -m "feat: define policy preview configurations"
```

---

### Task 2: Nonblocking Export Process and Bounded Logs

**Files:**
- Modify: `open_duck_tools/policy_preview.py`
- Modify: `tests/test_policy_preview.py`

**Interfaces:**
- Consumes: `ValidatedPreview.argv_for(output_path)` and `ValidatedPreview.cwd` from Task 1.
- Produces: `ProcessOutcome(returncode: int, cancelled: bool, log_tail: str)`.
- Produces: `PreviewProcess.start(validated, output_path, *, popen_factory=subprocess.Popen, clock=time.monotonic) -> PreviewProcess`.
- Produces: `PreviewProcess.poll() -> ProcessOutcome | None`, `request_cancel() -> None`, and `close(force: bool = False) -> None`.

- [ ] **Step 1: Add failing real-I/O and fake-process lifecycle tests**

Append to `tests/test_policy_preview.py`:

```python
import io
import sys
import time
from unittest import mock

from open_duck_tools.policy_preview import PreviewProcess


class FakeProcess:
    def __init__(self, output=b"", returncode=None):
        self.stdout = io.BytesIO(output)
        self.returncode = returncode
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1
        self.returncode = -9


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

    def test_cancel_terminates_then_kills_after_grace_period(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            fake = FakeProcess()
            now = [10.0]
            job = PreviewProcess.start(
                validated,
                output,
                popen_factory=lambda *args, **kwargs: fake,
                clock=lambda: now[0],
                cancel_grace_s=0.5,
            )
            job.request_cancel()
            self.assertEqual(fake.terminated, 1)
            self.assertIsNone(job.poll())
            now[0] = 10.6
            outcome = job.poll()
            job.close()

        self.assertEqual(fake.killed, 1)
        self.assertTrue(outcome.cancelled)
        self.assertEqual(outcome.returncode, -9)

    def test_force_close_kills_live_child_and_removes_temporary_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = self.validated(root)
            output = temporary_output_path(validated)
            output.write_bytes(b"partial")
            fake = FakeProcess()
            job = PreviewProcess.start(validated, output, popen_factory=lambda *a, **k: fake)
            job.close(force=True)
            self.assertEqual(fake.killed, 1)
            self.assertFalse(output.exists())
```

- [ ] **Step 2: Run process tests and verify missing interface failures**

Run:

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_policy_preview.PreviewProcessTests')); raise SystemExit(not r.wasSuccessful())"
```

Expected: FAIL because `PreviewProcess` is not defined.

- [ ] **Step 3: Implement the process runner**

Add these imports and result type to `policy_preview.py`:

```python
from collections import deque
import subprocess
import threading
import time


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    cancelled: bool
    log_tail: str
```

Implement `PreviewProcess` with:

- `Popen(validated.argv_for(output), cwd=str(validated.cwd), shell=False,
  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)`;
- a daemon reader thread that calls `stdout.read(4096)` until EOF;
- a locked `bytearray` tail trimmed from the front to `log_limit_bytes`;
- `poll()` returning `None` while live or while the reader has not reached EOF,
  and a single immutable `ProcessOutcome` after both process exit and reader
  completion;
- `request_cancel()` setting `cancelled=True`, calling `terminate()` once, and
  recording `clock() + cancel_grace_s`;
- `poll()` calling `kill()` when the cancellation deadline expires;
- `close(force=True)` killing a live process, closing stdout, joining the
  reader for at most `0.25 s`, and unlinking the temporary output;
- `close(force=False)` joining and closing after normal completion, unlinking
  output only for cancellation or nonzero exit.

Decode log bytes with `errors="replace"`. Do not expose the mutable buffer or
allow the reader thread to call project or Blender code.

- [ ] **Step 4: Run all policy-preview unit tests**

Run:

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_policy_preview')); raise SystemExit(not r.wasSuccessful())"
```

Expected: all Task 1 and Task 2 tests pass without thread warnings or leaked processes.

- [ ] **Step 5: Commit the runner**

```bash
git add open_duck_tools/policy_preview.py tests/test_policy_preview.py
git commit -m "feat: run policy previews without blocking Blender"
```

---

### Task 3: Blender Job Controller and Transactional Import

**Files:**
- Modify: `open_duck_tools/addon.py`
- Modify: `tests/test_addon.py`

**Interfaces:**
- Consumes: `PreviewConfig`, `ValidatedPreview`, `PreviewProcess`, `PolicyPreviewError`, `temporary_output_path`, and `validate_preview_config`.
- Consumes: `load_motion(path, profile)` for side-effect-free cache/archive validation.
- Consumes: existing `import_motion_action(..., before_mutation=...)` transactional import.
- Produces: `DUCK_OT_generate_policy_preview`, `DUCK_OT_cancel_policy_preview`.
- Produces internal controller functions: `_start_policy_preview(armature, context)`, `_poll_policy_preview_job()`, `_clear_policy_preview_job(*, force=True)`, `_import_policy_preview(session, path)`, `_ensure_policy_preview_timer()`.

- [ ] **Step 1: Add failing Blender operator/controller tests**

Extend `tests/test_addon.py` with this concrete fixture. Add imports for
`ProcessOutcome`, `PreviewConfig`, and `validate_preview_config` from
`open_duck_tools.policy_preview`, plus the standard-library `json` module:

```python
class FakePreviewProcess:
    def __init__(self, outcome=None):
        self.outcome = outcome
        self.cancel_requests = 0
        self.closed = []

    def poll(self):
        value, self.outcome = self.outcome, None
        return value

    def request_cancel(self):
        self.cancel_requests += 1

    def close(self, force=False):
        self.closed.append(force)


class PolicyPreviewOperatorTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon.register()
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)
        self.armature.data["duck_robot_profile_json"] = profile_to_json(self.profile)
        bpy.context.view_layer.objects.active = self.armature
        self.armature.select_set(True)
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        runtime = root / "microduck"
        rollout = root / "microduck_rl"
        policy = runtime / "policies/alpha_walking.onnx"
        exporter = rollout / "scripts/export_policy_rollout.py"
        policy.parent.mkdir(parents=True)
        exporter.parent.mkdir(parents=True)
        policy.write_bytes(b"policy")
        exporter.write_text("raise SystemExit(0)\n")
        self.validated = validate_preview_config(
            PreviewConfig(
                runtime, rollout, policy, (0.3, 0.0, 0.0), 0.06, 0, root / "cache"
            ),
            which=lambda _name: "/usr/bin/uv",
        )

    def tearDown(self):
        addon.unregister()
        self.temporary.cleanup()

    def write_archive(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = action_payload(self.profile)
        payload["source_hashes_json"] = np.asarray(
            [
                json.dumps(
                    {
                        "policy_sha256": self.validated.policy_sha256,
                        "rollout_config_sha256": self.validated.rollout_config_sha256,
                    },
                    sort_keys=True,
                )
            ]
        )
        np.savez_compressed(path, **payload)

    def install_session(self, process, output):
        addon._POLICY_PREVIEW_SESSION = addon._PolicyPreviewSession(
            self.armature.name,
            self.armature.as_pointer(),
            bpy.context.scene.name,
            self.validated,
            output,
            process,
        )

    def test_success_creates_new_action_without_replacing_existing_actions(self):
        existing = bpy.data.actions.new("Existing")
        self.armature.animation_data_create().action = existing
        output = self.validated.cache_path.with_name("finished.npz")
        self.write_archive(output)
        process = FakePreviewProcess(ProcessOutcome(0, False, "Frames: 3"))
        self.install_session(process, output)

        self.assertIsNone(addon._poll_policy_preview_job())

        self.assertIsNotNone(bpy.data.actions.get("Existing"))
        active = self.armature.animation_data.action
        self.assertEqual(active.name, "PolicyWalk_x0.30_y0.00_yaw0.00")
        self.assertEqual(active["duck_motion_kind"], "policy_preview")
        self.assertEqual(active["duck_policy_preview_cache_key"], self.validated.cache_key)
        self.assertIsNone(addon._POLICY_PREVIEW_SESSION)

    def test_exact_cache_hit_imports_without_starting_process(self):
        self.write_archive(self.validated.cache_path)
        with mock.patch.object(
            addon, "validate_preview_config", return_value=self.validated
        ), mock.patch.object(addon.PreviewProcess, "start") as start:
            result = bpy.ops.duck.generate_policy_preview()

        self.assertEqual(result, {"FINISHED"})
        start.assert_not_called()
        self.assertEqual(
            self.armature.animation_data.action.name,
            "PolicyWalk_x0.30_y0.00_yaw0.00",
        )
        self.assertEqual(
            self.armature.duck_policy_status,
            "Imported PolicyWalk_x0.30_y0.00_yaw0.00 (3 frames)",
        )

    def test_invalid_cache_is_removed_and_regenerated(self):
        self.validated.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.validated.cache_path, bad=np.array([1]))
        existing = bpy.data.actions.new("Existing")
        self.armature.animation_data_create().action = existing
        process = FakePreviewProcess()
        with mock.patch.object(
            addon, "validate_preview_config", return_value=self.validated
        ), mock.patch.object(
            addon.PreviewProcess, "start", return_value=process
        ) as start, mock.patch.object(addon, "_ensure_policy_preview_timer"):
            result = bpy.ops.duck.generate_policy_preview()

        self.assertEqual(result, {"FINISHED"})
        start.assert_called_once()
        self.assertFalse(self.validated.cache_path.exists())
        self.assertIs(self.armature.animation_data.action, existing)
        self.assertIsNotNone(addon._POLICY_PREVIEW_SESSION)

    def test_valid_archive_with_wrong_provenance_is_not_imported(self):
        self.write_archive(self.validated.cache_path)
        with np.load(self.validated.cache_path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload["source_hashes_json"] = np.asarray(
            [json.dumps({"policy_sha256": "0" * 64, "rollout_config_sha256": "1" * 64})]
        )
        np.savez_compressed(self.validated.cache_path, **payload)
        process = FakePreviewProcess()
        with mock.patch.object(
            addon, "validate_preview_config", return_value=self.validated
        ), mock.patch.object(
            addon.PreviewProcess, "start", return_value=process
        ), mock.patch.object(addon, "_ensure_policy_preview_timer"):
            result = bpy.ops.duck.generate_policy_preview()

        self.assertEqual(result, {"FINISHED"})
        self.assertFalse(self.validated.cache_path.exists())
        self.assertIsNone(self.armature.animation_data.action)
        self.assertIsNotNone(addon._POLICY_PREVIEW_SESSION)

    def test_failed_child_preserves_live_scene_state(self):
        scene = bpy.context.scene
        existing = bpy.data.actions.new("Existing")
        self.armature.animation_data_create().action = existing
        self.armature.location = (9.0, 8.0, 7.0)
        self.armature.duck_mouth_open = 0.7
        scene.frame_start, scene.frame_end = 7, 9
        scene.frame_set(8)
        original_matrix = self.armature.matrix_world.copy()
        original_actions = {action.name for action in bpy.data.actions}
        output = self.validated.cache_path.with_name("failed.npz")
        process = FakePreviewProcess(
            ProcessOutcome(2, False, "Policy rollout failed: incompatible input")
        )
        self.install_session(process, output)

        self.assertIsNone(addon._poll_policy_preview_job())

        self.assertIs(self.armature.animation_data.action, existing)
        self.assertEqual({action.name for action in bpy.data.actions}, original_actions)
        self.assertEqual((scene.frame_start, scene.frame_end, scene.frame_current), (7, 9, 8))
        self.assertEqual(self.armature.matrix_world, original_matrix)
        self.assertAlmostEqual(self.armature.duck_mouth_open, 0.7)
        self.assertEqual(self.armature.duck_policy_status, "Policy preview failed")
        self.assertIn("incompatible input", self.armature.duck_policy_details)

    def test_cancel_requests_termination_and_poll_finishes_cleanup(self):
        process = FakePreviewProcess()
        output = self.validated.cache_path.with_name("cancelled.npz")
        self.install_session(process, output)

        self.assertEqual(bpy.ops.duck.cancel_policy_preview(), {"FINISHED"})
        self.assertEqual(process.cancel_requests, 1)
        self.assertEqual(self.armature.duck_policy_status, "Cancelling")
        process.outcome = ProcessOutcome(-15, True, "")
        self.assertIsNone(addon._poll_policy_preview_job())
        self.assertIsNone(addon._POLICY_PREVIEW_SESSION)
        self.assertEqual(self.armature.duck_policy_status, "Cancelled")

    def test_unregister_force_closes_child_and_removes_timer(self):
        process = FakePreviewProcess()
        output = self.validated.cache_path.with_name("live.npz")
        self.install_session(process, output)
        bpy.app.timers.register(addon._poll_policy_preview_job, first_interval=60.0)

        addon.unregister()

        self.assertEqual(process.closed, [True])
        self.assertFalse(bpy.app.timers.is_registered(addon._poll_policy_preview_job))
        self.assertIsNone(addon._POLICY_PREVIEW_SESSION)
```

Use `mock.patch.object()` only at the external process boundary; the tests use
real Blender actions, transforms, properties, archive loading, and
`import_motion_action`.

- [ ] **Step 2: Run the controller tests and verify registration failures**

Run:

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('tests.test_addon.PolicyPreviewOperatorTests')); raise SystemExit(not r.wasSuccessful())"
```

Expected: FAIL because the two operators and controller globals do not exist.

- [ ] **Step 3: Implement job state and status properties**

In `addon.py`, import the Task 1/2 interfaces, `load_motion`, `os`, and
`dataclass`. Add:

```python
@dataclass
class _PolicyPreviewSession:
    armature_name: str
    armature_pointer: int
    scene_name: str
    validated: ValidatedPreview
    output_path: Path
    process: PreviewProcess


_POLICY_PREVIEW_SESSION: _PolicyPreviewSession | None = None
_POLICY_PREVIEW_TIMER_INTERVAL = 0.1
```

Register these `bpy.types.Object` properties, all limited to profiled Microduck
objects by panel/operator polling:

```python
duck_microduck_root = StringProperty(name="microduck checkout", subtype="DIR_PATH", default=str(Path.home() / "MyCode/microduck"))
duck_microduck_rl_root = StringProperty(name="microduck_rl checkout", subtype="DIR_PATH", default=str(Path.home() / "MyCode/microduck_rl"))
duck_policy_path = StringProperty(name="Walking policy", subtype="FILE_PATH", default=str(Path.home() / "MyCode/microduck/policies/alpha_walking.onnx"))
duck_policy_forward = FloatProperty(name="Forward", default=0.30, unit="VELOCITY")
duck_policy_lateral = FloatProperty(name="Lateral", default=0.0, unit="VELOCITY")
duck_policy_yaw = FloatProperty(name="Yaw rate", default=0.0, unit="ROTATION")
duck_policy_duration = FloatProperty(name="Duration", default=4.0, min=0.02, unit="TIME")
duck_policy_seed = IntProperty(name="Seed", default=0)
duck_policy_setup_open = BoolProperty(name="Setup", default=False)
duck_policy_status = StringProperty(name="Policy preview status", default="Idle", options={"SKIP_SAVE"})
duck_policy_details = StringProperty(name="Policy preview details", default="", options={"SKIP_SAVE"})
```

Delete each property in reverse order during unregister.

- [ ] **Step 4: Implement generate, poll, import, cancel, and cleanup**

`DUCK_OT_generate_policy_preview.execute()` must:

1. reject a non-Microduck object or existing global session;
2. build `PreviewConfig` from expanded/resolved Blender properties and
   `bpy.utils.user_resource("CACHE", path="open_duck/policy_previews", create=True)`;
3. set status `Preflight`, clear stale details, and call
   `validate_preview_config()`;
4. if `cache_path` exists, call `_validate_policy_preview_archive()`; on
   validation error unlink only that keyed cache file and proceed to generation;
5. on a valid cache hit, call `_import_policy_preview()` immediately;
6. on a miss, create the temporary path, call `PreviewProcess.start()`, store
   `_PolicyPreviewSession`, set status `Exporting`, and register
   `_poll_policy_preview_job` through `_ensure_policy_preview_timer()` if it is
   not registered. That helper is the only code path that calls
   `bpy.app.timers.register()`.

If cache deletion, temporary-path creation, or `PreviewProcess.start()` raises,
unlink only the current temporary path, clear the global session, set
`Policy preview failed`, retain a bounded exception message in details, and
return `{"CANCELLED"}` without touching actions or scene state.

`_poll_policy_preview_job()` returns `0.1` while the process is live. On a
nonzero or cancelled outcome it closes the process, clears the session, removes
the temporary output, preserves Blender data, writes a concise status/details,
and returns `None`. Implement the shared cache/completion validator as:

```python
def _validate_policy_preview_archive(path, profile, validated):
    motion = load_motion(path, profile)
    if motion.frames != validated.frames:
        raise MotionError(
            f"policy preview has {motion.frames} frames, expected {validated.frames}"
        )
    with np.load(path, allow_pickle=False) as archive:
        source_hashes = json.loads(str(archive["source_hashes_json"][0]))
    if source_hashes.get("policy_sha256") != validated.policy_sha256:
        raise MotionError("policy preview archive has the wrong policy SHA-256")
    if (
        source_hashes.get("rollout_config_sha256")
        != validated.rollout_config_sha256
    ):
        raise MotionError("policy preview archive has the wrong rollout configuration")
    return motion
```

On success it must:

```python
profile = profile_from_armature(armature)
armature.duck_policy_status = "Validating"
motion = _validate_policy_preview_archive(
    session.output_path, profile, session.validated
)  # validation before cache mutation
os.replace(session.output_path, session.validated.cache_path)
armature.duck_policy_status = "Importing"
action = import_motion_action(
    armature,
    profile,
    session.validated.cache_path,
    action_name=session.validated.action_name,
    motion_kind="policy_preview",
    before_mutation=lambda: _stop_playback(bpy.context),
)
action["duck_policy_preview_cache_key"] = session.validated.cache_key
```

Then report `f"Imported {action.name} ({motion.frames} frames)"`, close and clear
the session, and return `None`. Resolve the armature by both stored name and `as_pointer()`;
if it disappeared, force cleanup without import.

Wrap validation, `os.replace`, and import in one exception boundary covering
`MotionError`, `ProfileError`, `OSError`, `ValueError`, and `json.JSONDecodeError`.
On any exception, leave a successfully validated cache entry in place when it
already exists, rely on `import_motion_action` to restore its just-in-time
snapshot, mark the job failed, and perform the same process/session cleanup.

`DUCK_OT_cancel_policy_preview.execute()` calls `request_cancel()` once and sets
status `Cancelling`; the timer owns final cleanup. `_clear_policy_preview_job`
must unregister the timer when safe, force-close the process on unregister or
file load, remove only its temporary output, and clear the global.

Add a persistent `load_pre` handler tagged `_duck_policy_preview_handler` that
calls forced cleanup. Register an `atexit` callback that performs process-only
forced cleanup without touching `bpy`; unregister both hooks idempotently.

- [ ] **Step 5: Run add-on and policy-preview tests**

Run:

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromNames(['tests.test_policy_preview','tests.test_addon'])); r=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not r.wasSuccessful())"
```

Expected: all tests pass; Blender exits without a child process or registered timer.

- [ ] **Step 6: Commit Blender orchestration**

```bash
git add open_duck_tools/addon.py tests/test_addon.py
git commit -m "feat: generate walking previews from Blender"
```

---

### Task 4: Sidebar UI, Embedded Module, and Release Contract

**Files:**
- Modify: `open_duck_tools/addon.py`
- Modify: `open_duck_tools/builder.py`
- Modify: `tests/test_addon.py`
- Modify: `tests/test_builder.py`
- Modify: `tools/check_microduck_release.py`

**Interfaces:**
- Consumes: Task 3 operators, properties, and `_POLICY_PREVIEW_SESSION`.
- Produces: beginner-facing **Generate Policy Preview** panel state.
- Produces: self-contained text block `open_duck_tools.policy_preview` and bootstrap loading order.

- [ ] **Step 1: Add failing UI and embedding tests**

Add to `AddonRegistrationTests`:

```python
def test_registers_policy_preview_operators_and_defaults(self):
    addon.register()
    self.assertTrue(hasattr(bpy.types, "DUCK_OT_generate_policy_preview"))
    self.assertTrue(hasattr(bpy.types, "DUCK_OT_cancel_policy_preview"))
    self.assertEqual(bpy.types.Object.bl_rna.properties["duck_policy_forward"].default, 0.30)
    self.assertEqual(bpy.types.Object.bl_rna.properties["duck_policy_duration"].default, 4.0)
```

In the canonical builder/embed test, require:

```python
self.assertIsNotNone(bpy.data.texts.get("open_duck_tools.policy_preview"))
self.assertIn('"policy_preview"', bpy.data.texts["open_duck_bootstrap.py"].as_string())
```

In `tools/check_microduck_release.py`, add `policy_preview` to `expected_texts`.

- [ ] **Step 2: Run focused tests and verify missing UI/module failures**

Run:

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromNames(['tests.test_addon.AddonRegistrationTests','tests.test_builder.SceneBuilderTests.test_builds_profiled_rig_visual_and_driven_mouth_link'])); r=unittest.TextTestRunner(verbosity=2).run(suite); raise SystemExit(not r.wasSuccessful())"
```

Expected: FAIL because the operators/properties and embedded text are absent.

- [ ] **Step 3: Draw the beginner policy-preview box**

In `DUCK_PT_tools.draw()`, after Animation and before manual Import/Export, draw
only for `microduck-alpha`:

```python
preview = layout.box()
preview.label(text="Generate Policy Preview", icon="PLAY")
preview.prop(armature, "duck_policy_path", text="Policy")
commands = preview.column(align=True)
commands.prop(armature, "duck_policy_forward")
commands.prop(armature, "duck_policy_lateral")
commands.prop(armature, "duck_policy_yaw")
row = preview.row(align=True)
row.prop(armature, "duck_policy_duration")
row.prop(armature, "duck_policy_seed")
preview.prop(armature, "duck_policy_setup_open", text="Setup", toggle=True)
if armature.duck_policy_setup_open:
    preview.prop(armature, "duck_microduck_root")
    preview.prop(armature, "duck_microduck_rl_root")
if _POLICY_PREVIEW_SESSION is None:
    preview.operator("duck.generate_policy_preview", icon="FILE_REFRESH")
else:
    preview.operator("duck.cancel_policy_preview", icon="CANCEL")
preview.label(text=armature.duck_policy_status, icon="INFO")
if armature.duck_policy_details:
    for line in armature.duck_policy_details.splitlines()[-4:]:
        preview.label(text=line[:120])
```

Disable Generate when another object owns the running session, and label that
condition `Another policy preview is running` rather than exposing Cancel for
the wrong armature.

- [ ] **Step 4: Embed the pure policy-preview module before the add-on**

In `_embed_addon()`, change both module tuples to:

```python
("profile", "motion", "blender_bridge", "motion_import", "ik", "policy_preview", "addon")
```

The bootstrap must execute `policy_preview` before `addon` so the relative
import succeeds. Update embed tests to execute the bootstrap twice and verify
operator class identity remains idempotent, matching existing registration
coverage.

- [ ] **Step 5: Run focused UI/embed/release tests**

Run:

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromNames(['tests.test_addon','tests.test_builder','tests.test_build_cli'])); r=unittest.TextTestRunner(verbosity=1).run(suite); raise SystemExit(not r.wasSuccessful())"
```

Expected: all focused suites pass.

- [ ] **Step 6: Commit UI and embedding**

```bash
git add open_duck_tools/addon.py open_duck_tools/builder.py tests/test_addon.py tests/test_builder.py tools/check_microduck_release.py
git commit -m "feat: expose one-click policy previews"
```

---

### Task 5: Real Policy Acceptance Tool and Documentation

**Files:**
- Create: `tools/check_policy_preview.py`
- Modify: `README.md`
- Modify: `tools/check_microduck_release.py`
- Test: real sibling repositories and `~/MyCode/microduck/policies/alpha_walking.onnx`

**Interfaces:**
- Consumes: Task 3 operator/controller and existing `collect_armature_motion()` / `save_motion_npz()`.
- Produces: a headless end-to-end acceptance command for the exact beginner workflow.

- [ ] **Step 1: Write the failing acceptance script contract**

Create `tools/check_policy_preview.py` to:

1. parse a required `.blend` path, `--timeout` defaulting to `120`, and
   `--roundtrip-output` defaulting to
   `/tmp/microduck-policy-preview-roundtrip.npz`;
2. open the blend, register the source add-on with autoexec disabled, select
   `MicroduckRig`, and set the default policy/repository/command fields;
3. record the existing action names;
4. invoke `bpy.ops.duck.generate_policy_preview()`;
5. repeatedly call `addon._poll_policy_preview_job()` and sleep `0.05 s` until
   the global session clears or the timeout expires;
6. assert status begins with `Imported`, the action set gained exactly one
   action, the new active action has `duck_motion_kind == "policy_preview"`,
   and the range is exactly `1..200`;
7. export that action to the resolved `--roundtrip-output` `.npz` with
   `collect_armature_motion()` and `save_motion_npz()`;
8. print one JSON object containing action name, frames, cache key, root travel,
   and round-trip archive path;
9. always cancel/force-clean any remaining child in `finally`.

Before implementing Task 3/4 behavior, run:

```bash
blender --disable-autoexec --background microduck-alpha.blend --python tools/check_policy_preview.py -- microduck-alpha.blend
```

Expected: FAIL because `duck.generate_policy_preview` is unavailable.

- [ ] **Step 2: Implement the acceptance script exactly as specified**

Use `argparse`, `json`, and `time.monotonic`. Reject an output suffix other than
`.npz` and any operator return other than `{"FINISHED"}`. Calculate root travel as
`float(archive["body_pos_w"][-1, 0, 0] - archive["body_pos_w"][0, 0, 0])`
and require it to exceed `0.1 m`. Do not inspect UI pixels or bypass the operator.

- [ ] **Step 3: Document setup, generation, caching, and cancellation**

Add a README section immediately after **Import a policy rollout into Blender**
that gives this beginner path:

```text
1. Select MicroduckRig and open N -> Open Duck.
2. Expand Generate Policy Preview.
3. Keep alpha_walking.onnx or choose another compatible walking policy.
4. Set Forward/Lateral/Yaw, Duration, and Seed.
5. Press Generate & Import; continue using Blender or press Cancel.
6. The new PolicyWalk_* action becomes active when validation succeeds.
```

State that `uv`, `~/MyCode/microduck`, and `~/MyCode/microduck_rl` are required
for new generation; normal playback of existing actions remains self-contained.
Explain that exact configurations can reuse a validated user-cache archive and
that every click still creates a new action.

- [ ] **Step 4: Run real end-to-end acceptance and external validation**

Run:

```bash
blender --disable-autoexec --background microduck-alpha.blend \
  --python tools/check_policy_preview.py -- microduck-alpha.blend \
  | tee /tmp/microduck-policy-preview-check.log
```

The default output path is stable after Blender exits. Run:

```bash
cd /home/mcao/MyCode/microduck_rl
uv run scripts/validate_blender_motion.py /tmp/microduck-policy-preview-roundtrip.npz
```

Expected: 200 frames, root travel greater than `0.1 m`, and validator position
and orientation errors within its existing acceptance thresholds.

- [ ] **Step 5: Commit acceptance and documentation**

```bash
git add tools/check_policy_preview.py README.md tools/check_microduck_release.py
git commit -m "test: accept one-click policy previews"
```

---

### Task 6: Release Artifact, Cross-Version Gate, and MCP Inspection

**Files:**
- Modify: `microduck-alpha.blend`
- Verify: `open-duck-mini.blend` remains byte-identical.

**Interfaces:**
- Consumes: all prior tasks and sibling `microduck` / `microduck_rl` repositories.
- Produces: Blender 4.3.2-authored release with embedded one-click generation UI.

- [ ] **Step 1: Run the complete Blender 4.3.2 suite before rebuilding**

```bash
/tmp/microduck-blender43.mNP7Ow/blender-4.3.2-linux-x64/blender --background --factory-startup --python-expr \
  "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover('tests',top_level_dir='.')); raise SystemExit(not r.wasSuccessful())"
```

Expected: all tests pass; only documented version/environment skips remain.

- [ ] **Step 2: Rebuild with Blender 4.3.2**

```bash
/tmp/microduck-blender43.mNP7Ow/blender-4.3.2-linux-x64/blender --background --factory-startup \
  --python tools/build_microduck_blend.py -- \
  --runtime-root /home/mcao/MyCode/microduck \
  --rl-root /home/mcao/MyCode/microduck_rl
```

Expected: `microduck-alpha.blend` saves successfully and its manifest records
Blender 4.3.2.

- [ ] **Step 3: Run release checker and real operator acceptance**

```bash
/tmp/microduck-blender43.mNP7Ow/blender-4.3.2-linux-x64/blender --background microduck-alpha.blend \
  --python tools/check_microduck_release.py -- microduck-alpha.blend

/tmp/microduck-blender43.mNP7Ow/blender-4.3.2-linux-x64/blender --disable-autoexec --background microduck-alpha.blend \
  --python tools/check_policy_preview.py -- microduck-alpha.blend
```

Expected: release checker reports 15 canonical bodies, 70 visuals, no external
libraries, material viewports, open sidebars, and the acceptance tool imports a
new 200-frame action.

- [ ] **Step 4: Run the complete Blender 5.2.1 suite and release checker**

```bash
blender --background --factory-startup --python-expr \
  "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.discover('tests',top_level_dir='.')); raise SystemExit(not r.wasSuccessful())"

blender --disable-autoexec --background microduck-alpha.blend \
  --python tools/check_microduck_release.py -- microduck-alpha.blend
```

Expected: all tests and the release checker pass. The known Blender 5.2 shutdown
allocator diagnostic may appear only after successful tests and must not change
the exit code.

- [ ] **Step 5: Verify protected and generated artifact hashes**

```bash
sha256sum open-duck-mini.blend microduck-alpha.blend
git lfs status
git diff --check
```

Expected: `open-duck-mini.blend` remains
`5b1f21f2e7827ef683fc023a3826d105e73e7b9f799a4efb56ab35315cc67926`;
only the intended Microduck LFS artifact is modified.

- [ ] **Step 6: Inspect the live workflow through Blender MCP**

Open the rebuilt file in Blender 5.2.1, select `MicroduckRig`, and inspect the
Open Duck sidebar. Generate the default preview, confirm the status transitions
without UI freezing, then inspect frames 1, 100, and 200. Verify both feet remain
contact-consistent with the validated rollout and the root advances forward.
Press Generate again and confirm a suffixed new action appears without deleting
the first preview or `Policy_alpha_walking_forward`.

- [ ] **Step 7: Commit the final release**

```bash
git add microduck-alpha.blend
git commit -m "feat: ship one-click walking policy previews"
```

- [ ] **Step 8: Request independent review**

Use `superpowers:requesting-code-review` with the design, this plan, the base
commit, and the feature head. Fix all Critical and Important findings, rerun the
affected focused tests, then rerun both complete Blender suites before offering
branch integration.
