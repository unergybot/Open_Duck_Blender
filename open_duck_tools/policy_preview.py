"""Pure-Python configuration and identity boundary for policy previews."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import threading
import time
import uuid


CONTROL_HZ = 50
EXPORTER_CONTRACT_VERSION = 1


class PolicyPreviewError(ValueError):
    """Raised when a policy preview cannot be configured safely."""


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
            str(self.uv_executable),
            "run",
            "scripts/export_policy_rollout.py",
            str(self.config.policy_path.resolve()),
            "--output",
            str(Path(output_path)),
            "--duration",
            _canonical_float(self.config.duration_s),
            "--lin-vel-x",
            _canonical_float(x),
            "--lin-vel-y",
            _canonical_float(y),
            "--ang-vel-z",
            _canonical_float(yaw),
            "--seed",
            str(self.config.seed),
        )

    @property
    def argv(self) -> tuple[str, ...]:
        return self.argv_for(self.cache_path)


@dataclass(frozen=True)
class ProcessOutcome:
    returncode: int
    cancelled: bool
    log_tail: str


class PreviewProcess:
    """A nonblocking policy-export child process with a bounded output tail."""

    def __init__(
        self,
        process,
        output_path: Path,
        *,
        clock,
        cancel_grace_s: float,
        log_limit_bytes: int,
    ):
        self._process = process
        self._output_path = Path(output_path)
        self._clock = clock
        self._cancel_grace_s = cancel_grace_s
        self._log_limit_bytes = log_limit_bytes
        self._tail = bytearray()
        self._tail_lock = threading.Lock()
        self._reader_done = threading.Event()
        self._reader = threading.Thread(target=self._drain_stdout, daemon=True)
        self._cancelled = False
        self._cancel_deadline = None
        self._kill_requested = False
        self._outcome = None
        self._reader.start()

    @classmethod
    def start(
        cls,
        validated: ValidatedPreview,
        output_path: Path,
        *,
        popen_factory=subprocess.Popen,
        clock=time.monotonic,
        log_limit_bytes: int = 32768,
        cancel_grace_s: float = 2.0,
    ) -> PreviewProcess:
        process = popen_factory(
            validated.argv_for(output_path),
            cwd=str(validated.cwd),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return cls(
            process,
            output_path,
            clock=clock,
            cancel_grace_s=cancel_grace_s,
            log_limit_bytes=log_limit_bytes,
        )

    def _drain_stdout(self) -> None:
        stdout = self._process.stdout
        try:
            while True:
                chunk = stdout.read(4096)
                if not chunk:
                    return
                with self._tail_lock:
                    self._tail.extend(chunk)
                    excess = len(self._tail) - self._log_limit_bytes
                    if excess > 0:
                        del self._tail[:excess]
        except (OSError, ValueError):
            pass
        finally:
            self._reader_done.set()

    def _close_stdout(self) -> None:
        stdout = self._process.stdout
        if stdout is not None:
            stdout.close()

    def poll(self) -> ProcessOutcome | None:
        returncode = self._process.poll()
        if (
            returncode is None
            and self._cancel_deadline is not None
            and not self._kill_requested
            and self._clock() >= self._cancel_deadline
        ):
            self._process.kill()
            self._kill_requested = True
            returncode = self._process.poll()
        if returncode is None or not self._reader_done.is_set():
            return None
        if self._outcome is None:
            with self._tail_lock:
                log_tail = bytes(self._tail).decode(errors="replace")
            self._outcome = ProcessOutcome(returncode, self._cancelled, log_tail)
        return self._outcome

    def request_cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._cancel_deadline = self._clock() + self._cancel_grace_s
        if self._process.poll() is None:
            self._process.terminate()

    def close(self, force: bool = False) -> None:
        if force:
            if self._process.poll() is None:
                self._process.kill()
            self._close_stdout()
            self._reader.join(0.25)
            self._output_path.unlink(missing_ok=True)
            return

        outcome = self.poll()
        if outcome is None:
            return
        self._reader.join()
        self._close_stdout()
        if outcome.cancelled or outcome.returncode != 0:
            self._output_path.unlink(missing_ok=True)


def _canonical_float(value: float) -> str:
    return format(float(value), ".17g")


def preview_action_name(command: tuple[float, float, float]) -> str:
    """Return the stable, human-readable action label for a command."""

    labels = []
    for value in command:
        numeric = float(value)
        if abs(numeric) < 0.0005:
            numeric = 0.0
        labels.append(f"{numeric:.2f}")
    x, y, yaw = labels
    return f"PolicyWalk_x{x}_y{y}_yaw{yaw}"


def validate_preview_config(
    config: PreviewConfig, *, which=shutil.which
) -> ValidatedPreview:
    """Validate dependencies and return an immutable, cache-identifying preview."""

    uv = which("uv")
    if not uv:
        raise PolicyPreviewError("uv executable was not found")

    microduck_root = Path(config.microduck_root).resolve()
    microduck_rl_root = Path(config.microduck_rl_root).resolve()
    if not microduck_root.is_dir():
        raise PolicyPreviewError(f"microduck root is not a directory: {microduck_root}")
    if not microduck_rl_root.is_dir():
        raise PolicyPreviewError(f"microduck_rl root is not a directory: {microduck_rl_root}")

    exporter_path = (microduck_rl_root / "scripts/export_policy_rollout.py").resolve()
    if not exporter_path.is_file():
        raise PolicyPreviewError(f"exporter script is missing: {exporter_path}")

    policy_path = Path(config.policy_path).resolve()
    if not policy_path.is_file():
        raise PolicyPreviewError(f"policy file is missing: {policy_path}")

    try:
        command = tuple(float(value) for value in config.command)
    except (TypeError, ValueError) as exc:
        raise PolicyPreviewError("command must contain three finite numbers") from exc
    if len(command) != 3 or not all(math.isfinite(value) for value in command):
        raise PolicyPreviewError("command must be finite")

    try:
        duration_s = float(config.duration_s)
    except (TypeError, ValueError) as exc:
        raise PolicyPreviewError("duration must be positive") from exc
    if not math.isfinite(duration_s) or duration_s <= 0:
        raise PolicyPreviewError("duration must be positive")
    exact_frames = duration_s * CONTROL_HZ
    frames = round(exact_frames)
    if not math.isclose(exact_frames, frames, rel_tol=0.0, abs_tol=1e-9):
        raise PolicyPreviewError("duration must resolve to an integral number of 50 Hz frames")

    try:
        seed = int(config.seed)
    except (TypeError, ValueError) as exc:
        raise PolicyPreviewError("seed must be an integer") from exc

    cache_root = Path(config.cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    normalized_config = PreviewConfig(
        microduck_root=microduck_root,
        microduck_rl_root=microduck_rl_root,
        policy_path=policy_path,
        command=command,
        duration_s=duration_s,
        seed=seed,
        cache_root=cache_root,
    )
    policy_sha256 = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    payload = {
        "command": [_canonical_float(value) for value in command],
        "duration_s": _canonical_float(duration_s),
        "exporter_contract_version": EXPORTER_CONTRACT_VERSION,
        "policy_path": str(policy_path),
        "policy_sha256": policy_sha256,
        "seed": seed,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    cache_key = hashlib.sha256(canonical.encode()).hexdigest()
    cache_path = cache_root / f"{cache_key}.npz"

    rollout_payload = {
        "command": [float(value) for value in command],
        "control_decimation": 4,
        "control_hz": 50,
        "duration_s": float(duration_s),
        "seed": seed,
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

    return ValidatedPreview(
        config=normalized_config,
        uv_executable=Path(uv).resolve(),
        exporter_path=exporter_path,
        cwd=microduck_rl_root,
        frames=frames,
        policy_sha256=policy_sha256,
        rollout_config_sha256=rollout_config_sha256,
        canonical_config_json=canonical,
        cache_key=cache_key,
        cache_path=cache_path,
        action_name=preview_action_name(command),
    )


def temporary_output_path(validated: ValidatedPreview) -> Path:
    return validated.cache_path.with_name(f".{validated.cache_key}.{uuid.uuid4().hex}.npz")
