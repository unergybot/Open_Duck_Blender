"""Versioned robot-profile construction independent of Blender."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET


class ProfileError(ValueError):
    """A robot source violates the versioned profile contract."""


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class MouthLink:
    name: str
    meshes: tuple[str, ...]
    parent: str


@dataclass(frozen=True)
class MouthSample:
    servo_rad: float
    poses: dict[str, Pose]


@dataclass(frozen=True)
class MouthLinkage:
    schema_version: int
    closed_rad: float
    open_rad: float
    links: tuple[MouthLink, ...]
    samples: tuple[MouthSample, ...]
    validation_poses: tuple[MouthSample, ...]
    source_sha256: str


@dataclass(frozen=True)
class JointSpec:
    name: str
    parent_body: str
    child_body: str
    axis: tuple[float, float, float]
    range_rad: tuple[float, float]


@dataclass(frozen=True)
class BodySpec:
    name: str
    parent: str | None
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


@dataclass(frozen=True)
class RobotProfile:
    schema_version: int
    robot_id: str
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    home_positions: tuple[float, ...]
    joints: tuple[JointSpec, ...]
    bodies: tuple[BodySpec, ...]
    mouth: MouthLinkage
    source_sha256: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_to_json(profile: RobotProfile) -> str:
    return json.dumps(profile.to_dict(), sort_keys=True, separators=(",", ":"))


def profile_from_json(value: str) -> RobotProfile:
    try:
        payload = json.loads(value)
        mouth_payload = payload["mouth"]

        def make_sample(sample):
            return MouthSample(
                float(sample["servo_rad"]),
                {
                    name: Pose(
                        tuple(pose["position"]),
                        tuple(pose["quaternion_wxyz"]),
                    )
                    for name, pose in sample["poses"].items()
                },
            )

        mouth = MouthLinkage(
            int(mouth_payload["schema_version"]),
            float(mouth_payload["closed_rad"]),
            float(mouth_payload["open_rad"]),
            tuple(
                MouthLink(link["name"], tuple(link["meshes"]), link["parent"])
                for link in mouth_payload["links"]
            ),
            tuple(make_sample(sample) for sample in mouth_payload["samples"]),
            tuple(make_sample(sample) for sample in mouth_payload["validation_poses"]),
            mouth_payload["source_sha256"],
        )
        return RobotProfile(
            int(payload["schema_version"]),
            payload["robot_id"],
            tuple(payload["joint_names"]),
            tuple(payload["body_names"]),
            tuple(float(value) for value in payload["home_positions"]),
            tuple(
                JointSpec(
                    joint["name"],
                    joint["parent_body"],
                    joint["child_body"],
                    tuple(joint["axis"]),
                    tuple(joint["range_rad"]),
                )
                for joint in payload["joints"]
            ),
            tuple(
                BodySpec(
                    body["name"],
                    body["parent"],
                    tuple(body["position"]),
                    tuple(body["quaternion_wxyz"]),
                )
                for body in payload["bodies"]
            ),
            mouth,
            dict(payload["source_sha256"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ProfileError(f"embedded robot profile is malformed: {exc}") from exc


def _vector(value: str | None, size: int, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return default
    result = tuple(float(item) for item in value.split())
    if len(result) != size or not all(math.isfinite(item) for item in result):
        raise ProfileError(f"expected {size} finite values, got {value!r}")
    return result


def _unit_quaternion(value: Any, field: str) -> tuple[float, float, float, float]:
    try:
        quat = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{field} must be four finite numbers") from exc
    if len(quat) != 4 or not all(math.isfinite(component) for component in quat):
        raise ProfileError(f"{field} must be four finite numbers")
    norm = math.sqrt(sum(component * component for component in quat))
    if norm < 1e-12:
        raise ProfileError(f"{field} cannot be a zero quaternion")
    return tuple(component / norm for component in quat)


def _vector_from_json(value: Any, size: int, field: str) -> tuple[float, ...]:
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"{field} must be {size} finite numbers") from exc
    if len(result) != size or not all(math.isfinite(component) for component in result):
        raise ProfileError(f"{field} must be {size} finite numbers")
    return result


def _pose(payload: Any, field: str) -> Pose:
    if not isinstance(payload, dict):
        raise ProfileError(f"{field} must be an object")
    return Pose(
        _vector_from_json(payload.get("position"), 3, f"{field}.position"),
        _unit_quaternion(payload.get("quaternion_wxyz"), f"{field}.quaternion_wxyz"),
    )


def _samples(payload: Any, link_names: set[str], field: str) -> tuple[MouthSample, ...]:
    if not isinstance(payload, list) or not payload:
        raise ProfileError(f"{field} must contain at least one pose")
    result = []
    for index, sample in enumerate(payload):
        try:
            servo_rad = float(sample["servo_rad"])
            poses_payload = sample["poses"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError(f"{field}[{index}] is malformed") from exc
        if not math.isfinite(servo_rad):
            raise ProfileError(f"{field}[{index}].servo_rad must be finite")
        if set(poses_payload) != link_names:
            raise ProfileError(f"{field}[{index}].poses must cover every mouth link")
        result.append(
            MouthSample(
                servo_rad,
                {
                    name: _pose(pose, f"{field}[{index}].poses.{name}")
                    for name, pose in poses_payload.items()
                },
            )
        )
    return tuple(result)


def load_mouth_linkage(path: str | Path) -> MouthLinkage:
    path = Path(path)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"mouth linkage is not valid JSON: {exc}") from exc
    if payload.get("schema_version") != 1 or payload.get("units") != "m":
        raise ProfileError("mouth linkage requires schema_version 1 and units 'm'")
    servo = payload.get("servo", {})
    if servo.get("name") != "mouth":
        raise ProfileError("mouth linkage servo.name must be 'mouth'")
    try:
        closed_rad = float(servo["closed_rad"])
        open_rad = float(servo["open_rad"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileError("mouth linkage servo endpoints are required") from exc
    if not math.isclose(closed_rad, math.radians(-5), abs_tol=1e-9) or not math.isclose(
        open_rad, math.radians(30), abs_tol=1e-9
    ):
        raise ProfileError("mouth linkage endpoints must match runtime -5 and +30 degrees")
    links_payload = payload.get("links")
    if not isinstance(links_payload, list) or not links_payload:
        raise ProfileError("mouth linkage links must not be empty")
    try:
        links = tuple(
            MouthLink(
                str(link["name"]),
                tuple(str(mesh) for mesh in link["meshes"]),
                str(link["parent"]),
            )
            for link in links_payload
        )
    except (KeyError, TypeError) as exc:
        raise ProfileError("mouth linkage links are malformed") from exc
    link_names = {link.name for link in links}
    if len(link_names) != len(links) or any(not link.meshes for link in links):
        raise ProfileError("mouth linkage link names must be unique and own at least one mesh")
    samples = _samples(payload.get("samples"), link_names, "samples")
    angles = tuple(sample.servo_rad for sample in samples)
    if tuple(sorted(angles)) != angles or len(set(angles)) != len(angles):
        raise ProfileError("mouth linkage samples must be strictly increasing")
    if not math.isclose(angles[0], closed_rad, abs_tol=1e-9) or not math.isclose(
        angles[-1], open_rad, abs_tol=1e-9
    ):
        raise ProfileError("mouth linkage samples must include closed and open endpoints")
    validation_poses = _samples(
        payload.get("validation_poses"), link_names, "validation_poses"
    )
    return MouthLinkage(
        1,
        closed_rad,
        open_rad,
        links,
        samples,
        validation_poses,
        hashlib.sha256(raw).hexdigest(),
    )


def _rust_array(text: str, name: str) -> str:
    match = re.search(rf"(?:pub\s+)?const\s+{name}\s*:[^=]+?=\s*\[(.*?)\];", text, re.S)
    if not match:
        raise ProfileError(f"runtime source is missing {name}")
    return match.group(1)


def _runtime_contract(model_path: Path, joint_contract_path: Path | None) -> tuple[list[str], list[float]]:
    model_text = model_path.read_text()
    names_text = model_text
    if not re.search(r"const\s+JOINT_NAMES\s*:", names_text):
        if joint_contract_path is None:
            joint_contract_path = model_path.parents[2] / "duck-ipc-proto" / "src" / "lib.rs"
        if not joint_contract_path.exists():
            raise ProfileError("runtime source is missing JOINT_NAMES and no protocol contract was found")
        names_text = joint_contract_path.read_text()
    names = re.findall(r'"([^"\n]+)"', _rust_array(names_text, "JOINT_NAMES"))
    numeric = re.sub(r"//.*", "", _rust_array(model_text, "DEFAULT_POSITION"))
    try:
        positions = [float(token.strip()) for token in numeric.split(",") if token.strip()]
    except ValueError as exc:
        raise ProfileError("runtime DEFAULT_POSITION contains a non-literal value") from exc
    if len(names) != len(positions):
        raise ProfileError("runtime joint names and home positions have different lengths")
    return names, positions


def _mjcf_contract(path: Path) -> tuple[list[BodySpec], list[JointSpec]]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ProfileError(f"cannot parse MJCF: {exc}") from exc
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ProfileError("MJCF has no worldbody")
    bodies: list[BodySpec] = []
    joints: list[JointSpec] = []

    def walk(element: ET.Element, parent: str | None) -> None:
        name = element.get("name")
        if not name:
            raise ProfileError("MJCF contains an unnamed body")
        bodies.append(
            BodySpec(
                name,
                parent,
                _vector(element.get("pos"), 3, (0.0, 0.0, 0.0)),
                _unit_quaternion(
                    _vector(element.get("quat"), 4, (1.0, 0.0, 0.0, 0.0)),
                    f"body {name} quaternion",
                ),
            )
        )
        joint = element.find("joint")
        if joint is not None and joint.get("type") != "free":
            joint_name = joint.get("name")
            if not joint_name:
                raise ProfileError(f"body {name} has an unnamed joint")
            axis = _vector(joint.get("axis"), 3, (0.0, 0.0, 1.0))
            norm = math.sqrt(sum(component * component for component in axis))
            if norm < 1e-12:
                raise ProfileError(f"joint {joint_name} has a zero axis")
            joints.append(
                JointSpec(
                    joint_name,
                    parent or "",
                    name,
                    tuple(component / norm for component in axis),
                    _vector(joint.get("range"), 2, (-math.inf, math.inf)),
                )
            )
        for child in element.findall("body"):
            walk(child, name)

    for body in worldbody.findall("body"):
        walk(body, None)
    return bodies, joints


def build_microduck_profile(
    mjcf_path: str | Path,
    runtime_model_path: str | Path,
    mouth_linkage_path: str | Path,
    joint_contract_path: str | Path | None = None,
) -> RobotProfile:
    mjcf_path = Path(mjcf_path)
    runtime_model_path = Path(runtime_model_path)
    contract = Path(joint_contract_path) if joint_contract_path is not None else None
    runtime_names, runtime_home = _runtime_contract(runtime_model_path, contract)
    if runtime_names.count("mouth") != 1:
        raise ProfileError("runtime contract must contain mouth exactly once")
    policy_names = [name for name in runtime_names if name != "mouth"]
    home_positions = [
        position for name, position in zip(runtime_names, runtime_home) if name != "mouth"
    ]
    bodies, joints = _mjcf_contract(mjcf_path)
    mjcf_names = [joint.name for joint in joints]
    if policy_names != mjcf_names:
        raise ProfileError(
            f"runtime/MJCF joint order differs: runtime={policy_names}, mjcf={mjcf_names}"
        )
    mouth = load_mouth_linkage(mouth_linkage_path)
    hashes = {
        "mjcf": hashlib.sha256(mjcf_path.read_bytes()).hexdigest(),
        "runtime_model": hashlib.sha256(runtime_model_path.read_bytes()).hexdigest(),
        "mouth_linkage": mouth.source_sha256,
    }
    if contract is not None:
        hashes["joint_contract"] = hashlib.sha256(contract.read_bytes()).hexdigest()
    return RobotProfile(
        1,
        "microduck-alpha",
        tuple(policy_names),
        tuple(body.name for body in bodies),
        tuple(home_positions),
        tuple(joints),
        tuple(bodies),
        mouth,
        hashes,
    )
