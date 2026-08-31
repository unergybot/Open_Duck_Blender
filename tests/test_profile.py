from dataclasses import replace
import hashlib
import json
import math
import tempfile
import unittest
from unittest import mock
from pathlib import Path
import xml.etree.ElementTree as ET

from open_duck_tools.profile import (
    ProfileError,
    SiteSpec,
    build_microduck_profile,
    approximate_mouth_linkage,
    load_mouth_linkage,
    profile_from_json,
    profile_to_json,
)


MJCF = """
<mujoco model="microduck">
  <worldbody>
    <body name="trunk_base" pos="0 0 0.12">
      <body name="left_link" pos="0 0.1 0">
        <joint name="left_hip_yaw" axis="0 0 1" range="-0.4 0.5"/>
      </body>
      <body name="head_link" pos="0.02 0 0.03">
        <joint name="head_pitch" axis="0 0 1" range="-0.6 0.7"/>
      </body>
      <body name="right_link" pos="0 -0.1 0">
        <joint name="right_hip_yaw" axis="0 0 1" range="-0.5 0.4"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

RUNTIME_MODEL = """
pub const NUM_JOINTS: usize = 4;
pub const JOINT_NAMES: [&str; NUM_JOINTS] = [
    "left_hip_yaw", "head_pitch", "mouth", "right_hip_yaw",
];
pub const DEFAULT_POSITION: [f64; NUM_JOINTS] = [
    0.1, 0.2, 0.0, -0.1,
];
pub const MOUTH_CLOSED: f64 = -5.0 * std::f64::consts::PI / 180.0;
pub const MOUTH_OPEN: f64 = 30.0 * std::f64::consts::PI / 180.0;
"""

SITE_MJCF = """
<mujoco model="microduck-sites">
  <worldbody>
    <body name="trunk_base" pos="0 0 0.12">
      <body name="ankle_left" pos="0.01 0.02 -0.03">
        <joint name="left_ankle" axis="0 0 1" range="-1.5 1.5"/>
        <site name="left_foot" pos="0 -0.0238146 -0.0140852" quat="0 0 1 1"/>
      </body>
      <body name="ankle_right" pos="0.01 -0.02 -0.03">
        <joint name="right_ankle" axis="0 0 1" range="-1.5 1.5"/>
        <site name="right_foot" pos="0 -0.0238146 -0.0140852" quat="1 -1 0 0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

SITE_RUNTIME = """
pub const NUM_JOINTS: usize = 3;
pub const JOINT_NAMES: [&str; NUM_JOINTS] = [
    "left_ankle", "mouth", "right_ankle",
];
pub const DEFAULT_POSITION: [f64; NUM_JOINTS] = [0.0, 0.0, 0.0];
"""


def linkage_payload():
    return {
        "schema_version": 1,
        "units": "m",
        "servo": {
            "name": "mouth",
            "closed_rad": math.radians(-5),
            "open_rad": math.radians(30),
        },
        "links": [
            {"name": "lower_beak", "meshes": ["jaw"], "parent": "jaw_soft"}
        ],
        "samples": [
            {
                "servo_rad": math.radians(angle),
                "poses": {
                    "lower_beak": {
                        "position": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            }
            for angle in (-5, 12.5, 30)
        ],
        "validation_poses": [
            {
                "servo_rad": math.radians(3.25),
                "poses": {
                    "lower_beak": {
                        "position": [0.0, 0.0, 0.0],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    }
                },
            }
        ],
    }


class MouthLinkageTests(unittest.TestCase):
    def test_approximation_is_versioned_and_owns_lower_beak_meshes(self):
        linkage = approximate_mouth_linkage()
        self.assertEqual(linkage.links[0].meshes, ("jaw", "jaw_soft"))
        self.assertEqual(linkage.closed_rad, math.radians(-5))
        self.assertEqual(linkage.open_rad, math.radians(30))
        self.assertEqual(linkage.samples[0].poses[linkage.links[0].name].position, (0.018, 0.0, -0.018))
        self.assertLess(linkage.samples[-1].poses[linkage.links[0].name].quaternion_wxyz[1], 0.0)
        self.assertTrue(linkage.source_sha256)
    def test_rejects_linkage_without_validation_pose(self):
        payload = linkage_payload()
        payload["validation_poses"] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mouth.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ProfileError, "validation_poses"):
                load_mouth_linkage(path)

    def test_loads_complete_linkage_and_normalizes_quaternions(self):
        payload = linkage_payload()
        payload["samples"][1]["poses"]["lower_beak"]["quaternion_wxyz"] = [2, 0, 0, 0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mouth.json"
            path.write_text(json.dumps(payload))
            linkage = load_mouth_linkage(path)
        self.assertEqual(linkage.links[0].name, "lower_beak")
        self.assertEqual(linkage.samples[1].poses["lower_beak"].quaternion_wxyz, (1, 0, 0, 0))

    def test_rejects_validation_pose_that_disagrees_with_samples(self):
        payload = linkage_payload()
        payload["validation_poses"][0]["poses"]["lower_beak"]["position"][0] = 0.001
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mouth.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ProfileError, "disagrees with samples"):
                load_mouth_linkage(path)

    def test_rejects_undeclared_nested_fields(self):
        payload = linkage_payload()
        payload["servo"]["untrusted"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mouth.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ProfileError, "servo fields"):
                load_mouth_linkage(path)


class ProfileBuildTests(unittest.TestCase):
    def write_sources(self, root: Path, runtime=RUNTIME_MODEL):
        mjcf = root / "robot_walk.xml"
        model = root / "model.rs"
        mouth = root / "mouth.json"
        mjcf.write_text(MJCF)
        model.write_text(runtime)
        mouth.write_text(json.dumps(linkage_payload()))
        return mjcf, model, mouth

    def write_site_sources(self, root: Path, mjcf_text=SITE_MJCF):
        mjcf = root / "robot_walk.xml"
        model = root / "model.rs"
        mouth = root / "mouth.json"
        mjcf.write_text(mjcf_text)
        model.write_text(SITE_RUNTIME)
        mouth.write_text(json.dumps(linkage_payload()))
        return mjcf, model, mouth

    def test_builds_policy_order_by_removing_only_mouth(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_sources(Path(directory))
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=3, expected_body_count=4
            )
        self.assertEqual(
            profile.joint_names,
            ("left_hip_yaw", "head_pitch", "right_hip_yaw"),
        )
        self.assertEqual(profile.body_names, ("trunk_base", "left_link", "head_link", "right_link"))
        self.assertEqual(profile.home_positions, (0.1, 0.2, -0.1))
        self.assertEqual(profile.joints[0].range_rad, (-0.4, 0.5))
        self.assertEqual(profile.schema_version, 2)
        self.assertEqual(profile.sites, ())

    def test_parses_exact_normalized_physical_foot_sites_and_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_site_sources(Path(directory))
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=2, expected_body_count=3
            )
            restored = profile_from_json(profile_to_json(profile))

        self.assertEqual(
            tuple((site.name, site.parent_body, site.position) for site in profile.sites),
            (
                ("left_foot", "ankle_left", (0.0, -0.0238146, -0.0140852)),
                ("right_foot", "ankle_right", (0.0, -0.0238146, -0.0140852)),
            ),
        )
        expected_quaternions = (
            (0.0, 0.0, 2**-0.5, 2**-0.5),
            (2**-0.5, -(2**-0.5), 0.0, 0.0),
        )
        for site, expected in zip(profile.sites, expected_quaternions, strict=True):
            for actual_component, expected_component in zip(
                site.quaternion_wxyz, expected, strict=True
            ):
                self.assertAlmostEqual(actual_component, expected_component, places=15)
        self.assertEqual(restored, profile)

    def test_schema_one_embedded_profile_has_clear_physical_contract_error(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_site_sources(Path(directory))
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=2, expected_body_count=3
            )
        payload = json.loads(profile_to_json(replace(profile, schema_version=1)))
        with self.assertRaisesRegex(
            ProfileError, r"schema_version 1.*physical IK foot-site contract"
        ):
            profile_from_json(json.dumps(payload))

    def test_rejects_missing_duplicate_wrong_parent_and_invalid_foot_sites(self):
        cases = (
            (
                SITE_MJCF.replace(
                    '<site name="right_foot" pos="0 -0.0238146 -0.0140852" quat="1 -1 0 0"/>',
                    "",
                ),
                r"left_foot and right_foot exactly once",
            ),
            (
                SITE_MJCF.replace(
                    '<site name="left_foot"',
                    '<site name="left_foot" pos="0 0 0"/><site name="left_foot"',
                ),
                r"left_foot and right_foot exactly once",
            ),
            (
                SITE_MJCF.replace('body name="ankle_left"', 'body name="wrong_left"'),
                r"left_foot.*ankle_left",
            ),
            (
                SITE_MJCF.replace(
                    'pos="0 -0.0238146 -0.0140852" quat="0 0 1 1"',
                    'pos="nan 0 0" quat="0 0 1 1"',
                    1,
                ),
                r"expected 3 finite values",
            ),
            (
                SITE_MJCF.replace('quat="0 0 1 1"', 'quat="0 0 0 0"'),
                r"left_foot.*zero quaternion",
            ),
        )
        for index, (mjcf_text, message) in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                mjcf, runtime, mouth = self.write_site_sources(
                    Path(directory), mjcf_text
                )
                with self.assertRaisesRegex(ProfileError, message):
                    build_microduck_profile(
                        mjcf,
                        runtime,
                        mouth,
                        expected_joint_count=2,
                        expected_body_count=3,
                    )

    def test_profile_construction_parses_mjcf_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_site_sources(Path(directory))
            with mock.patch.object(ET, "parse", wraps=ET.parse) as parse:
                build_microduck_profile(
                    mjcf,
                    runtime,
                    mouth,
                    expected_joint_count=2,
                    expected_body_count=3,
                )
        self.assertEqual(parse.call_count, 1)

    def test_default_contract_rejects_non_microduck_cardinality(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_sources(Path(directory))
            with self.assertRaisesRegex(ProfileError, "14 policy joints and 15 bodies"):
                build_microduck_profile(mjcf, runtime, mouth)

    def test_rejects_runtime_and_mjcf_order_drift(self):
        drifted = RUNTIME_MODEL.replace(
            '"left_hip_yaw", "head_pitch", "mouth", "right_hip_yaw"',
            '"head_pitch", "left_hip_yaw", "mouth", "right_hip_yaw"',
        )
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_sources(Path(directory), drifted)
            with self.assertRaisesRegex(ProfileError, "joint order"):
                build_microduck_profile(
                    mjcf, runtime, mouth, expected_joint_count=3, expected_body_count=4
                )

    def test_profile_json_round_trip_preserves_typed_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, runtime, mouth = self.write_sources(Path(directory))
            original = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=3, expected_body_count=4
            )
            restored = profile_from_json(profile_to_json(original))
        self.assertEqual(restored, original)

    def test_hashes_each_referenced_stl_and_a_sorted_visual_asset_aggregate(self):
        asset_mjcf = MJCF.replace(
            "<worldbody>",
            '<compiler meshdir="assets"/><asset>'
            '<mesh name="part_b" file="part-b.stl"/>'
            '<mesh name="part_a" file="part-a.stl"/>'
            '</asset><worldbody>',
        ).replace(
            '<body name="trunk_base" pos="0 0 0.12">',
            '<body name="trunk_base" pos="0 0 0.12">'
            '<geom type="mesh" class="visual" mesh="part_b"/>'
            '<geom type="mesh" class="visual" mesh="part_a"/>'
            '<geom type="mesh" class="visual" mesh="part_b"/>',
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assets = root / "assets"
            assets.mkdir()
            part_a = b"canonical-a"
            part_b = b"canonical-b"
            (assets / "part-a.stl").write_bytes(part_a)
            (assets / "part-b.stl").write_bytes(part_b)
            mjcf, runtime, mouth = self.write_sources(root)
            mjcf.write_text(asset_mjcf)
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=3, expected_body_count=4
            )

        per_asset = {
            "part-a.stl": hashlib.sha256(part_a).hexdigest(),
            "part-b.stl": hashlib.sha256(part_b).hexdigest(),
        }
        aggregate = hashlib.sha256(
            json.dumps(per_asset, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(profile.source_sha256["stl:part-a.stl"], per_asset["part-a.stl"])
        self.assertEqual(profile.source_sha256["stl:part-b.stl"], per_asset["part-b.stl"])
        self.assertEqual(profile.source_sha256["visual_assets"], aggregate)


if __name__ == "__main__":
    unittest.main()
