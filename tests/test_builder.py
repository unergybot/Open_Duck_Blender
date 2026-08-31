import hashlib
import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

from open_duck_tools.builder import generate_microduck_scene
from open_duck_tools.motion import MotionError, build_motion_archive
from open_duck_tools import addon
from open_duck_tools.profile import ProfileError, build_microduck_profile


MJCF = """
<mujoco model="microduck">
  <compiler meshdir="assets"/>
  <worldbody>
    <body name="trunk_base" pos="0 0 0.12">
      <body name="child" pos="0.1 0 0">
        <joint name="hinge" axis="0 0 1" range="-1 1"/>
      </body>
      <body name="jaw_soft" pos="0 0 0.1">
        <geom type="mesh" class="visual" mesh="jaw" material="jaw_material"/>
      </body>
    </body>
  </worldbody>
  <asset>
    <mesh name="jaw" file="jaw.stl"/>
    <material name="jaw_material" rgba="1 0.7 0 1"/>
  </asset>
</mujoco>
"""

RUNTIME = """
pub const JOINT_NAMES: [&str; 2] = ["hinge", "mouth"];
pub const DEFAULT_POSITION: [f64; 2] = [0.0, 0.0];
"""

CANONICAL_ROBOT_ROOT = (
    Path.home() / "MyCode/microduck_rl/src/mjlab_microduck/robot/microduck"
)
CANONICAL_MJCF = CANONICAL_ROBOT_ROOT / "robot_walk.xml"
CANONICAL_RUNTIME = Path.home() / "MyCode/microduck/duck-control/src/model.rs"
CANONICAL_CONTRACT = Path.home() / "MyCode/microduck/duck-ipc-proto/src/lib.rs"


def mouth_payload():
    identity = {
        "position": [0, 0, 0],
        "quaternion_wxyz": [1, 0, 0, 0],
    }
    return {
        "schema_version": 1,
        "units": "m",
        "servo": {
            "name": "mouth",
            "closed_rad": math.radians(-5),
            "open_rad": math.radians(30),
        },
        "links": [{"name": "lower_beak", "meshes": ["jaw"], "parent": "jaw_soft"}],
        "samples": [
            {"servo_rad": math.radians(-5), "poses": {"lower_beak": identity}},
            {
                "servo_rad": math.radians(30),
                "poses": {
                    "lower_beak": {
                        "position": [0.01, 0, 0],
                        "quaternion_wxyz": [1, 0, 0, 0],
                    }
                },
            },
        ],
        "validation_poses": [
            {
                "servo_rad": math.radians(12.5),
                "poses": {
                    "lower_beak": {
                        "position": [0.005, 0, 0],
                        "quaternion_wxyz": [1, 0, 0, 0],
                    }
                },
            }
        ],
    }


def write_binary_stl(path: Path):
    record = struct.pack(
        "<12fH",
        0,
        0,
        1,
        0,
        0,
        0,
        0.01,
        0,
        0,
        0,
        0.01,
        0,
        0,
    )
    path.write_bytes(bytes(80) + struct.pack("<I", 1) + record)


def assert_matrix_almost_equal(test_case, actual, expected, places=6):
    for row in range(4):
        for column in range(4):
            test_case.assertAlmostEqual(
                actual[row][column],
                expected[row][column],
                places=places,
                msg=f"matrix differs at [{row}][{column}]",
            )


def write_builder_sources(root: Path, mjcf_text=MJCF, linkage=None):
    (root / "assets").mkdir()
    mjcf = root / "robot_walk.xml"
    runtime = root / "model.rs"
    mouth = root / "mouth.json"
    mjcf.write_text(mjcf_text)
    runtime.write_text(RUNTIME)
    mouth.write_text(json.dumps(linkage or mouth_payload()))
    write_binary_stl(root / "assets" / "jaw.stl")
    profile = build_microduck_profile(
        mjcf, runtime, mouth, expected_joint_count=1, expected_body_count=3
    )
    return mjcf, profile


class SceneBuilderTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)

    def tearDown(self):
        addon.unregister()

    def test_builds_profiled_rig_visual_and_driven_mouth_link(self):
        bpy.context.scene.render.fps_base = 1.25
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            mjcf = root / "robot_walk.xml"
            runtime = root / "model.rs"
            mouth = root / "mouth.json"
            mjcf.write_text(MJCF)
            runtime.write_text(RUNTIME)
            mouth.write_text(json.dumps(mouth_payload()))
            write_binary_stl(root / "assets" / "jaw.stl")
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=1, expected_body_count=3
            )
            armature = generate_microduck_scene(profile, mjcf, Path("open_duck_tools"))

        self.assertEqual(armature.get("duck_robot_id"), "microduck-alpha")
        self.assertEqual(bpy.context.scene.render.fps, 50)
        self.assertEqual(bpy.context.scene.render.fps_base, 1.0)
        self.assertEqual(
            set(profile.body_names),
            {bone.name for bone in armature.data.bones if not bone.name.startswith("mouth::")},
        )
        self.assertIn("mouth::lower_beak", armature.data.bones)
        self.assertIsNotNone(bpy.data.objects.get("visual::jaw"))
        self.assertTrue(armature.data.get("duck_robot_profile_json"))
        bootstrap = bpy.data.texts.get("open_duck_bootstrap.py")
        self.assertIsNotNone(bootstrap)
        self.assertTrue(bootstrap.use_module)
        self.assertEqual(bootstrap.as_string().rstrip().splitlines()[-1], "register()")
        addon.unregister()
        first_bootstrap = {}
        exec(
            compile(bootstrap.as_string(), "<test-open-duck-bootstrap>", "exec"),
            first_bootstrap,
        )
        first_addon = sys.modules["open_duck_tools_embedded.addon"]
        second_bootstrap = {}
        try:
            exec(
                compile(bootstrap.as_string(), "<test-open-duck-bootstrap>", "exec"),
                second_bootstrap,
            )
            second_addon = sys.modules["open_duck_tools_embedded.addon"]
            self.assertIs(bpy.types.DUCK_PT_tools, second_addon.DUCK_PT_tools)
            self.assertTrue(hasattr(bpy.types.Object, "duck_action_name"))
            self.assertTrue(hasattr(bpy.types, "DUCK_OT_toggle_animation"))
        finally:
            registered = getattr(bpy.types, "DUCK_PT_tools", None)
            current_addon = sys.modules.get("open_duck_tools_embedded.addon")
            if current_addon and registered is current_addon.DUCK_PT_tools:
                current_addon.unregister()
            elif registered is first_addon.DUCK_PT_tools:
                first_addon.unregister()
        addon.register()
        armature.duck_mouth_open = 1.0
        bpy.context.view_layer.update()
        self.assertAlmostEqual(
            armature.pose.bones["mouth::lower_beak"].matrix_basis.to_translation().x,
            0.01,
            places=6,
        )

    def test_rest_bones_keep_exact_world_orientation_when_created(self):
        oriented_mjcf = MJCF.replace(
            '<body name="child" pos="0.1 0 0">',
            '<body name="child" pos="0.1 0 0" quat="0.7071067811865476 0.7071067811865476 0 0">',
        )
        linkage = mouth_payload()
        for sample in (*linkage["samples"], *linkage["validation_poses"]):
            sample["poses"]["lower_beak"]["quaternion_wxyz"] = [
                0.7071067811865476,
                0.0,
                0.7071067811865476,
                0.0,
            ]
        with tempfile.TemporaryDirectory() as directory:
            mjcf, profile = write_builder_sources(
                Path(directory), oriented_mjcf, linkage
            )
            armature = generate_microduck_scene(
                profile, mjcf, Path("open_duck_tools")
            )

        trunk = Matrix.Translation((0.0, 0.0, 0.12))
        child = trunk @ Matrix.Translation((0.1, 0.0, 0.0))
        child @= Quaternion(
            (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
        ).to_matrix().to_4x4()
        jaw_body = trunk @ Matrix.Translation((0.0, 0.0, 0.1))
        mouth = jaw_body @ Quaternion(
            (0.7071067811865476, 0.0, 0.7071067811865476, 0.0)
        ).to_matrix().to_4x4()
        assert_matrix_almost_equal(self, armature.data.bones["child"].matrix_local, child)
        assert_matrix_almost_equal(
            self, armature.data.bones["mouth::lower_beak"].matrix_local, mouth
        )
        self.assertAlmostEqual(armature.data.bones["child"].length, 0.015, places=6)
        self.assertAlmostEqual(
            armature.data.bones["mouth::lower_beak"].length, 0.01, places=6
        )

    def test_bone_parenting_preserves_closed_world_matrix_and_moves_with_mouth(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, profile = write_builder_sources(Path(directory))
            armature = generate_microduck_scene(
                profile, mjcf, Path("open_duck_tools")
            )

        visual = bpy.data.objects["visual::jaw"]
        bpy.context.view_layer.update()
        closed = Matrix.Translation((0.0, 0.0, 0.22))
        assert_matrix_almost_equal(self, visual.matrix_world, closed)

        armature.duck_mouth_open = 1.0
        bpy.context.view_layer.update()
        opened = Matrix.Translation((0.01, 0.0, 0.22))
        assert_matrix_almost_equal(self, visual.matrix_world, opened)

    def test_shared_mesh_material_combinations_have_one_effective_slot(self):
        duplicated = MJCF.replace(
            '<geom type="mesh" class="visual" mesh="jaw" material="jaw_material"/>',
            '<geom type="mesh" class="visual" mesh="jaw" material="jaw_material"/>'
            '<geom type="mesh" class="visual" mesh="jaw" material="jaw_material"/>'
            '<geom type="mesh" class="visual" mesh="jaw" material="jaw_secondary"/>',
        ).replace(
            '<material name="jaw_material" rgba="1 0.7 0 1"/>',
            '<material name="jaw_material" rgba="1 0.7 0 1"/>'
            '<material name="jaw_secondary" rgba="0.2 0.3 0.4 1"/>',
        )
        with tempfile.TemporaryDirectory() as directory:
            mjcf, profile = write_builder_sources(Path(directory), duplicated)
            generate_microduck_scene(profile, mjcf, Path("open_duck_tools"))

        first = bpy.data.objects["visual::jaw"]
        second = bpy.data.objects["visual::jaw::jaw_soft::1"]
        third = bpy.data.objects["visual::jaw::jaw_soft::2"]
        self.assertIs(first.data, second.data)
        self.assertIsNot(first.data, third.data)
        self.assertEqual([slot.name for slot in first.data.materials], ["jaw_material"])
        self.assertEqual([slot.name for slot in third.data.materials], ["jaw_secondary"])

    def test_initial_cream_colorway_is_applied_after_material_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, profile = write_builder_sources(Path(directory))
            armature = generate_microduck_scene(
                profile, mjcf, Path("open_duck_tools")
            )

        expected_trim = (0.9130986518, 0.3419144249, 0.0033465358, 1.0)
        material = bpy.data.materials["jaw_material"]
        self.assertEqual(armature.duck_colorway, "CREAM")
        self.assertEqual(set(addon.COLORWAYS), {"CREAM", "GRAPHITE", "LAVENDER", "SKY"})
        for actual, expected in zip(material.diffuse_color, expected_trim):
            self.assertAlmostEqual(actual, expected, places=6)
        node_color = material.node_tree.nodes["Principled BSDF"].inputs[
            "Base Color"
        ].default_value
        for actual, expected in zip(node_color, expected_trim):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_colorway_callback_does_not_collide_with_enum_backing_property(self):
        with tempfile.TemporaryDirectory() as directory:
            mjcf, profile = write_builder_sources(Path(directory))
            armature = generate_microduck_scene(
                profile, mjcf, Path("open_duck_tools")
            )

        for key in ("CREAM", "GRAPHITE", "LAVENDER", "SKY"):
            with self.subTest(colorway=key):
                armature.duck_colorway = key
                addon._colorway_updated(armature, None)
                self.assertEqual(armature.duck_colorway, key)

    def test_rejects_non_finite_and_zero_visual_geom_transforms(self):
        invalid_values = (
            ('pos="nan 0 0"', "visual geom jaw position"),
            ('quat="0 0 0 0"', "visual geom jaw quaternion"),
            ('quat="nan 0 0 0"', "visual geom jaw quaternion"),
        )
        for attribute, message in invalid_values:
            with self.subTest(attribute=attribute), tempfile.TemporaryDirectory() as directory:
                invalid = MJCF.replace(
                    '<geom type="mesh" class="visual" mesh="jaw" material="jaw_material"/>',
                    f'<geom type="mesh" class="visual" mesh="jaw" '
                    f'material="jaw_material" {attribute}/>',
                )
                mjcf, profile = write_builder_sources(Path(directory), invalid)
                with self.assertRaisesRegex(ProfileError, message):
                    generate_microduck_scene(
                        profile, mjcf, Path("open_duck_tools")
                    )

    def test_normalizes_visual_geom_quaternion_before_building_matrix(self):
        normalized = MJCF.replace(
            '<geom type="mesh" class="visual" mesh="jaw" material="jaw_material"/>',
            '<geom type="mesh" class="visual" mesh="jaw" material="jaw_material" '
            'pos="0.02 0.03 0.04" quat="2 2 0 0"/>',
        )
        with tempfile.TemporaryDirectory() as directory:
            mjcf, profile = write_builder_sources(Path(directory), normalized)
            generate_microduck_scene(profile, mjcf, Path("open_duck_tools"))

        expected = Matrix.Translation((0.0, 0.0, 0.22))
        expected @= Matrix.Translation((0.02, 0.03, 0.04))
        expected @= Quaternion(
            (0.7071067811865476, 0.7071067811865476, 0.0, 0.0)
        ).to_matrix().to_4x4()
        bpy.context.view_layer.update()
        assert_matrix_almost_equal(
            self, bpy.data.objects["visual::jaw"].matrix_world, expected
        )

    @unittest.skipUnless(
        all(
            path.is_file()
            for path in (CANONICAL_MJCF, CANONICAL_RUNTIME, CANONICAL_CONTRACT)
        ),
        "canonical Microduck source checkouts are unavailable",
    )
    def test_canonical_scene_preserves_all_rest_transforms_visuals_and_stls(self):
        profile = build_microduck_profile(
            CANONICAL_MJCF,
            CANONICAL_RUNTIME,
            None,
            joint_contract_path=CANONICAL_CONTRACT,
        )
        armature = generate_microduck_scene(
            profile, CANONICAL_MJCF, Path("open_duck_tools")
        )
        root = ET.parse(CANONICAL_MJCF).getroot()

        rest = {}
        for body in profile.bodies:
            local = Matrix.Translation(body.position) @ Quaternion(
                body.quaternion_wxyz
            ).to_matrix().to_4x4()
            rest[body.name] = local if body.parent is None else rest[body.parent] @ local
            assert_matrix_almost_equal(
                self, armature.data.bones[body.name].matrix_local, rest[body.name]
            )

        visual_specs = []

        def collect(body):
            body_name = body.get("name")
            for geom in body.findall("geom"):
                if geom.get("mesh") and geom.get("class") in (None, "visual"):
                    position = tuple(
                        float(value) for value in geom.get("pos", "0 0 0").split()
                    )
                    quaternion = tuple(
                        float(value) for value in geom.get("quat", "1 0 0 0").split()
                    )
                    norm = math.sqrt(sum(value * value for value in quaternion))
                    local = Matrix.Translation(position) @ Quaternion(
                        tuple(value / norm for value in quaternion)
                    ).to_matrix().to_4x4()
                    visual_specs.append((body_name, geom.get("mesh"), local))
            for child in body.findall("body"):
                collect(child)

        for top in root.find("worldbody").findall("body"):
            collect(top)
        visuals = [obj for obj in bpy.data.objects if obj.name.startswith("visual::")]
        self.assertEqual((len(profile.bodies), len(visual_specs), len(visuals)), (15, 70, 70))
        bpy.context.view_layer.update()
        objects_by_mesh = {}
        for obj in visuals:
            mesh_name = obj.data.name.removeprefix("mesh::").split("::", 1)[0]
            objects_by_mesh.setdefault(mesh_name, []).append(obj)
            self.assertLessEqual(len(obj.data.materials), 1)
        expected_by_mesh = {}
        for body_name, mesh_name, local in visual_specs:
            expected_by_mesh.setdefault(mesh_name, []).append(
                armature.matrix_world @ armature.pose.bones[body_name].matrix @ local
            )
        for mesh_name, objects in objects_by_mesh.items():
            remaining = expected_by_mesh[mesh_name].copy()
            for obj in objects:
                differences = [
                    max(
                        abs(obj.matrix_world[row][column] - expected[row][column])
                        for row in range(4)
                        for column in range(4)
                    )
                    for expected in remaining
                ]
                match = min(range(len(remaining)), key=differences.__getitem__)
                with self.subTest(visual=obj.name, mesh=mesh_name):
                    self.assertLess(differences[match], 1e-5)
                remaining.pop(match)
            self.assertFalse(remaining)

        asset = root.find("asset")
        compiler = root.find("compiler")
        mesh_dir = CANONICAL_MJCF.parent / compiler.get("meshdir", ".")
        assets = {}
        for item in asset.findall("mesh"):
            filename = item.get("file") or f"{item.get('name')}.stl"
            assets[item.get("name") or Path(filename).stem] = filename
        referenced = sorted({mesh_name for _, mesh_name, _ in visual_specs})
        self.assertEqual((len(referenced), len(bpy.data.meshes)), (38, 38))
        per_asset = {}
        for mesh_name in referenced:
            filename = assets[mesh_name]
            raw = (mesh_dir / filename).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            per_asset[Path(filename).as_posix()] = digest
            self.assertEqual(profile.source_sha256[f"stl:{Path(filename).as_posix()}"], digest)
            triangle_count = struct.unpack_from("<I", raw, 80)[0]
            expected_vertices = []
            for index in range(triangle_count):
                triangle = struct.unpack_from("<12fH", raw, 84 + index * 50)
                expected_vertices.extend(
                    (triangle[3:6], triangle[6:9], triangle[9:12])
                )
            actual_vertices = [
                tuple(vertex.co)
                for vertex in objects_by_mesh[mesh_name][0].data.vertices
            ]
            self.assertEqual(actual_vertices, expected_vertices)
        aggregate = hashlib.sha256(
            json.dumps(per_asset, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(profile.source_sha256["visual_assets"], aggregate)

        armature.duck_mouth_open = 1.0
        bpy.context.view_layer.update()
        link = profile.mouth.links[0]
        closed_pose = profile.mouth.samples[0].poses[link.name]
        open_pose = profile.mouth.samples[-1].poses[link.name]
        closed = Matrix.Translation(closed_pose.position) @ Quaternion(
            closed_pose.quaternion_wxyz
        ).to_matrix().to_4x4()
        opened = Matrix.Translation(open_pose.position) @ Quaternion(
            open_pose.quaternion_wxyz
        ).to_matrix().to_4x4()
        mouth_meshes = set(link.meshes)
        expected_by_mesh = {}
        for body_name, mesh_name, local in visual_specs:
            body_world = armature.matrix_world @ armature.pose.bones[body_name].matrix
            expected = (
                body_world @ opened @ closed.inverted() @ local
                if mesh_name in mouth_meshes
                else body_world @ local
            )
            expected_by_mesh.setdefault(mesh_name, []).append(expected)
        for mesh_name, objects in objects_by_mesh.items():
            remaining = expected_by_mesh[mesh_name].copy()
            for obj in objects:
                differences = [
                    max(
                        abs(obj.matrix_world[row][column] - expected[row][column])
                        for row in range(4)
                        for column in range(4)
                    )
                    for expected in remaining
                ]
                match = min(range(len(remaining)), key=differences.__getitem__)
                with self.subTest(open_visual=obj.name, mesh=mesh_name):
                    self.assertLess(differences[match], 1e-5)
                remaining.pop(match)
            self.assertFalse(remaining)

    def test_builds_51_frame_demo_action_from_native_motion_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            mjcf = root / "robot_walk.xml"
            runtime = root / "model.rs"
            mouth = root / "mouth.json"
            motion = root / "demo.npz"
            mjcf.write_text(MJCF)
            runtime.write_text(RUNTIME)
            mouth.write_text(json.dumps(mouth_payload()))
            write_binary_stl(root / "assets" / "jaw.stl")
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=1, expected_body_count=3
            )
            joint_pos = np.zeros((51, 1), dtype=np.float32)
            joint_pos[25, 0] = 0.4
            body_pos = np.zeros((51, 3, 3), dtype=np.float32)
            body_pos[:, :, 2] = 0.12
            body_pos[:, 0, 0] = np.linspace(0.0, 0.5, 51)
            archive = build_motion_archive(
                joint_pos,
                body_pos,
                np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (51, 3, 1)),
                fps=50,
                joint_names=profile.joint_names,
                body_names=profile.body_names,
                joint_ranges=tuple(joint.range_rad for joint in profile.joints),
                source_hashes={"fixture": "deadbeef"},
            )
            np.savez_compressed(motion, **archive)
            armature = generate_microduck_scene(
                profile,
                mjcf,
                Path("open_duck_tools"),
                demo_motion_path=motion,
            )

        self.assertEqual((bpy.context.scene.frame_start, bpy.context.scene.frame_end), (1, 51))
        self.assertIsNotNone(armature.animation_data)
        self.assertEqual(armature.animation_data.action.name, "MicroduckCrouchTest")
        bpy.context.scene.frame_set(26)
        self.assertAlmostEqual(armature.pose.bones["child"].rotation_euler.z, 0.4, places=6)
        self.assertAlmostEqual(armature.duck_mouth_open, 1.0, places=6)
        bpy.context.scene.frame_set(51)
        self.assertAlmostEqual(armature.location.x, 0.5, places=6)

    def test_rejects_malformed_demo_archives_with_motion_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "assets").mkdir()
            mjcf = root / "robot_walk.xml"
            runtime = root / "model.rs"
            mouth = root / "mouth.json"
            mjcf.write_text(MJCF)
            runtime.write_text(RUNTIME)
            mouth.write_text(json.dumps(mouth_payload()))
            write_binary_stl(root / "assets" / "jaw.stl")
            profile = build_microduck_profile(
                mjcf, runtime, mouth, expected_joint_count=1, expected_body_count=3
            )
            malformed = (
                {"joint_names": np.array(["hinge"]), "joint_pos": np.array([[0.0]])},
                {
                    "fps": np.array([50]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array(0.0),
                },
                {
                    "fps": np.array([50.9]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
                {
                    "fps": np.array(["bad"]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
                {
                    "fps": np.array([np.nan]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
                {
                    "fps": np.array([np.inf]),
                    "joint_names": np.array(["hinge"]),
                    "joint_pos": np.array([[0.0]]),
                },
            )
            for index, payload in enumerate(malformed):
                motion = root / f"malformed-{index}.npz"
                np.savez_compressed(motion, **payload)
                with self.subTest(index=index), self.assertRaisesRegex(
                    MotionError, "archive|fps"
                ):
                    generate_microduck_scene(
                        profile,
                        mjcf,
                        Path("open_duck_tools"),
                        demo_motion_path=motion,
                    )


if __name__ == "__main__":
    unittest.main()
