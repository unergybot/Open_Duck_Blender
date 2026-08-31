"""Tests for strict, side-effect-free mjlab motion archive loading."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bpy
import numpy as np
from mathutils import Matrix, Quaternion

from open_duck_tools import ImportedMotion
from open_duck_tools.motion import MotionError, build_motion_archive
from open_duck_tools.motion_import import import_motion_action, load_motion
from open_duck_tools.profile import (
    BodySpec,
    JointSpec,
    MouthLinkage,
    RobotProfile,
)


def test_profile() -> RobotProfile:
    """A compact profile whose root has a non-identity MJCF rest transform."""
    return RobotProfile(
        schema_version=1,
        robot_id="test-duck",
        joint_names=("hip", "knee"),
        body_names=("root", "hip_link", "knee_link"),
        home_positions=(0.0, 0.0),
        joints=(
            JointSpec("hip", "root", "hip_link", (0.0, 0.0, 1.0), (-1.0, 1.0)),
            JointSpec("knee", "hip_link", "knee_link", (0.0, 0.0, 1.0), (-0.5, 0.5)),
        ),
        bodies=(
            BodySpec("root", None, (0.0, 0.0, 0.125), (1.0, 0.0, 0.0, 0.0)),
            BodySpec("hip_link", "root", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0)),
            BodySpec("knee_link", "hip_link", (0.0, 0.0, 0.1), (1.0, 0.0, 0.0, 0.0)),
        ),
        mouth=MouthLinkage(1, 0.0, 1.0, (), (), (), "test"),
        source_sha256={},
    )


def archive_payload(profile: RobotProfile) -> dict[str, np.ndarray]:
    joint_pos = np.array([[0.0, 0.0], [0.25, -0.25], [0.5, 0.25]])
    root_positions = np.array(
        [[1.0, 2.0, 3.0], [1.5, 2.0, 3.0], [2.0, 2.0, 3.0]]
    )
    root_angles = (0.0, math.pi / 2.0, math.pi)

    def z_quaternion(angle: float) -> np.ndarray:
        return np.array([math.cos(angle / 2.0), 0.0, 0.0, math.sin(angle / 2.0)])

    positions = []
    quaternions = []
    for frame, (root_position, root_angle) in enumerate(
        zip(root_positions, root_angles)
    ):
        by_name_position = {
            "root": root_position,
            "hip_link": root_position + (0.0, 0.0, 0.1),
            "knee_link": root_position + (0.0, 0.0, 0.2),
        }
        hip_angle = root_angle + joint_pos[frame, 0]
        by_name_quaternion = {
            "root": z_quaternion(root_angle),
            "hip_link": z_quaternion(hip_angle),
            "knee_link": z_quaternion(hip_angle + joint_pos[frame, 1]),
        }
        positions.append([by_name_position[name] for name in profile.body_names])
        quaternions.append([by_name_quaternion[name] for name in profile.body_names])

    archive = build_motion_archive(
        joint_pos=joint_pos,
        body_pos_w=np.asarray(positions),
        body_quat_w=np.asarray(quaternions),
        fps=50,
        joint_names=profile.joint_names,
        body_names=profile.body_names,
        joint_ranges=tuple(joint.range_rad for joint in profile.joints),
        source_hashes={"fixture": "deadbeef"},
    )
    # The negative final quaternions represent the same rotations; the loader
    # must normalize them and choose interpolation-continuous signs per body.
    archive["body_quat_w"][2] *= -1
    return archive


class MotionLoaderTests(unittest.TestCase):
    def write_archive(self, directory: Path, payload: dict[str, np.ndarray]) -> Path:
        path = directory / "motion.npz"
        np.savez_compressed(path, **payload)
        return path

    def test_loads_complete_native_archive_with_full_normalized_body_state_and_exact_hash(self):
        profile = test_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(Path(directory), archive_payload(profile))
            expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            motion = load_motion(path, profile)

        self.assertIsInstance(motion, ImportedMotion)
        self.assertEqual(motion.fps, 50)
        self.assertEqual(motion.frames, 3)
        self.assertEqual(motion.joint_pos.shape, (3, 2))
        self.assertTrue(hasattr(motion, "body_pos_w"))
        self.assertTrue(hasattr(motion, "body_quat_w"))
        self.assertTrue(hasattr(motion, "source_sha256"))
        self.assertEqual(motion.body_pos_w.shape, (3, 3, 3))
        self.assertEqual(motion.body_quat_w.shape, (3, 3, 4))
        self.assertEqual(motion.root_pos_w.shape, (3, 3))
        self.assertEqual(motion.root_quat_wxyz.shape, (3, 4))
        self.assertEqual(motion.source_sha256, expected_sha256)
        self.assertTrue(np.isfinite(motion.joint_pos).all())
        self.assertTrue(np.isfinite(motion.body_pos_w).all())
        self.assertTrue(np.isfinite(motion.body_quat_w).all())
        np.testing.assert_allclose(np.linalg.norm(motion.body_quat_w, axis=2), 1.0)
        self.assertTrue(
            np.all(
                np.sum(
                    motion.body_quat_w[1:] * motion.body_quat_w[:-1], axis=2
                )
                >= 0
            )
        )
        np.testing.assert_allclose(motion.joint_pos[:, 1], [0.0, -0.25, 0.25])

    def test_uses_profile_root_name_when_body_order_does_not_start_with_root(self):
        profile = replace(
            test_profile(), body_names=("hip_link", "root", "knee_link")
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(Path(directory), archive_payload(profile))
            motion = load_motion(path, profile)

        np.testing.assert_allclose(
            motion.root_pos_w,
            [[1.0, 2.0, 3.0], [1.5, 2.0, 3.0], [2.0, 2.0, 3.0]],
        )

    def test_rejects_body_translation_inconsistent_with_root_and_joint_fk(self):
        profile = test_profile()
        payload = archive_payload(profile)
        payload["body_pos_w"][1, 2, 0] += 2e-4

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                MotionError,
                r"frame 1.*knee_link.*position residual.*0\.0002.*rotation residual",
            ):
                load_motion(self.write_archive(Path(directory), payload), profile)

    def test_rejects_body_orientation_inconsistent_with_root_and_joint_fk(self):
        profile = test_profile()
        payload = archive_payload(profile)
        archived = Quaternion(tuple(payload["body_quat_w"][1, 1]))
        drifted = archived @ Quaternion((math.cos(1e-3), math.sin(1e-3), 0.0, 0.0))
        payload["body_quat_w"][1, 1] = tuple(drifted)

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                MotionError,
                r"frame 1.*hip_link.*position residual.*rotation residual.*0\.002",
            ):
                load_motion(self.write_archive(Path(directory), payload), profile)

    def test_rejects_each_native_schema_or_sample_contract_violation(self):
        profile = test_profile()
        cases = []
        missing = archive_payload(profile)
        missing.pop("body_ang_vel_w")
        cases.append((missing, r"archive keys.*body_ang_vel_w"))
        extra = archive_payload(profile)
        extra["surprise"] = np.array([1])
        cases.append((extra, r"archive keys.*surprise"))
        names = archive_payload(profile)
        names["joint_names"] = np.array(["wrong", "knee"])
        cases.append((names, r"joint_names.*index 0.*wrong.*hip"))
        body_names = archive_payload(profile)
        body_names["body_names"] = np.array(["root", "wrong", "knee_link"])
        cases.append((body_names, r"body_names.*index 1.*wrong.*hip_link"))
        nonfinite = archive_payload(profile)
        nonfinite["body_pos_w"][1, 2, 0] = np.nan
        cases.append((nonfinite, r"body_pos_w.*frame 1.*index 2"))
        zero = archive_payload(profile)
        zero["body_quat_w"][2, 0] = 0
        cases.append((zero, r"body_quat_w.*frame 2.*index 0"))
        nonunit = archive_payload(profile)
        nonunit["body_quat_w"][1, 2] = (2.0, 0.0, 0.0, 0.0)
        cases.append((nonunit, r"body_quat_w.*frame 1.*index 2.*unit"))
        over_limit = archive_payload(profile)
        over_limit["joint_pos"][1, 1] = 0.75
        cases.append((over_limit, r"joint_pos.*frame 1.*index 1.*knee"))
        scalar_joint_pos = archive_payload(profile)
        scalar_joint_pos["joint_pos"] = np.array(0.0)
        cases.append((scalar_joint_pos, r"joint_pos.*shape"))
        vector_joint_pos = archive_payload(profile)
        vector_joint_pos["joint_pos"] = np.array([0.0, 0.0])
        cases.append((vector_joint_pos, r"joint_pos.*shape"))
        matrix_joint_names = archive_payload(profile)
        matrix_joint_names["joint_names"] = np.array([["hip", "knee"]])
        cases.append((matrix_joint_names, r"joint_names.*one-dimensional"))

        with tempfile.TemporaryDirectory() as directory:
            for index, (payload, message) in enumerate(cases):
                with self.subTest(index=index), self.assertRaisesRegex(MotionError, message):
                    load_motion(self.write_archive(Path(directory), payload), profile)

    def test_translates_corrupt_npz_to_motion_error(self):
        profile = test_profile()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corrupt.npz"
            path.write_bytes(b"PK\x03\x04truncated")

            with self.assertRaisesRegex(MotionError, "could not load motion archive"):
                load_motion(path, profile)


if __name__ == "__main__":
    unittest.main()


def build_minimal_rig(profile: RobotProfile):
    data = bpy.data.armatures.new("TestRig")
    armature = bpy.data.objects.new("TestRig", data)
    bpy.context.scene.collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bones = {}
    world = Matrix.Identity(4)
    for body in profile.bodies:
        world = Matrix.Translation(body.position) if body.parent is None else world @ Matrix.Translation(body.position)
        bone = data.edit_bones.new(body.name)
        bone.length = 0.01
        bone.matrix = world
        if body.parent is not None:
            bone.parent = bones[body.parent]
        bones[body.name] = bone
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.rotation_mode = "QUATERNION"
    for joint in profile.joints:
        armature.pose.bones[joint.child_body].rotation_mode = "XYZ"
    return armature


def action_payload(profile: RobotProfile) -> dict[str, np.ndarray]:
    return archive_payload(profile)


def action_fcurves(action):
    legacy = getattr(action, "fcurves", None)
    if legacy is not None:
        return list(legacy)
    result = []
    for layer in action.layers:
        for strip in layer.strips:
            if strip.type == "KEYFRAME":
                for channelbag in strip.channelbags:
                    result.extend(channelbag.fcurves)
    return result


def assert_vector_close(test: unittest.TestCase, actual, expected):
    np.testing.assert_allclose(tuple(actual), tuple(expected), atol=1e-6)


def assert_quaternion_close(test: unittest.TestCase, actual, expected):
    test.assertAlmostEqual(abs(Quaternion(actual).dot(Quaternion(expected))), 1.0, places=6)


class MotionActionTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)

    def import_action(self, action_name="Walk"):
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            return import_motion_action(self.armature, self.profile, path, action_name=action_name)

    def test_imports_calibrated_root_and_joint_keys_into_fake_user_action(self):
        bpy.context.scene.render.fps_base = 1.25
        action = self.import_action()
        scene = bpy.context.scene
        pose_bone = self.armature.pose.bones["hip_link"]

        self.assertEqual(action.name, "Walk")
        self.assertTrue(action.use_fake_user)
        self.assertTrue(action.use_frame_range)
        self.assertEqual((action.frame_start, action.frame_end), (1.0, 3.0))
        self.assertEqual((scene.frame_start, scene.frame_end), (1, 3))
        self.assertEqual((scene.render.fps, scene.render.fps_base), (50, 1.0))
        scene.frame_set(3)
        assert_vector_close(self, self.armature.location, (2.0, 2.0, 2.875))
        assert_quaternion_close(self, self.armature.rotation_quaternion, (0.0, 0.0, 0.0, 1.0))
        self.assertAlmostEqual(pose_bone.rotation_euler.z, 0.5, places=6)

    def test_import_tags_action_with_kind_exact_source_hash_and_non_loopability(self):
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(
                Path(directory), action_payload(self.profile)
            )
            expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            action = import_motion_action(
                self.armature, self.profile, path, action_name="Tagged"
            )

        self.assertIn("duck_motion_kind", action)
        self.assertIn("duck_source_sha256", action)
        self.assertIn("duck_loopable", action)
        self.assertEqual(action["duck_motion_kind"], "mjlab_import")
        self.assertEqual(action["duck_source_sha256"], expected_sha256)
        self.assertFalse(action["duck_loopable"])

    def test_import_sets_every_keyframe_to_linear_in_both_action_apis(self):
        action = self.import_action()
        curves = action_fcurves(action)

        self.assertGreater(len(curves), 0)
        self.assertGreater(sum(len(curve.keyframe_points) for curve in curves), 0)
        self.assertEqual(
            {
                point.interpolation
                for curve in curves
                for point in curve.keyframe_points
            },
            {"LINEAR"},
        )

    def _add_ik_constraint(self, influence=0.75):
        constraint = self.armature.pose.bones["knee_link"].constraints.new(
            "COPY_LOCATION"
        )
        constraint.name = "DUCK_IK_TEST"
        constraint.influence = influence
        self.armature["fk_ik"] = 1.0
        return constraint

    def test_import_forces_fk_before_assigning_the_new_action(self):
        constraint = self._add_ik_constraint()
        self.armature.animation_data_create()
        real_animation_data = self.armature.animation_data

        class GuardedAnimationData:
            def __init__(self, wrapped):
                self.wrapped = wrapped

            @property
            def action(self):
                return self.wrapped.action

            @action.setter
            def action(self, value):
                if value is not None and value.name.startswith("Guarded"):
                    if constraint.influence != 0.0 or self_armature["fk_ik"] != 0.0:
                        raise AssertionError("action assigned before FK was forced")
                self.wrapped.action = value

        self_armature = self.armature
        guarded_animation_data = GuardedAnimationData(real_animation_data)

        class GuardedArmature:
            def __getattr__(self, name):
                if name == "animation_data":
                    return guarded_animation_data
                return getattr(self_armature, name)

            def __getitem__(self, key):
                return self_armature[key]

            def __setitem__(self, key, value):
                self_armature[key] = value

            def get(self, key, default=None):
                return self_armature.get(key, default)

        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(
                Path(directory), action_payload(self.profile)
            )
            try:
                action = import_motion_action(
                    GuardedArmature(), self.profile, path, action_name="GuardedImport"
                )
            except MotionError as exc:
                self.fail(str(exc))

        self.assertEqual(action.name, "GuardedImport")
        self.assertEqual(constraint.influence, 0.0)
        self.assertEqual(self.armature["fk_ik"], 0.0)

    def test_rejects_inconsistent_archive_before_mutating_or_forcing_fk(self):
        scene = bpy.context.scene
        constraint = self._add_ik_constraint(influence=0.6)
        prior = bpy.data.actions.new("Prior")
        self.armature.animation_data_create()
        self.armature.animation_data.action = prior
        scene.frame_start, scene.frame_end = 7, 9
        scene.frame_set(8)
        original_matrix = self.armature.matrix_world.copy()
        payload = action_payload(self.profile)
        payload["body_pos_w"][1, 2, 1] += 1e-3

        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), payload)
            with self.assertRaisesRegex(MotionError, r"frame 1.*knee_link"):
                import_motion_action(
                    self.armature, self.profile, path, action_name="Rejected"
                )

        self.assertIs(self.armature.animation_data.action, prior)
        self.assertEqual((scene.frame_start, scene.frame_end, scene.frame_current), (7, 9, 8))
        self.assertEqual(self.armature.matrix_world, original_matrix)
        self.assertAlmostEqual(constraint.influence, 0.6, places=6)
        self.assertEqual(self.armature["fk_ik"], 1.0)
        self.assertIsNone(bpy.data.actions.get("Rejected"))

    def test_preserves_existing_action_when_blender_assigns_collision_suffix(self):
        existing = bpy.data.actions.new("Walk")
        action = self.import_action()
        self.assertEqual(existing.name, "Walk")
        self.assertEqual(action.name, "Walk.001")

    def test_restores_scene_and_removes_partial_action_when_keying_fails(self):
        scene = bpy.context.scene
        prior = bpy.data.actions.new("Prior")
        self.armature.animation_data_create()
        self.armature.animation_data.action = prior
        self.armature.location = (9.0, 8.0, 7.0)
        self.armature.rotation_mode = "QUATERNION"
        self.armature.rotation_quaternion = (0.5, 0.5, 0.5, 0.5)
        pose_bone = self.armature.pose.bones["hip_link"]
        constraint = self._add_ik_constraint(influence=0.65)
        pose_bone.rotation_euler.z = 0.2
        scene.frame_start, scene.frame_end = 7, 9
        scene.frame_set(8)
        influence_at_failure = []

        def fail_keyframe(*_args, **_kwargs):
            influence_at_failure.append(constraint.influence)
            raise RuntimeError("injected")

        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            with mock.patch.object(
                type(pose_bone), "keyframe_insert", side_effect=fail_keyframe
            ):
                with self.assertRaisesRegex(MotionError, "injected"):
                    import_motion_action(self.armature, self.profile, path, action_name="Broken")

        self.assertIs(self.armature.animation_data.action, prior)
        self.assertEqual((scene.frame_start, scene.frame_end, scene.frame_current), (7, 9, 8))
        assert_vector_close(self, self.armature.location, (9.0, 8.0, 7.0))
        assert_quaternion_close(self, self.armature.rotation_quaternion, (0.5, 0.5, 0.5, 0.5))
        self.assertAlmostEqual(pose_bone.rotation_euler.z, 0.2, places=6)
        self.assertEqual(influence_at_failure, [0.0])
        self.assertAlmostEqual(constraint.influence, 0.65, places=6)
        self.assertEqual(self.armature["fk_ik"], 1.0)
        self.assertIsNone(bpy.data.actions.get("Broken"))

    def test_restores_keyed_action_and_unkeyed_overrides_after_keying_fails(self):
        scene = bpy.context.scene
        created_mouth_property = not hasattr(bpy.types.Object, "duck_mouth_open")
        if created_mouth_property:
            bpy.types.Object.duck_mouth_open = bpy.props.FloatProperty(default=0.0)
            self.addCleanup(delattr, bpy.types.Object, "duck_mouth_open")

        bpy.context.view_layer.objects.active = self.armature
        bpy.ops.object.mode_set(mode="EDIT")
        mouth_edit_bone = self.armature.data.edit_bones.new("mouth::lower_beak")
        mouth_edit_bone.parent = self.armature.data.edit_bones["root"]
        mouth_edit_bone.matrix = Matrix.Translation((0.0, 0.0, 0.2))
        mouth_edit_bone.length = 0.01
        bpy.ops.object.mode_set(mode="OBJECT")

        prior = bpy.data.actions.new("PriorKeyed")
        self.armature.animation_data_create()
        self.armature.animation_data.action = prior
        hip = self.armature.pose.bones["hip_link"]
        mouth = self.armature.pose.bones["mouth::lower_beak"]

        self.armature.location = (1.0, 2.0, 3.0)
        self.armature.keyframe_insert(data_path="location", frame=7)
        self.armature.location = (5.0, 6.0, 7.0)
        self.armature.keyframe_insert(data_path="location", frame=9)
        hip.rotation_mode = "XYZ"
        hip.rotation_euler.z = 0.1
        hip.keyframe_insert(data_path="rotation_euler", index=2, frame=7)
        hip.rotation_euler.z = 0.9
        hip.keyframe_insert(data_path="rotation_euler", index=2, frame=9)
        mouth.location = (0.01, 0.0, 0.0)
        mouth.keyframe_insert(data_path="location", frame=7)
        mouth.location = (0.03, 0.0, 0.0)
        mouth.keyframe_insert(data_path="location", frame=9)
        self.armature.duck_mouth_open = 0.1
        self.armature.keyframe_insert(data_path="duck_mouth_open", frame=7)
        self.armature.duck_mouth_open = 0.9
        self.armature.keyframe_insert(data_path="duck_mouth_open", frame=9)

        scene.frame_start, scene.frame_end = 7, 9
        scene.render.fps = 23
        scene.render.fps_base = 1.001
        scene.frame_set(8)
        self.armature.rotation_mode = "QUATERNION"
        self.armature.matrix_world = Matrix.Translation((9.0, 8.0, 7.0)) @ Quaternion(
            (0.9238795, 0.0, 0.3826834, 0.0)
        ).to_matrix().to_4x4()
        hip.rotation_mode = "ZYX"
        hip.matrix_basis = Matrix.Translation((0.01, 0.02, 0.03)) @ Quaternion(
            (0.9659258, 0.2588190, 0.0, 0.0)
        ).to_matrix().to_4x4()
        mouth.rotation_mode = "AXIS_ANGLE"
        mouth.matrix_basis = Matrix.Translation((0.04, 0.05, 0.06)) @ Quaternion(
            (0.9848078, 0.0, 0.0, 0.1736482)
        ).to_matrix().to_4x4()
        self.armature.duck_mouth_open = 0.73
        expected_matrix = self.armature.matrix_world.copy()
        expected_hip_basis = hip.matrix_basis.copy()
        expected_mouth_basis = mouth.matrix_basis.copy()

        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(
                Path(directory), action_payload(self.profile)
            )
            with mock.patch.object(
                type(hip), "keyframe_insert", side_effect=RuntimeError("injected")
            ):
                with self.assertRaisesRegex(MotionError, "injected"):
                    import_motion_action(
                        self.armature, self.profile, path, action_name="BrokenKeyed"
                    )

        self.assertIs(self.armature.animation_data.action, prior)
        self.assertEqual(
            (
                scene.frame_start,
                scene.frame_end,
                scene.frame_current,
                scene.render.fps,
            ),
            (7, 9, 8, 23),
        )
        self.assertAlmostEqual(scene.render.fps_base, 1.001, places=6)
        np.testing.assert_allclose(self.armature.matrix_world, expected_matrix, atol=1e-6)
        self.assertEqual(self.armature.rotation_mode, "QUATERNION")
        np.testing.assert_allclose(hip.matrix_basis, expected_hip_basis, atol=1e-6)
        self.assertEqual(hip.rotation_mode, "ZYX")
        np.testing.assert_allclose(mouth.matrix_basis, expected_mouth_basis, atol=1e-6)
        self.assertEqual(mouth.rotation_mode, "AXIS_ANGLE")
        self.assertAlmostEqual(self.armature.duck_mouth_open, 0.73, places=6)
        self.assertIsNone(bpy.data.actions.get("BrokenKeyed"))

    def test_removes_new_animation_data_when_keying_fails_without_prior_animation(self):
        self.assertIsNone(self.armature.animation_data)
        pose_bone = self.armature.pose.bones["hip_link"]
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            with mock.patch.object(type(pose_bone), "keyframe_insert", side_effect=RuntimeError("injected")):
                with self.assertRaisesRegex(MotionError, "injected"):
                    import_motion_action(self.armature, self.profile, path, action_name="NoAnimation")

        self.assertIsNone(self.armature.animation_data)
        self.assertIsNone(bpy.data.actions.get("NoAnimation"))

    def test_translates_animation_data_creation_failure_without_mutation(self):
        class FailingAnimationCreateArmature:
            def __init__(self, armature):
                self._armature = armature

            def __getattr__(self, name):
                return getattr(self._armature, name)

            def animation_data_create(self):
                raise RuntimeError("create injected")

        self.assertIsNone(self.armature.animation_data)
        with tempfile.TemporaryDirectory() as directory:
            path = MotionLoaderTests().write_archive(Path(directory), action_payload(self.profile))
            with self.assertRaisesRegex(MotionError, "create injected"):
                import_motion_action(
                    FailingAnimationCreateArmature(self.armature),
                    self.profile,
                    path,
                    action_name="CreateFailure",
                )

        self.assertIsNone(self.armature.animation_data)
        self.assertIsNone(bpy.data.actions.get("CreateFailure"))
