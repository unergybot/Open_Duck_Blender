import math
import json
import unittest
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import mock

import bpy
import numpy as np
from mathutils import Matrix

from open_duck_tools import addon
from open_duck_tools.motion import MotionError
from open_duck_tools.motion_import import import_motion_action
from open_duck_tools.policy_preview import (
    ProcessOutcome,
    PreviewConfig,
    validate_preview_config,
)
from open_duck_tools.profile import MouthSample, profile_to_json
from tests.test_motion_import import (
    action_payload,
    build_minimal_rig,
    test_profile,
)


def play_once_handlers():
    return [
        handler
        for handler in bpy.app.handlers.frame_change_post
        if getattr(handler, "_duck_play_once_handler", False)
    ]


def native_playback_cleanup_handlers():
    return [
        handler
        for handler in bpy.app.handlers.animation_playback_post
        if getattr(handler, "_duck_native_playback_handler", False)
    ]


class AddonRegistrationTests(unittest.TestCase):
    def tearDown(self):
        addon.unregister()

    def test_registers_tools_panel_and_is_idempotent(self):
        addon.register()
        addon.register()
        self.assertTrue(hasattr(bpy.types, "DUCK_PT_tools"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_import_motion"))
        self.assertIn(addon.DUCK_OT_import_motion, addon.CLASSES)
        self.assertTrue(hasattr(bpy.types.Object, "duck_colorway"))

    def test_approximate_mouth_control_is_explicitly_labeled(self):
        addon.register()
        self.assertEqual(
            bpy.types.Object.bl_rna.properties["duck_mouth_open"].name,
            "Mouth (visual approximation)",
        )

    def test_registers_policy_preview_operators_and_defaults(self):
        addon.register()
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_generate_policy_preview"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_cancel_policy_preview"))
        self.assertAlmostEqual(
            bpy.types.Object.bl_rna.properties["duck_policy_forward"].default, 0.30
        )
        self.assertEqual(
            bpy.types.Object.bl_rna.properties["duck_policy_duration"].default, 4.0
        )


class RecordedLayout:
    def __init__(self, calls, *, enabled=True):
        self.calls = calls
        self.enabled = enabled

    def _child(self, kind, **kwargs):
        self.calls.append((kind, kwargs, self.enabled))
        return type(self)(self.calls, enabled=self.enabled)

    def box(self):
        return self._child("box")

    def column(self, **kwargs):
        return self._child("column", **kwargs)

    def row(self, **kwargs):
        return self._child("row", **kwargs)

    def label(self, **kwargs):
        self.calls.append(("label", kwargs, self.enabled))

    def operator(self, operator_id, **kwargs):
        self.calls.append(("operator", {"id": operator_id, **kwargs}, self.enabled))
        return SimpleNamespace()

    def prop(self, _data, property_name, **kwargs):
        self.calls.append(("prop", {"property": property_name, **kwargs}, self.enabled))

    def prop_search(
        self, _data, property_name, _search_data, _search_property, **kwargs
    ):
        self.calls.append(
            ("prop_search", {"property": property_name, **kwargs}, self.enabled)
        )


class PolicyPreviewPanelTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon.register()
        self.armature = bpy.data.objects.new(
            "MicroduckPreview", bpy.data.armatures.new("MicroduckPreview")
        )
        bpy.context.scene.collection.objects.link(self.armature)
        self.armature["duck_robot_id"] = "microduck-alpha"
        bpy.context.view_layer.objects.active = self.armature
        self.armature.select_set(True)

    def tearDown(self):
        addon.unregister()

    def draw_panel(self):
        calls = []
        addon.DUCK_PT_tools.draw(
            SimpleNamespace(layout=RecordedLayout(calls)), bpy.context
        )
        return calls

    def test_draws_policy_preview_controls_for_microduck_only(self):
        calls = self.draw_panel()

        self.assertIn(
            ("label", {"text": "Generate Policy Preview", "icon": "PLAY"}, True),
            calls,
        )
        self.assertEqual(
            [
                call[1]["property"]
                for call in calls
                if call[0] == "prop" and call[1]["property"].startswith("duck_policy_")
            ],
            [
                "duck_policy_path",
                "duck_policy_forward",
                "duck_policy_lateral",
                "duck_policy_yaw",
                "duck_policy_duration",
                "duck_policy_seed",
                "duck_policy_setup_open",
            ],
        )
        self.assertIn(("label", {"text": "Idle", "icon": "INFO"}, True), calls)
        self.assertIn(
            (
                "operator",
                {
                    "id": "duck.generate_policy_preview",
                    "text": "Generate & Import",
                    "icon": "FILE_REFRESH",
                },
                True,
            ),
            calls,
        )
        self.armature["duck_robot_id"] = "other-duck"
        calls = self.draw_panel()
        self.assertNotIn(
            ("label", {"text": "Generate Policy Preview", "icon": "PLAY"}, True),
            calls,
        )

    def test_disables_generate_for_another_armatures_running_preview(self):
        owner = bpy.data.objects.new(
            "OtherMicroduck", bpy.data.armatures.new("OtherMicroduck")
        )
        bpy.context.scene.collection.objects.link(owner)
        owner["duck_robot_id"] = "microduck-alpha"
        addon._POLICY_PREVIEW_SESSION = addon._PolicyPreviewSession(
            owner.name,
            owner.as_pointer(),
            bpy.context.scene.name,
            mock.Mock(),
            Path(tempfile.gettempdir()) / "policy-preview.npz",
            FakePreviewProcess(),
        )

        calls = self.draw_panel()

        self.assertIn(
            (
                "label",
                {"text": "Another policy preview is running", "icon": "INFO"},
                True,
            ),
            calls,
        )
        self.assertIn(
            (
                "operator",
                {
                    "id": "duck.generate_policy_preview",
                    "text": "Generate & Import",
                    "icon": "FILE_REFRESH",
                },
                False,
            ),
            calls,
        )
        self.assertNotIn(
            (
                "operator",
                {"id": "duck.cancel_policy_preview", "icon": "CANCEL"},
                True,
            ),
            calls,
        )


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
        base_profile = test_profile()
        mouth_sample = MouthSample(0.0, {})
        self.profile = replace(
            base_profile,
            robot_id="microduck-alpha",
            mouth=replace(
                base_profile.mouth,
                samples=(mouth_sample,),
                validation_poses=(mouth_sample,),
            ),
        )
        self.armature = build_minimal_rig(self.profile)
        self.armature["duck_robot_id"] = self.profile.robot_id
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
        self.armature.duck_microduck_root = str(runtime)
        self.armature.duck_microduck_rl_root = str(rollout)
        self.armature.duck_policy_path = str(policy)
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
        self.assertTrue(self.validated.cache_path.is_file())
        self.assertEqual(process.closed, [False])
        self.assertIsNone(addon._POLICY_PREVIEW_SESSION)

    def test_success_imports_into_launch_scene_when_ambient_scene_changed(self):
        launch_scene = bpy.context.scene
        launch_scene.frame_start, launch_scene.frame_end = 7, 9
        launch_scene.frame_set(8)
        launch_scene.render.fps = 24
        launch_scene.render.fps_base = 1.0
        output = self.validated.cache_path.with_name("scene-switched.npz")
        self.write_archive(output)
        process = FakePreviewProcess(ProcessOutcome(0, False, "Frames: 3"))
        self.install_session(process, output)
        bpy.ops.screen.animation_play()

        ambient_scene = bpy.data.scenes.new("Ambient")
        ambient_scene.frame_start, ambient_scene.frame_end = 30, 40
        ambient_scene.frame_set(35)
        ambient_scene.render.fps = 60
        ambient_scene.render.fps_base = 1.25
        ambient_before = (
            ambient_scene.frame_start,
            ambient_scene.frame_end,
            ambient_scene.frame_current,
            ambient_scene.render.fps,
            ambient_scene.render.fps_base,
        )
        bpy.context.window.scene = ambient_scene

        self.assertIsNone(addon._poll_policy_preview_job())

        self.assertIs(bpy.context.scene, ambient_scene)
        self.assertEqual(
            (
                ambient_scene.frame_start,
                ambient_scene.frame_end,
                ambient_scene.frame_current,
                ambient_scene.render.fps,
                ambient_scene.render.fps_base,
            ),
            ambient_before,
        )
        self.assertEqual(
            (
                launch_scene.frame_start,
                launch_scene.frame_end,
                launch_scene.frame_current,
                launch_scene.render.fps,
                launch_scene.render.fps_base,
            ),
            (1, 3, 1, 50, 1.0),
        )
        self.assertFalse(bpy.context.screen.is_animation_playing)
        self.assertEqual(
            self.armature.animation_data.action.name,
            "PolicyWalk_x0.30_y0.00_yaw0.00",
        )

    def test_missing_launch_scene_force_cleans_without_import(self):
        launch_scene = bpy.context.scene
        replacement = bpy.data.scenes.new("Replacement")
        replacement.collection.objects.link(self.armature)
        replacement.frame_start, replacement.frame_end = 20, 30
        replacement.frame_set(25)
        output = self.validated.cache_path.with_name("missing-scene.npz")
        self.write_archive(output)
        process = FakePreviewProcess(ProcessOutcome(0, False, "Frames: 3"))
        self.install_session(process, output)
        original_actions = {action.name for action in bpy.data.actions}
        bpy.context.window.scene = replacement
        bpy.data.scenes.remove(launch_scene)

        self.assertIsNone(addon._poll_policy_preview_job())

        self.assertEqual({action.name for action in bpy.data.actions}, original_actions)
        self.assertTrue(
            self.armature.animation_data is None
            or self.armature.animation_data.action is None
        )
        self.assertEqual(
            (
                replacement.frame_start,
                replacement.frame_end,
                replacement.frame_current,
            ),
            (20, 30, 25),
        )
        self.assertEqual(process.closed, [True])
        self.assertFalse(output.exists())
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
        self.assertTrue(
            self.armature.animation_data is None
            or self.armature.animation_data.action is None
        )
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

    def test_generation_requires_declared_and_embedded_microduck_identity(self):
        cases = (
            ("other-duck", self.profile),
            (
                "microduck-alpha",
                replace(self.profile, robot_id="other-duck"),
            ),
        )
        for declared_robot_id, profile in cases:
            with self.subTest(
                declared_robot_id=declared_robot_id,
                profile_robot_id=profile.robot_id,
            ):
                self.armature["duck_robot_id"] = declared_robot_id
                self.armature.data["duck_robot_profile_json"] = profile_to_json(profile)
                with mock.patch.object(addon.PreviewProcess, "start") as start:
                    result = addon._start_policy_preview(self.armature, bpy.context)

                try:
                    self.assertEqual(result, {"CANCELLED"})
                    self.assertIsNone(addon._POLICY_PREVIEW_SESSION)
                    start.assert_not_called()
                finally:
                    addon._clear_policy_preview_job(force=True)

    def test_wrong_owner_cannot_cancel_live_session_from_python_or_operator_search(self):
        process = FakePreviewProcess()
        output = self.validated.cache_path.with_name("owned.npz")
        self.install_session(process, output)
        wrong_owner = build_minimal_rig(self.profile)
        wrong_owner["duck_robot_id"] = "microduck-alpha"
        wrong_owner.data["duck_robot_profile_json"] = profile_to_json(self.profile)
        self.armature.select_set(False)
        wrong_owner.select_set(True)
        bpy.context.view_layer.objects.active = wrong_owner

        self.assertFalse(bpy.ops.duck.cancel_policy_preview.poll())
        result = addon.DUCK_OT_cancel_policy_preview.execute(
            SimpleNamespace(), bpy.context
        )

        self.assertEqual(result, {"CANCELLED"})
        self.assertEqual(process.cancel_requests, 0)
        self.assertIsNotNone(addon._POLICY_PREVIEW_SESSION)

    def test_blender_duration_property_normalizes_argv_cache_and_provenance(self):
        cases = (
            (
                0.02,
                1,
                "0.02",
                "d7dae6bf54b8af080af43e71a54d262317c76b48281ceb4c555848932f9a370f",
            ),
            (
                0.06,
                3,
                "0.059999999999999998",
                "dfdb079540035d4b9b57d40e463aaf5501e6b5901112e05894048f24a9fea6d2",
            ),
            (
                0.10,
                5,
                "0.10000000000000001",
                "fd0457d6d84ef0e59beaed2205435a29f263215bc06bd3ef3dd140d30ab16fa5",
            ),
        )
        for duration, frames, cli_duration, provenance in cases:
            with self.subTest(duration=duration):
                self.armature.duck_policy_duration = duration
                cache_root = Path(self.temporary.name) / f"property-cache-{frames}"
                process = FakePreviewProcess()

                def validate_with_uv(config):
                    return validate_preview_config(
                        config, which=lambda _name: "/usr/bin/uv"
                    )

                with mock.patch.object(
                    addon, "_policy_preview_cache_root", return_value=cache_root
                ), mock.patch.object(
                    addon,
                    "validate_preview_config",
                    side_effect=validate_with_uv,
                ), mock.patch.object(
                    addon.PreviewProcess, "start", return_value=process
                ) as start, mock.patch.object(addon, "_ensure_policy_preview_timer"):
                    result = addon._start_policy_preview(self.armature, bpy.context)

                try:
                    self.assertEqual(result, {"FINISHED"})
                    validated = start.call_args.args[0]
                    duration_index = validated.argv.index("--duration") + 1
                    self.assertEqual(validated.frames, frames)
                    self.assertEqual(validated.config.duration_s, duration)
                    self.assertEqual(validated.argv[duration_index], cli_duration)
                    self.assertEqual(
                        json.loads(validated.canonical_config_json)["duration_s"],
                        cli_duration,
                    )
                    self.assertEqual(validated.rollout_config_sha256, provenance)

                    exact = validate_preview_config(
                        PreviewConfig(
                            Path(self.armature.duck_microduck_root),
                            Path(self.armature.duck_microduck_rl_root),
                            Path(self.armature.duck_policy_path),
                            (
                                self.armature.duck_policy_forward,
                                self.armature.duck_policy_lateral,
                                self.armature.duck_policy_yaw,
                            ),
                            duration,
                            0,
                            cache_root,
                        ),
                        which=lambda _name: "/usr/bin/uv",
                    )
                    self.assertEqual(validated.cache_key, exact.cache_key)
                finally:
                    addon._clear_policy_preview_job(force=True)

    def test_blender_duration_property_rejects_nonintegral_value(self):
        self.armature.duck_policy_duration = 0.031
        with mock.patch.object(addon.PreviewProcess, "start") as start:
            result = addon._start_policy_preview(self.armature, bpy.context)

        self.assertEqual(result, {"CANCELLED"})
        start.assert_not_called()

    def test_load_pre_and_atexit_force_process_cleanup(self):
        for cleanup in (
            addon._policy_preview_load_pre_handler,
            addon._atexit_policy_preview_cleanup,
        ):
            with self.subTest(cleanup=cleanup.__name__):
                process = FakePreviewProcess()
                output = self.validated.cache_path.with_name(
                    f"{cleanup.__name__}.npz"
                )
                output.write_bytes(b"partial")
                self.install_session(process, output)

                cleanup()

                self.assertEqual(process.closed, [True])
                self.assertIsNone(addon._POLICY_PREVIEW_SESSION)
                if cleanup is addon._policy_preview_load_pre_handler:
                    self.assertFalse(output.exists())

    def test_unregister_force_closes_child_and_removes_timer(self):
        process = FakePreviewProcess()
        output = self.validated.cache_path.with_name("live.npz")
        self.install_session(process, output)
        bpy.app.timers.register(addon._poll_policy_preview_job, first_interval=60.0)

        addon.unregister()

        self.assertEqual(process.closed, [True])
        self.assertFalse(bpy.app.timers.is_registered(addon._poll_policy_preview_job))
        self.assertIsNone(addon._POLICY_PREVIEW_SESSION)

class MotionImportOperatorTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon.register()
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)
        self.armature.data["duck_robot_profile_json"] = profile_to_json(self.profile)

    def tearDown(self):
        addon.unregister()

    def write_archive(self, directory: Path, name: str, payload: dict) -> Path:
        path = directory / name
        np.savez_compressed(path, **payload)
        return path

    def test_imports_fixture_with_sanitized_default_action_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(
                Path(directory), "Policy alpha walking forward!.npz", action_payload(self.profile)
            )
            result = bpy.ops.duck.import_motion(filepath=str(path), action_name="")

        self.assertEqual(result, {"FINISHED"})
        self.assertEqual(
            self.armature.animation_data.action.name, "Policy_alpha_walking_forward"
        )
        self.assertEqual((bpy.context.scene.frame_start, bpy.context.scene.frame_end), (1, 3))

    def test_cancels_malformed_archive_without_changing_scene(self):
        scene = bpy.context.scene
        prior = bpy.data.actions.new("Prior")
        self.armature.animation_data_create()
        self.armature.animation_data.action = prior
        self.armature.location = (9.0, 8.0, 7.0)
        scene.frame_start, scene.frame_end = 7, 9
        scene.frame_set(8)
        self.armature.keyframe_insert(data_path="location", frame=8)
        original_matrix = self.armature.matrix_world.copy()
        original_actions = {action.name for action in bpy.data.actions}
        original_action_range = tuple(prior.frame_range)
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(Path(directory), "malformed.npz", {"bad": np.array([1])})
            result = bpy.ops.duck.import_motion(filepath=str(path), action_name="Ignored")

        self.assertEqual(result, {"CANCELLED"})
        self.assertIs(self.armature.animation_data.action, prior)
        self.assertEqual((scene.frame_start, scene.frame_end, scene.frame_current), (7, 9, 8))
        self.assertEqual(self.armature.matrix_world, original_matrix)
        self.assertEqual({action.name for action in bpy.data.actions}, original_actions)
        self.assertEqual(tuple(prior.frame_range), original_action_range)

    def test_failed_import_stops_playback_and_cleans_temporary_handler(self):
        action = bpy.data.actions.new("OneShot")
        action.use_frame_range = True
        action.frame_start = 1
        action.frame_end = 3
        action["duck_loopable"] = False
        self.armature.duck_action_name = action.name
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertTrue(bpy.context.screen.is_animation_playing)

        with tempfile.TemporaryDirectory() as directory:
            path = self.write_archive(
                Path(directory), "malformed.npz", {"bad": np.array([1])}
            )
            result = bpy.ops.duck.import_motion(
                filepath=str(path), action_name="Ignored"
            )

        self.assertEqual(result, {"CANCELLED"})
        self.assertFalse(bpy.context.screen.is_animation_playing)
        self.assertEqual(play_once_handlers(), [])


class MotionExportTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "export-fixture.npz"
            np.savez_compressed(path, **action_payload(self.profile))
            import_motion_action(
                self.armature, self.profile, path, action_name="ExportFixture"
            )

    def test_rejects_non_50hz_effective_rate_before_changing_frame(self):
        scene = bpy.context.scene
        scene.render.fps = 60
        scene.render.fps_base = 1.0
        scene.frame_set(17)
        changed_frames = []

        def record_frame_change(changed_scene, *_args):
            changed_frames.append(changed_scene.frame_current)

        bpy.app.handlers.frame_change_post.append(record_frame_change)
        try:
            with self.assertRaisesRegex(MotionError, r"effective.*60.*50 Hz"):
                addon.collect_armature_motion(self.armature, self.profile, 1, 3)
        finally:
            bpy.app.handlers.frame_change_post.remove(record_frame_change)

        self.assertEqual(scene.frame_current, 17)
        self.assertEqual(changed_frames, [])

    def test_accepts_100_over_2_and_writes_canonical_50hz_metadata(self):
        scene = bpy.context.scene
        scene.render.fps = 100
        scene.render.fps_base = 2.0
        scene.frame_set(17)

        try:
            archive = addon.collect_armature_motion(
                self.armature, self.profile, 1, 3
            )
        except MotionError as exc:
            self.fail(str(exc))

        self.assertEqual(archive["fps"].tolist(), [50])
        self.assertEqual(scene.frame_current, 17)

    def test_rejects_off_axis_joint_rotation_and_restores_original_frame(self):
        scene = bpy.context.scene
        hip = self.armature.pose.bones["hip_link"]
        for frame, angle in ((1, 0.0), (2, 0.02), (3, 0.0)):
            scene.frame_set(frame)
            hip.rotation_euler.x = angle
            hip.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)
        scene.frame_set(17)

        with self.assertRaisesRegex(
            MotionError, r"frame 2.*hip_link.*rotation residual"
        ):
            addon.collect_armature_motion(self.armature, self.profile, 1, 3)

        self.assertEqual(scene.frame_current, 17)

    def test_rejects_evaluated_scale_drift_and_identifies_frame_and_body(self):
        scene = bpy.context.scene
        knee = self.armature.pose.bones["knee_link"]
        for frame, scale in ((1, 1.0), (2, 1.002), (3, 1.0)):
            scene.frame_set(frame)
            knee.scale.x = scale
            knee.keyframe_insert(data_path="scale", index=0, frame=frame)
        scene.frame_set(17)

        with self.assertRaisesRegex(
            MotionError, r"frame 2.*knee_link.*affine residual"
        ):
            addon.collect_armature_motion(self.armature, self.profile, 1, 3)

        self.assertEqual(scene.frame_current, 17)

    def test_rejects_non_finite_evaluated_matrix_instead_of_canonicalizing_it(self):
        scene = bpy.context.scene
        scene.frame_set(1)
        matrices = addon._evaluated_body_matrices(
            self.armature, self.profile.body_names
        )
        matrices["knee_link"] = matrices["knee_link"].copy()
        matrices["knee_link"][0][0] = math.nan
        scene.frame_set(17)

        with mock.patch.object(
            addon, "_evaluated_body_matrices", return_value=matrices
        ):
            with self.assertRaisesRegex(
                MotionError, r"frame 1.*knee_link.*affine residual"
            ):
                addon.collect_armature_motion(self.armature, self.profile, 1, 1)

        self.assertEqual(scene.frame_current, 17)

    def test_restores_frame_and_subframe_after_success_and_failure(self):
        scene = bpy.context.scene
        scene.frame_set(2, subframe=0.5)
        addon.collect_armature_motion(self.armature, self.profile, 1, 3)
        self.assertEqual(scene.frame_current, 2)
        self.assertAlmostEqual(scene.frame_subframe, 0.5, places=7)

        hip = self.armature.pose.bones["hip_link"]
        for frame, angle in ((1, 0.0), (2, 0.02), (3, 0.0)):
            scene.frame_set(frame)
            hip.rotation_euler.x = angle
            hip.keyframe_insert(data_path="rotation_euler", index=0, frame=frame)
        scene.frame_set(2, subframe=0.5)
        with self.assertRaisesRegex(MotionError, r"rotation residual"):
            addon.collect_armature_motion(self.armature, self.profile, 1, 3)
        self.assertEqual(scene.frame_current, 2)
        self.assertAlmostEqual(scene.frame_subframe, 0.5, places=7)


class AnimationControlTests(unittest.TestCase):
    def setUp(self):
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon.register()
        self.profile = test_profile()
        self.armature = build_minimal_rig(self.profile)
        self.armature.data["duck_robot_profile_json"] = profile_to_json(self.profile)
        self.walk = bpy.data.actions.new("Policy_alpha_walking_forward")
        self.walk.use_frame_range = True
        self.walk.frame_start = 1
        self.walk.frame_end = 200
        self.walk["duck_loopable"] = False
        self.crouch = bpy.data.actions.new("KinematicCrouchTest")
        self.crouch.use_frame_range = True
        self.crouch.frame_start = 1
        self.crouch.frame_end = 51
        self.crouch["duck_loopable"] = False

    def tearDown(self):
        if bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        addon.unregister()

    def test_registers_beginner_animation_controls(self):
        self.assertTrue(hasattr(bpy.types.Object, "duck_action_name"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_select_action"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_toggle_animation"))
        self.assertTrue(hasattr(bpy.types, "DUCK_OT_reset_animation"))

    def test_action_search_selection_activates_action_range_and_first_frame(self):
        bpy.context.scene.frame_set(17)

        self.armature.duck_action_name = self.walk.name

        self.assertIs(self.armature.animation_data.action, self.walk)
        self.assertEqual(self.armature.duck_action_name, self.walk.name)
        self.assertEqual(
            (bpy.context.scene.frame_start, bpy.context.scene.frame_end),
            (1, 200),
        )
        self.assertEqual(bpy.context.scene.frame_current, 1)

    def test_action_selection_and_play_force_fk_without_baking(self):
        constraint = self.armature.pose.bones["knee_link"].constraints.new(
            "COPY_LOCATION"
        )
        constraint.name = "DUCK_IK_PLAYBACK"
        constraint.influence = 0.75
        self.armature["fk_ik"] = 1.0

        self.armature.duck_action_name = self.walk.name

        self.assertEqual(constraint.influence, 0.0)
        self.assertEqual(self.armature["fk_ik"], 0.0)
        constraint.influence = 0.5
        self.armature["fk_ik"] = 1.0
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(constraint.influence, 0.0)
        self.assertEqual(self.armature["fk_ik"], 0.0)

    def test_selection_play_and_reset_clear_unkeyed_canonical_pose_contamination(self):
        root = self.armature.pose.bones["root"]
        hip = self.armature.pose.bones["hip_link"]
        knee = self.armature.pose.bones["knee_link"]

        def contaminate():
            root.location.x = 0.05
            hip.rotation_euler.x = 0.2
            hip.location.y = 0.03
            knee.scale.y = 1.2

        def assert_canonical():
            for bone_name in self.profile.body_names:
                np.testing.assert_allclose(
                    self.armature.pose.bones[bone_name].matrix_basis,
                    Matrix.Identity(4),
                    atol=1e-6,
                    err_msg=bone_name,
                )

        contaminate()
        self.armature.duck_action_name = self.walk.name
        assert_canonical()

        contaminate()
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        assert_canonical()
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})

        contaminate()
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        assert_canonical()

    def test_retained_crouch_action_is_searchable_but_not_a_beginner_preset(self):
        self.assertTrue(hasattr(addon, "BEGINNER_ACTION_PRESETS"))
        self.assertEqual(
            addon.BEGINNER_ACTION_PRESETS,
            (("Policy_alpha_walking_forward", "Walk"),),
        )
        self.armature.duck_action_name = self.walk.name

        result = bpy.ops.duck.select_action(action_name=self.crouch.name)

        self.assertEqual(result, {"FINISHED"})
        self.assertIs(self.armature.animation_data.action, self.crouch)
        self.assertEqual(
            (
                bpy.context.scene.frame_start,
                bpy.context.scene.frame_end,
                bpy.context.scene.frame_current,
            ),
            (1, 51, 1),
        )

    def test_missing_preset_cancels_without_changing_animation(self):
        self.armature.duck_action_name = self.walk.name
        bpy.context.scene.frame_set(25)

        result = bpy.ops.duck.select_action(action_name="MissingAction")

        self.assertEqual(result, {"CANCELLED"})
        self.assertIs(self.armature.animation_data.action, self.walk)
        self.assertEqual(
            (
                bpy.context.scene.frame_start,
                bpy.context.scene.frame_end,
                bpy.context.scene.frame_current,
            ),
            (1, 200, 25),
        )

    def test_play_pause_and_reset_controls_real_screen_playback(self):
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertTrue(bpy.context.screen.is_animation_playing)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertFalse(bpy.context.screen.is_animation_playing)
        bpy.context.scene.frame_set(100)
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        self.assertEqual(bpy.context.scene.frame_current, 1)

    def test_play_once_restarts_from_action_start_at_end_and_disables_preview(self):
        scene = bpy.context.scene
        self.armature.duck_action_name = self.walk.name
        scene.use_preview_range = True
        scene.frame_preview_start = 10
        scene.frame_preview_end = 20
        scene.frame_set(200)

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})

        self.assertTrue(bpy.context.screen.is_animation_playing)
        self.assertEqual(scene.frame_current, 1)
        self.assertFalse(scene.use_preview_range)
        self.assertEqual((scene.frame_start, scene.frame_end), (1, 200))

    @unittest.skipUnless(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 5.2+ playback loop API",
    )
    def test_blender_5_2_play_once_uses_stop_end_frame_without_handler(self):
        scene = bpy.context.scene
        scene.playback_loop_mode = "INFINITE"
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})

        self.assertEqual(scene.playback_loop_mode, "STOP_END_FRAME")
        self.assertEqual(play_once_handlers(), [])

    @unittest.skipUnless(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 5.2+ playback loop API",
    )
    def test_blender_5_2_restores_native_loop_mode_on_every_terminal_path(self):
        scene = bpy.context.scene
        scene.playback_loop_mode = "BOUNCE"
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(scene.playback_loop_mode, "STOP_END_FRAME")
        self.assertEqual(len(native_playback_cleanup_handlers()), 1)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(scene.playback_loop_mode, "BOUNCE")
        self.assertEqual(native_playback_cleanup_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        cleanup = native_playback_cleanup_handlers()
        self.assertEqual(len(cleanup), 1)
        cleanup[0](scene, None)
        self.assertEqual(scene.playback_loop_mode, "BOUNCE")
        self.assertEqual(native_playback_cleanup_handlers(), [])

        if bpy.context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        self.assertEqual(scene.playback_loop_mode, "BOUNCE")
        self.assertEqual(native_playback_cleanup_handlers(), [])

    @unittest.skipIf(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 4.3 fallback only",
    )
    def test_blender_4_3_play_once_stops_at_endpoint_and_self_removes_handler(self):
        scene = bpy.context.scene
        self.armature.duck_action_name = self.walk.name

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        scene.frame_set(200)

        self.assertFalse(bpy.context.screen.is_animation_playing)
        self.assertEqual(scene.frame_current, 200)
        self.assertEqual(play_once_handlers(), [])

    @unittest.skipIf(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 4.3 fallback only",
    )
    def test_blender_4_3_handler_cleans_on_pause_reset_selection_and_unregister(self):
        self.armature.duck_action_name = self.walk.name
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(play_once_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        self.assertEqual(
            bpy.ops.duck.select_action(action_name=self.crouch.name), {"FINISHED"}
        )
        self.assertFalse(bpy.context.screen.is_animation_playing)
        self.assertEqual(play_once_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        self.assertEqual(bpy.ops.duck.reset_animation(), {"FINISHED"})
        self.assertEqual(play_once_handlers(), [])

        self.assertEqual(bpy.ops.duck.toggle_animation(), {"FINISHED"})
        self.assertEqual(len(play_once_handlers()), 1)
        addon.unregister()
        self.assertEqual(play_once_handlers(), [])

    @unittest.skipIf(
        hasattr(bpy.context.scene, "playback_loop_mode"),
        "Blender 4.3 fallback only",
    )
    def test_unregister_removes_temporary_handlers_from_every_scene(self):
        other_scene = bpy.data.scenes.new("OtherDuckScene")
        addon._install_play_once_handler(bpy.context.scene)
        addon._install_play_once_handler(other_scene)
        self.assertEqual(len(play_once_handlers()), 2)

        try:
            addon.unregister()
            self.assertEqual(play_once_handlers(), [])
        finally:
            addon._clear_play_once_handlers()


if __name__ == "__main__":
    unittest.main()
