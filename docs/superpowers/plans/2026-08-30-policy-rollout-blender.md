# Policy Rollout to Blender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export a deterministic four-second `alpha_walking.onnx` MuJoCo rollout and ship it as a root-moving, editable Blender action.

**Architecture:** `microduck_rl` owns ONNX execution, MuJoCo stepping, native motion generation, and kinematic validation. `Open_Duck_Blender` owns strict archive loading, root/joint keyframing, UI import, and the final two-action `.blend`; Blender never imports ONNX Runtime or MuJoCo.

**Tech Stack:** Python 3.12, NumPy, ONNX Runtime, MuJoCo, Blender 5.2 Python API, `unittest`, `pytest`, Git LFS.

**Spec:** `docs/superpowers/specs/2026-08-30-policy-rollout-blender-design.md`

## Global Constraints

- Use `/home/mcao/MyCode/microduck/policies/alpha_walking.onnx` unchanged.
- Generate 200 frames at exactly 50 Hz from a 4.0-second rollout.
- Use command `(0.30, 0.0, 0.0)` and deterministic seed `0`.
- Preserve exact 14-joint and 15-body canonical ordering; never silently reorder.
- Keep the mouth visual-only and absent from the motion archive.
- Do not add ONNX Runtime or MuJoCo to Blender.
- Preserve `MicroduckCrouchTest` and add `Policy_alpha_walking_forward`.
- All GitHub integration targets `unergybot` forks only.

### Final-review amendment

The user-approved `0.30 m/s` command supersedes the initial `0.15 m/s`
proposal because the lower-command candidate did not meet the required
`>0.1 m` forward-displacement gate. The corrected exporter must also set the
MuJoCo timestep to `0.005 s`; four substeps then equal one `0.02 s` / 50 Hz
control frame. Regenerated acceptance evidence at command `(0.30, 0.0, 0.0)`,
duration `4.0`, and seed `0` contains 200 finite frames, advances
`0.486906945705 m`, and has archive SHA-256
`822a1fbde45f31c7b703d09a225115f430672fd0ba4873d97201b35832348b54`.
The archive must also contain canonical rollout-configuration SHA-256
`eb7e3697bc1f166a458a080867f9fcf02f5c8005a404430a06b1437eb7187298`,
covering command, duration, seed, timestep, decimation, and control rate.

---

### Task 1: Headless ONNX Policy Rollout Core

**Repository:** `/home/mcao/MyCode/microduck_rl/.worktrees/policy-rollout-blender`

**Files:**
- Create: `src/mjlab_microduck/policy_rollout.py`
- Create: `tests/test_policy_rollout.py`
- Reference: `scripts/infer_policy.py`
- Reference: `src/mjlab_microduck/blender_motion.py`
- Reference: `src/mjlab_microduck/robot/microduck/scene.xml`

**Interfaces:**
- Consumes: an ONNX model with `obs[1,61] -> actions[1,14]`, canonical MuJoCo `scene.xml`, velocity command, duration, and seed.
- Produces: `PolicyRolloutConfig` and `export_policy_rollout(config: PolicyRolloutConfig) -> Path`.

- [ ] **Step 1: Write configuration and contract tests**

Create tests that construct a real constant-output ONNX model with ONNX helper APIs, including the exact 14 joint names in metadata. Assert:

```python
def test_rejects_duration_without_integral_50hz_frames(tmp_path, policy_path):
    cfg = PolicyRolloutConfig(policy_path, tmp_path / "out.npz", duration_s=0.011)
    with pytest.raises(PolicyRolloutError, match="integral number of 50 Hz frames"):
        export_policy_rollout(cfg)

@pytest.mark.parametrize("input_width,output_width", [(60, 14), (61, 13)])
def test_rejects_incompatible_onnx_contract(tmp_path, make_policy, input_width, output_width):
    policy = make_policy(input_width=input_width, output_width=output_width)
    cfg = PolicyRolloutConfig(policy, tmp_path / "out.npz", duration_s=0.02)
    with pytest.raises(PolicyRolloutError, match=r"\[1,61\].*\[1,14\]"):
        export_policy_rollout(cfg)

def test_rejects_joint_metadata_order_drift(tmp_path, make_policy):
    policy = make_policy(joint_names=tuple(reversed(EXPECTED_JOINT_NAMES)))
    with pytest.raises(PolicyRolloutError, match="joint_names.*index 0"):
        export_policy_rollout(PolicyRolloutConfig(policy, tmp_path / "out.npz", 0.02))
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
cd /home/mcao/MyCode/microduck_rl/.worktrees/policy-rollout-blender
uv run --with pytest pytest tests/test_policy_rollout.py -v
```

Expected: collection fails because `mjlab_microduck.policy_rollout` does not exist.

- [ ] **Step 3: Implement strict configuration and ONNX validation**

Create:

```python
class PolicyRolloutError(ValueError):
    pass

@dataclass(frozen=True)
class PolicyRolloutConfig:
    policy_path: Path
    output_path: Path
    duration_s: float = 4.0
    command: tuple[float, float, float] = (0.30, 0.0, 0.0)
    seed: int = 0
```

Validate finite command values, positive duration, exact integral frame count at 50 Hz, file existence, input/output names and shapes, and comma-separated `joint_names` metadata against canonical MuJoCo joint order. Error messages must include the differing index and names.

- [ ] **Step 4: Write the real rollout behavior test**

Use the real constant-output ONNX fixture and canonical MuJoCo scene. Export 3 frames (`duration_s=0.06`) and assert:

```python
archive = np.load(result, allow_pickle=False)
assert archive["joint_pos"].shape == (3, 14)
assert archive["body_pos_w"].shape == (3, 15, 3)
assert archive["fps"].tolist() == [50]
assert tuple(archive["joint_names"]) == EXPECTED_JOINT_NAMES
assert json.loads(str(archive["source_hashes_json"][0]))["policy_sha256"] == sha256(policy)
assert validate_motion(result).frames == 3
```

- [ ] **Step 5: Run the behavior test and verify RED**

Run the individual behavior test. Expected: failure because rollout generation is not implemented.

- [ ] **Step 6: Implement deterministic rollout sampling**

Implement the 61-value observation in this order:

```text
base angular velocity (3)
projected gravity (3)
joint position relative to DEFAULT_POSE (14)
joint velocity (14)
previous action (14)
command [vx, vy, wz, head(4)=0, body(6)=0] (13)
```

Initialize the free root at `(0, 0, 0.125)` and the servo joints/control at `DEFAULT_POSE`. For each frame: call `mujoco.mj_forward`, record canonical joint/body state, infer one action, set `ctrl = DEFAULT_POSE + action`, then call `mujoco.mj_step` four times. Build with `build_motion_archive`, write to a temporary sibling, call `validate_motion`, then atomically replace the destination.

- [ ] **Step 7: Run rollout and validator tests**

Run:

```bash
uv run --with pytest pytest tests/test_policy_rollout.py tests/test_blender_motion_validator.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit Task 1**

```bash
git add src/mjlab_microduck/policy_rollout.py tests/test_policy_rollout.py
git commit -m "feat: export deterministic ONNX policy rollouts"
```

---

### Task 2: Rollout CLI and Walking Milestone Archive

**Repository:** `/home/mcao/MyCode/microduck_rl/.worktrees/policy-rollout-blender`

**Files:**
- Create: `scripts/export_policy_rollout.py`
- Create: `tests/test_policy_rollout_cli.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `PolicyRolloutConfig` and `export_policy_rollout` from Task 1.
- Produces: a command-line exporter and `/tmp/alpha-walking-forward.npz` during acceptance.

- [ ] **Step 1: Write CLI parsing and error tests**

Import the script as a module and assert `_arguments()` maps the approved defaults exactly. Run the CLI against a missing policy with `subprocess.run` and assert exit code `2` plus `policy file does not exist` on stderr.

- [ ] **Step 2: Run CLI tests and verify RED**

Expected: failure because `scripts/export_policy_rollout.py` does not exist.

- [ ] **Step 3: Implement the thin CLI**

Accept positional `policy`, required `--output`, `--duration`, `--lin-vel-x`, `--lin-vel-y`, `--ang-vel-z`, and `--seed`. Catch `PolicyRolloutError`, `OSError`, and `ValueError`; print one `Policy rollout failed: ...` line and return `2`. On success print the output path, frame count, command, policy hash, and validator errors.

- [ ] **Step 4: Run CLI tests and the full RL suite**

```bash
uv run --with pytest pytest tests/test_policy_rollout_cli.py -v
uv run --with pytest pytest tests/ -q
```

Expected: CLI tests pass; full suite has no new failures.

- [ ] **Step 5: Generate the real 200-frame walk**

```bash
uv run scripts/export_policy_rollout.py \
  /home/mcao/MyCode/microduck/policies/alpha_walking.onnx \
  --output /tmp/alpha-walking-forward.npz \
  --duration 4 \
  --lin-vel-x 0.30 \
  --seed 0
uv run scripts/validate_blender_motion.py /tmp/alpha-walking-forward.npz
```

Expected: `Validated 200 frames`; position and orientation errors are each below `1e-4`.

- [ ] **Step 6: Check that the rollout visibly moves forward**

Load the archive and assert:

```python
root = archive["body_pos_w"][:, 0]
assert root[-1, 0] - root[0, 0] > 0.1
assert np.isfinite(root).all()
```

If it falls before frame 200, preserve the evidence and stop for diagnosis; do not hide the failure by shortening the rollout or editing state samples.

- [ ] **Step 7: Document and commit Task 2**

Add the exact CLI example and output contract to `README.md`, then:

```bash
git add scripts/export_policy_rollout.py tests/test_policy_rollout_cli.py README.md
git commit -m "feat: add headless policy rollout CLI"
```

---

### Task 3: Transactional Blender Motion Loader

**Repository:** `/home/mcao/MyCode/Open_Duck_Blender/.worktrees/policy-rollout-blender`

**Files:**
- Create: `open_duck_tools/motion_import.py`
- Create: `tests/test_motion_import.py`
- Modify: `open_duck_tools/builder.py`
- Modify: `open_duck_tools/__init__.py`

**Interfaces:**
- Consumes: native mjlab NPZ archives and `RobotProfile`.
- Produces: `load_motion(path, profile) -> ImportedMotion` and `import_motion_action(armature, profile, path, *, action_name) -> bpy.types.Action`.

- [ ] **Step 1: Write side-effect-free loader tests**

Create a 2-joint/3-body archive fixture with complete native keys. Assert exact 50 Hz, names, shapes, finite values, joint limits, normalized quaternion output, and continuous quaternion signs. Add parameterized failures for missing/extra keys, wrong names, non-finite values, zero quaternions, and joint-limit violations. Each failure must be `MotionError` and name the field/frame/index.

- [ ] **Step 2: Run loader tests and verify RED**

Run the Blender unittest loader for `tests.test_motion_import`; expected import failure because the module does not exist.

- [ ] **Step 3: Implement `ImportedMotion` and `load_motion`**

Use:

```python
@dataclass(frozen=True)
class ImportedMotion:
    joint_pos: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    fps: int
    frames: int
```

Read with `allow_pickle=False`, validate the complete native schema before mutation, copy arrays out of the NPZ context, normalize root quaternions, and flip signs where consecutive dot products are negative.

- [ ] **Step 4: Write Blender action and transaction tests**

Build the existing minimal test rig and import three frames with known root transforms. Assert:

```python
assert action.name == "Walk"
assert action.use_fake_user
assert scene.frame_start == 1 and scene.frame_end == 3
scene.frame_set(3)
assert_vector_close(armature.location, expected_location)
assert_quaternion_close(armature.rotation_quaternion, expected_rotation)
assert pose_bone.rotation_euler.z == pytest.approx(expected_joint)
```

Pre-create a `Walk` action and assert the import receives a numeric suffix. Force a keyframe error after validation and assert the previous action, frame range, object transform, and pose values are restored and the partial action is removed.

- [ ] **Step 5: Run action tests and verify RED**

Expected: loader tests pass but action tests fail because `import_motion_action` is absent.

- [ ] **Step 6: Implement calibrated root and joint keyframing**

Derive `root_mjcf_rest` from the root `BodySpec`; for each frame set:

```python
desired_root_world = Matrix.Translation(root_pos) @ Quaternion(root_quat).to_matrix().to_4x4()
armature.matrix_world = desired_root_world @ root_mjcf_rest.inverted_safe()
```

Key armature `location` and `rotation_quaternion`, and each joint child pose bone's local Z Euler channel. Create and assign a fresh action before key insertion, set `use_fake_user=True`, set constant 50 Hz/frame range, and restore prior state on exceptions.

- [ ] **Step 7: Replace the builder's private joint-only importer**

Delete `_create_demo_action`. Call `import_motion_action(..., action_name="MicroduckCrouchTest")`, then key the crouch-only triangular `duck_mouth_open` curve into that active action. Add `motion_import` to the embedded module list in `_embed_addon`.

- [ ] **Step 8: Run all Blender tests and commit Task 3**

```bash
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests',top_level_dir='.')); raise SystemExit(not r.wasSuccessful())"
git add open_duck_tools/motion_import.py open_duck_tools/builder.py open_duck_tools/__init__.py tests/test_motion_import.py tests/test_builder.py
git commit -m "feat: import root-moving mjlab actions"
```

---

### Task 4: Blender Import Operator and Sidebar UI

**Repository:** `/home/mcao/MyCode/Open_Duck_Blender/.worktrees/policy-rollout-blender`

**Files:**
- Modify: `open_duck_tools/addon.py`
- Modify: `tests/test_addon.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `import_motion_action` from Task 3.
- Produces: `bpy.ops.duck.import_motion(filepath=..., action_name=...)` and an **Import mjlab Motion** sidebar button.

- [ ] **Step 1: Write operator registration, success, and failure tests**

Assert `DUCK_OT_import_motion` registers idempotently, appears in `CLASSES`, derives a sanitized action name from the file stem when `action_name` is empty, and calls the real importer on a fixture archive. For malformed input, assert `{"CANCELLED"}` and no action/frame/transform changes.

- [ ] **Step 2: Run addon tests and verify RED**

Expected: failure because `bpy.ops.duck.import_motion` is unavailable.

- [ ] **Step 3: Implement the import operator**

Subclass `Operator` and `ImportHelper`; use `.npz` filter, optional `StringProperty` action name, active-armature/profile validation, and `self.report`. Add the operator immediately before export in `DUCK_PT_tools.draw`.

- [ ] **Step 4: Run addon and full Blender suites**

Expected: all pass with the new operator registered and unregisterable.

- [ ] **Step 5: Document the UI workflow and commit Task 4**

Document export in `microduck_rl`, then **Open Duck → Import mjlab Motion**, action selection, root motion, and mouth exclusion.

```bash
git add open_duck_tools/addon.py tests/test_addon.py README.md
git commit -m "feat: add mjlab motion import UI"
```

---

### Task 5: Version and Build the Walking Milestone Artifact

**Repositories:** both feature worktrees.

**Files:**
- Create: `Open_Duck_Blender/assets/motions/alpha-walking-forward.npz`
- Modify: `Open_Duck_Blender/tools/build_microduck_blend.py`
- Modify: `Open_Duck_Blender/tests/test_build_cli.py`
- Modify: `Open_Duck_Blender/microduck-alpha.blend`

**Interfaces:**
- Consumes: `/tmp/alpha-walking-forward.npz` from Task 2 and `import_motion_action` from Task 3.
- Produces: a deterministic build with `MicroduckCrouchTest` plus `Policy_alpha_walking_forward`.

- [ ] **Step 1: Copy the validated rollout into versioned assets**

Copy `/tmp/alpha-walking-forward.npz` to `assets/motions/alpha-walking-forward.npz`. Validate the copied file from the RL worktree and compare SHA-256 values to prove the copy is exact.

- [ ] **Step 2: Write failing build-default and two-action tests**

Assert `_arguments([]).policy_motion` equals the versioned archive. Build to a temporary blend, reopen it, and assert exact action names, fake-user persistence, frame rates, and frame ranges. `Policy_alpha_walking_forward` must be the active action at frame 1.

- [ ] **Step 3: Run build tests and verify RED**

Expected: missing `policy_motion` argument and absent walking action.

- [ ] **Step 4: Add deterministic walking import to the builder**

Add `--policy-motion` defaulting to `assets/motions/alpha-walking-forward.npz`. Require it during source validation. After creating and preserving `MicroduckCrouchTest`, import the walking archive as `Policy_alpha_walking_forward`, leave it active, and record its SHA-256 in `microduck-build-manifest.json`.

- [ ] **Step 5: Rebuild the tracked blend from fork paths**

```bash
blender --background --factory-startup \
  --python tools/build_microduck_blend.py -- \
  --runtime-root /home/mcao/MyCode/microduck \
  --rl-root /home/mcao/MyCode/microduck_rl/.worktrees/policy-rollout-blender \
  --output microduck-alpha.blend
```

Remove only Blender's untracked `.blend1` backup through trash after verifying the new `.blend` opens.

- [ ] **Step 6: Run full Blender verification and commit Task 5**

```bash
git add assets/motions/alpha-walking-forward.npz tools/build_microduck_blend.py tests/test_build_cli.py microduck-alpha.blend
git commit -m "feat: ship alpha walking Blender milestone"
```

---

### Task 6: End-to-End Round Trip and Visual Evidence

**Repositories:** both feature worktrees.

**Files:**
- Create: `Open_Duck_Blender/assets/previews/microduck-policy-walk-start.png`
- Create: `Open_Duck_Blender/assets/previews/microduck-policy-walk-mid.png`
- Create: `Open_Duck_Blender/assets/previews/microduck-policy-walk-end.png`
- Modify: `Open_Duck_Blender/README.md`

**Interfaces:**
- Consumes: the final blend, walking archive, Blender import/export operator, and RL validator.
- Produces: round-trip evidence and three reviewer-visible renders.

- [ ] **Step 1: Export the active walking action through the Blender operator**

Open `microduck-alpha.blend` headlessly, select `MicroduckRig`, set the walking action active, invoke `bpy.ops.duck.export_motion(filepath="/tmp/alpha-walking-roundtrip.npz")`, and assert `{"FINISHED"}`.

- [ ] **Step 2: Validate and numerically compare the round trip**

Validate with `scripts/validate_blender_motion.py`. Compare source and round-trip archives after quaternion sign alignment:

```python
assert np.max(np.abs(source["joint_pos"] - result["joint_pos"])) <= 1e-5
assert np.max(np.abs(source["body_pos_w"][:, 0] - result["body_pos_w"][:, 0])) <= 1e-6
dots = np.abs(np.sum(source["body_quat_w"][:, 0] * result["body_quat_w"][:, 0], axis=1))
assert np.max(2 * np.arccos(np.clip(dots, 0, 1))) <= 1e-5
```

- [ ] **Step 3: Inspect the live action through Blender MCP**

Open the final blend, confirm `Policy_alpha_walking_forward`, inspect frames 1, 100, and 200, and verify root X increases while the rig remains finite and upright. If the visual result contradicts the numerical archive, stop and diagnose before rendering.

- [ ] **Step 4: Render start/mid/end previews**

Use a temporary camera and three-point lights, render transparent 800×800 PNGs, and remove temporary scene objects before saving. Inspect all three PNGs locally.

- [ ] **Step 5: Run final suites and diff checks**

```bash
cd /home/mcao/MyCode/Open_Duck_Blender/.worktrees/policy-rollout-blender
git diff --check
blender --background --factory-startup --python-expr "import os,sys,unittest; sys.path.insert(0,os.getcwd()); r=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover('tests',top_level_dir='.')); raise SystemExit(not r.wasSuccessful())"

cd /home/mcao/MyCode/microduck_rl/.worktrees/policy-rollout-blender
git diff --check
uv run --with pytest pytest tests/ -q
```

- [ ] **Step 6: Commit visual evidence**

```bash
git add README.md assets/previews/microduck-policy-walk-*.png
git commit -m "docs: show policy walking milestone"
```

- [ ] **Step 7: Request independent code review**

Provide reviewers both repository base/head ranges, the approved spec, test counts, rollout validator output, round-trip tolerances, and preview paths. Fix all Critical and Important findings with a failing regression test before publishing.

- [ ] **Step 8: Publish fork-local PRs**

Push both `feat/policy-rollout-blender` branches to `fork`; create PRs targeting `unergybot/microduck_rl:develop` and `unergybot/Open_Duck_Blender:main`. Cross-link them and include the three preview images. Do not create or modify PRs under `pollen-robotics`.
