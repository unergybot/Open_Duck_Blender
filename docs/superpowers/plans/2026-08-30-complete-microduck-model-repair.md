# Complete Microduck Blender Model Repair

**Goal:** Regenerate `microduck-alpha.blend` as a canonically faithful, grounded,
beginner-ready Blender 4.3.2 deliverable without modifying the legacy Mini Duck
asset or hand-adjusting robot geometry.

## Global constraints

- MJCF body/site transforms, the 38 referenced STLs, runtime joint ordering, and
  validated motion archives are authoritative. Never ground the robot with a
  Blender-only root/mesh offset.
- Work and open PRs only against the `unergybot` forks. Never push or merge to a
  Pollen Robotics remote.
- `open-duck-mini.blend` and the user's untracked backup remain unchanged.
- The mouth stays an explicitly disclosed image-derived visual approximation;
  it is not part of the 14-joint motion contract.
- The released `.blend` is saved by Blender 4.3.2 and reopened/tested in both
  Blender 4.3.2 and the installed Blender 5.2.1.
- Every behavior change follows red-green TDD. Generated binary updates happen
  only after their source and acceptance tests pass.

## Task 1: Grounded canonical policy rollout (`microduck_rl`)

Work in `/home/mcao/MyCode/microduck_rl/.worktrees/grounded-rollout`.

- Add `warmup_frames: int = 2` to `PolicyRolloutConfig` and
  `--warmup-frames` to the rollout CLI. Reject booleans, non-integers, and
  negative values.
- During warm-up, execute the exact same observation, ONNX inference, action,
  and four 5 ms MuJoCo substeps as a recorded control frame, but do not append a
  sample. Then record the requested duration unchanged (200 frames for 4 s).
- Include `warmup_frames` in the canonical rollout configuration digest and CLI
  summary. The NPZ schema stays version 1.
- Update the independent reference rollout and tests first, prove the new tests
  fail, then implement. Cover defaults, validation, zero warm-up compatibility,
  state-sensitive inference continuity, digest provenance, and atomic output.
- Run the full 177-test RL suite. Then export a 4 s rollout from
  `/home/mcao/MyCode/microduck/policies/alpha_walking.onnx`, validate it, and
  independently replay contacts: the released 200-frame window must have foot
  contact on 200/200 frames.
- Commit this repository independently.

## Task 2: Canonical skeleton, visual binding, and material integrity

Work in the Blender repair worktree.

- Add failing Blender tests that reproduce: zero-length matrix orientation loss,
  tail-based bone-parent offset, duplicate shared-mesh material slots, stale
  initial Cream colors, unvalidated geom transforms, and Blender 4.3's enum
  IDProperty/string collision in the colorway callback. The suite output must
  be exception-free in both supported versions.
- Create body and mouth bones by setting a nonzero display length before their
  exact MJCF world-rest matrix. Canonical body bone local Z remains the joint
  axis used by actions/export.
- Attach each visual with the real Blender bone-tail parent transform while
  preserving its pre-parent world matrix. At closed mouth, all 70 visuals match
  `body_world @ geom_local`; mouth-open transforms follow the linkage profile.
- Normalize every quaternion used by builder transforms and reject non-finite or
  zero quaternions/positions.
- Cache shared geometry/material combinations so each mesh datablock has one
  effective material slot. Apply Cream explicitly after visuals/materials exist;
  retain all four colorways.
- Add deterministic per-STL SHA-256 values plus a sorted aggregate visual-assets
  hash to build provenance. Preserve exact STL vertices and all 70 visual geoms.
- Verify focused red/green tests and the full Blender suite, then commit.

## Task 3: Strict motion import/export and one-shot action controls

- Add failing tests for inconsistent archived body transforms, non-50-Hz
  effective timing, Bézier interpolation, off-axis export loss, policy playback
  while IK is enabled, and looping root-moving actions.
- Retain full body position/quaternion arrays when loading and validate every
  frame against canonical FK from root pose plus joint positions before scene
  mutation. Tolerance: `1e-5 m` and `1e-5 rad`.
- Set imported location/rotation keys to linear interpolation. Export only when
  `fps / fps_base == 50` and reject any evaluated body transform whose residual
  from extracted canonical root/joint state exceeds the same tolerance.
- Store action metadata for motion kind, source SHA-256, and loopability.
  `Policy_alpha_walking_forward` is one-shot; Open Duck playback stops at frame
  200. Selecting/importing/playing an action disables IK before evaluation.
- Rename the retained demo action to `KinematicCrouchTest`, tag it as a
  non-contact-valid kinematic test, and remove the beginner Crouch preset.
- Keep action selection, Play Once, Reset, import, and export functional in both
  supported Blender versions. Verify full suite and commit.

## Task 4: Physical, constrained leg IK

- Add failing evaluated-rig tests showing the old arbitrary ankle-tail endpoint,
  off-axis leg rotation, missing limits, FK/IK playback corruption, and pose
  jumps.
- Extend the embedded profile with normalized `left_foot` and `right_foot`
  `SiteSpec` records parsed exactly once from the MJCF ankle bodies; bump the
  profile schema while retaining a clear error for older profiles that lack the
  physical IK contract.
- Keep canonical body/FK bones authoritative. Add a separate helper/control
  chain within `MicroduckRig` for each leg, rooted below the FK hip-yaw body and
  built from exact hip-roll, hip-pitch, knee, ankle, and named MJCF foot-site
  pivots. Use locked offset helpers where a pivot displacement has a component
  along a hinge axis; only hinge controls carry a canonical DOF.
- Add foot and pole controls plus one keyframeable sagittal foot-pitch channel;
  do not expose an unsatisfiable arbitrary 3-axis foot rotation. Solve the four
  bounded hinge angles with exact MJCF forward kinematics and a projected,
  damped least-squares solver using an analytic position Jacobian, deterministic
  fallback seeds, joint clamping, and the pole only as a branch selector. Map
  solved hinge angles onto canonical local-Z rotations only while IK is active.
- Use a guarded dependency-graph/frame-change update path for moved or keyed
  controls. Mark unreachable targets clamped while retaining the best finite
  bounded solution; never introduce off-axis motion to reduce target error.
- Lock canonical joint location/scale/non-Z rotation; lock the canonical body
  root and make armature-object motion the sole floating-base control. Protect
  property-driven mouth helpers.
- FK->IK and IK->FK are transactional and preserve every canonical body matrix
  within `1e-5`. Reachable target moves hit the foot site within `1e-5 m`;
  unreachable moves clamp to joint limits without off-axis residual. Selecting
  Walk restores FK and exact archive playback.
- Verify focused red/green tests and the full suite, then commit.

## Task 5: Beginner-ready scene, manifest, documentation, and artifact

- Build deterministic `Microduck` child collections for Rig, Visuals, and
  Controls plus a separate Presentation collection excluded from source counts.
- Add a presentation-only z=0 ground, camera, and key/fill lights. Derive ground
  extent and a three-quarter camera framing from the complete approximately
  0.50 m walking trajectory with margin.
- Save Object Mode with the armature selected, Stick display shown in front,
  material-visible viewport, Open Duck sidebar open, Walk active at grounded
  frame 1, and an explicit FK/IK mode indicator.
- Label the mouth control `Mouth (visual approximation)` and show the kinematic
  warning when that action is selected.
- Bump the manifest schema and record motion/source/STL hashes, mouth mode, and
  build Blender version. Embed current add-on sources and keep the file fully
  self-contained.
- Replace `assets/motions/alpha-walking-forward.npz` with Task 1's grounded
  archive, rebuild `microduck-alpha.blend` with Blender 4.3.2, and update
  contact-visible Walk/mouth previews and first-open documentation. Do not alter
  `open-duck-mini.blend`.
- Verify focused tests and commit the generated `.blend` through Git LFS.

## Task 6: Cross-version release gate and final review

- Add or update a reusable headless acceptance check covering exact canonical
  rest/body/visual matrices across all 200 Walk and 51 Kinematic Crouch frames,
  mouth poses, material/colorway hygiene, provenance, presentation isolation,
  actions, bootstrap, and missing external resources.
- Run the full Blender suite in official Blender 4.3.2 and installed Blender
  5.2.1. Open the 4.3.2-saved release in both fresh processes.
- Independently validate/export round-trips with `microduck_rl`, require Walk
  contact on 200/200 frames, and render-inspect Walk 1/100/200, Crouch 1/26/51,
  and mouth closed/open.
- Confirm hashes for `open-duck-mini.blend` and the user's backup are unchanged.
- Run an independent whole-branch code/spec review and fix all Critical or
  Important findings before presenting the branches for fork-only integration.
