# Policy Rollout to Blender Design

## Goal

Run the vendored `alpha_walking.onnx` policy in canonical Microduck MuJoCo,
record a deterministic four-second forward walk, and import it into
`microduck-alpha.blend` as a root-moving Blender action that can be inspected,
edited, and rendered.

The first milestone uses:

- policy: `~/MyCode/microduck/policies/alpha_walking.onnx`
- command: `(0.15, 0.0, 0.0)` for forward, lateral, and yaw velocity
- duration: 4.0 seconds
- control rate: 50 Hz
- output frames: 200
- Blender action: `Policy_alpha_walking_forward`

## Repository Boundaries

`microduck_rl` owns policy execution and physics. It converts an ONNX policy
plus explicit command parameters into the existing native mjlab motion archive.

`Open_Duck_Blender` owns archive-to-action conversion and Blender UI. It never
runs ONNX Runtime or MuJoCo inside Blender. This keeps Blender dependencies
small and makes the simulation result independently reproducible and validatable.

The `microduck` repository supplies the versioned ONNX policy and runtime joint
contract. It is read-only in this milestone.

## Rollout Exporter

`microduck_rl` will add a package module for deterministic, headless policy
rollouts and a thin CLI script. The exporter will reuse the same observation
and action semantics as `scripts/infer_policy.py`:

- canonical `scene.xml` MuJoCo model
- initial root pose `(0, 0, 0.125)` with identity quaternion
- the 14-value `DEFAULT_POSE`
- projected gravity rather than raw accelerometer
- unified 61-value observation with 13 command values
- action target `DEFAULT_POSE + action * action_scale`
- four MuJoCo substeps per 50 Hz policy step

The package boundary will expose a typed configuration and one function:

```python
@dataclass(frozen=True)
class PolicyRolloutConfig:
    policy_path: Path
    output_path: Path
    duration_s: float = 4.0
    command: tuple[float, float, float] = (0.15, 0.0, 0.0)
    seed: int = 0

def export_policy_rollout(config: PolicyRolloutConfig) -> Path:
    ...
```

The CLI will be:

```bash
uv run scripts/export_policy_rollout.py \
  ~/MyCode/microduck/policies/alpha_walking.onnx \
  --output /tmp/alpha-walking-forward.npz \
  --duration 4 \
  --lin-vel-x 0.15 \
  --seed 0
```

The exporter records the state before each policy step, producing exactly 200
frames. Each frame contains joint positions and canonical body transforms in
the same 14-joint/15-body order already enforced by
`mjlab_microduck.blender_motion`. It writes the existing motion schema with a
50 Hz rate, source hashes, and additional provenance encoded in
`source_hashes_json`: policy SHA-256 plus rollout configuration SHA-256.

Before returning, the exporter calls the existing MuJoCo motion validator on
its own output. A failed validation removes no existing user file: generation
occurs at a temporary sibling path and atomically replaces the destination only
after validation succeeds.

## Blender Motion Importer

Open Duck Blender will move the current demo-action creation logic out of the
scene builder into a focused `open_duck_tools.motion_import` module. The module
will expose:

```python
def import_motion_action(
    armature: bpy.types.Object,
    profile: RobotProfile,
    motion_path: Path,
    *,
    action_name: str,
) -> bpy.types.Action:
    ...
```

The importer validates before modifying the scene:

- exact archive keys and finite array values
- `fps == 50`
- exact profile joint and body names in canonical order
- joint array shape `[T, 14]`
- body position shape `[T, 15, 3]`
- body quaternion shape `[T, 15, 4]` with nonzero normalized quaternions
- at least one frame
- joint values within the profile limits

After validation, it creates a new action without overwriting an unrelated
action. Name collisions receive Blender's normal numeric suffix. It keys every
policy joint on the generated pose bone's local Z rotation.

For root motion, the archive's first body is the canonical MJCF root body. The
armature object transform for each frame is solved from the same calibration
used by export:

```text
armature_world = desired_root_world @ inverse(root_mjcf_rest)
```

This makes a subsequent Blender export reconstruct the original canonical root
pose rather than double-applying the MJCF rest transform. Location and WXYZ
quaternion channels are keyed on the armature object. Quaternion signs are made
continuous before keying to prevent long-path interpolation.

The importer sets the scene to 50 Hz and the imported frame range. It leaves
`duck_mouth_open` unchanged because the mouth is outside the policy and motion
archive.

The deterministic crouch build will call this same importer with
`MicroduckCrouchTest`, eliminating the separate joint-only import path.

## Blender UI

The existing **Open Duck** sidebar gains **Import mjlab Motion** below the
export operator. The file picker accepts `.npz`. On success it selects the new
action and reports its name and frame count. On failure it reports a concise
validation error and leaves the current action, keyframes, frame range, and
object transform unchanged.

The milestone artifact will import the generated walk into the tracked
`microduck-alpha.blend`, leaving both actions available:

- `MicroduckCrouchTest`
- `Policy_alpha_walking_forward`

## Error Handling

The rollout CLI rejects missing policies, incompatible ONNX input/output
shapes, missing 14-joint metadata, non-finite commands, nonpositive duration,
and durations that do not resolve to an integral number of 50 Hz frames.

The Blender importer translates NPZ, schema, profile, and Blender action errors
into `MotionError` or `ProfileError`. Validation is side-effect-free. Scene
mutation starts only after all arrays and transforms pass validation.

No component silently reorders joints or bodies. A mismatch identifies the
first differing name and index.

## Testing and Acceptance

`microduck_rl` unit tests will cover configuration validation, ONNX metadata
and shape checks, exact frame count, state sampling order, and deterministic
output for a fixed seed. Existing validator tests remain authoritative for
archive kinematics.

Open Duck Blender tests will cover archive validation, root transform
calibration, quaternion continuity, 14-joint key creation, transactional
failure behavior, action-name collisions, and operator registration.

The acceptance test runs the real vendored policy end to end:

1. Export the four-second walk from `alpha_walking.onnx`.
2. Validate all 200 frames in canonical MuJoCo.
3. Import it into a freshly generated Blender project.
4. Confirm the named action, frame range, 50 Hz rate, and nonzero forward root
   displacement.
5. Re-export the Blender action through the UI operator.
6. Validate the re-export and compare joint/root arrays to the source within
   `1e-5 rad`, `1e-6 m`, and `1e-5 rad` orientation error.
7. Open the final `.blend` through Blender MCP and visually inspect frames 1,
   100, and 200 before publishing preview evidence.

## Out of Scope

- running ONNX or MuJoCo inside Blender
- interactive keyboard commands during rollout generation
- policy switching or multi-policy choreography
- roller, kick, sit/stand, ground-pick, or roulade imports
- automatic camera tracking or final cinematic rendering
- changes to the ONNX policies or Microduck runtime

Those behaviors can reuse the archive and importer boundaries after the first
walking milestone is proven.
