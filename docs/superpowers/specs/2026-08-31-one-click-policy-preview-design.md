# One-Click Walking Policy Preview Design

## Goal

Let a Blender beginner select the bundled Microduck walking ONNX policy, enter
a velocity command, and generate a new editable Blender action without leaving
the **Open Duck** sidebar. Blender remains responsive while the rollout runs.

The first milestone supports walking policies only. It defaults to:

- policy: `~/MyCode/microduck/policies/alpha_walking.onnx`
- command: forward `0.30 m/s`, lateral `0.00 m/s`, yaw `0.00 rad/s`
- duration: `4.0 s`
- seed: `0`
- runtime checkout: `~/MyCode/microduck`
- rollout checkout: `~/MyCode/microduck_rl`

Other policy command contracts, live inference inside Blender, and non-Linux
process handling are outside this milestone.

## Repository Boundary

`microduck_rl` remains the only component that runs ONNX Runtime and MuJoCo.
Blender does not import either dependency. It launches the existing validated
rollout CLI as a child process and imports the resulting native `.npz` archive
through `open_duck_tools.motion_import`.

This preserves the existing ownership boundary:

- `microduck` supplies the versioned ONNX policy and runtime contract.
- `microduck_rl` owns policy observations, inference, physics, rollout
  validation, and archive generation.
- `Open_Duck_Blender` owns user input, background-job state, cancellation,
  archive import, and action presentation.

The first milestone uses a background subprocess rather than importing
`microduck_rl` into Blender or introducing a persistent local service. The
subprocess reuses the proven exporter while avoiding dependency conflicts with
Blender's bundled Python.

## Sidebar Interface

The **Open Duck** sidebar gains a **Generate Policy Preview** section for a
selected Microduck armature. It contains:

- an ONNX policy path, defaulting to `alpha_walking.onnx` in the sibling
  `microduck` checkout;
- forward velocity, lateral velocity, and yaw-rate fields;
- duration in seconds, defaulting to `4.0`;
- an integer seed, defaulting to `0`;
- an expandable Setup area containing the `microduck` and `microduck_rl`
  checkout paths;
- **Generate & Import** while idle and **Cancel** while running;
- a short status line for preflight, export, validation, import, completion,
  cancellation, or failure;
- a bounded details field for actionable stderr when a job fails.

Repository paths are initialized from sibling checkouts under `~/MyCode` and
stored on the armature so the self-contained file retains explicit overrides.
The paths remain editable and are never inferred silently after the user has
overridden them.

The generated action name describes the walking command, for example
`PolicyWalk_x0.20_y0.00_yaw0.00`. The existing importer keeps Blender's normal
numeric suffix on collisions, so every successful click creates a new action
and preserves the shipped `Policy_alpha_walking_forward` milestone.

## Preflight and Command Construction

Preflight completes before creating a process or changing the scene. It checks:

- the selected object is the profiled Microduck armature;
- no preview job is already active;
- `uv` resolves to an executable;
- the two repository paths are directories;
- `scripts/export_policy_rollout.py` exists in `microduck_rl`;
- the selected `.onnx` policy exists and is a file;
- forward, lateral, yaw, and duration are finite;
- duration is positive and resolves to an integral number of 50 Hz frames;
- the seed is representable as an integer accepted by the exporter.

The subprocess command is built as an argument list, never as a shell string:

```text
uv run scripts/export_policy_rollout.py
  <policy.onnx>
  --output <temporary-cache-file.npz>
  --duration <seconds>
  --lin-vel-x <forward>
  --lin-vel-y <lateral>
  --ang-vel-z <yaw>
  --seed <seed>
```

The child runs with the `microduck_rl` checkout as its working directory.
Paths and numeric values are passed as individual arguments, so spaces and
shell metacharacters cannot change command meaning.

## Cache and Provenance

The rollout cache lives outside the repository under Blender's user cache area,
resolved with `bpy.utils.user_resource("CACHE", path="open_duck/policy_previews",
create=True)`.
Its key is the SHA-256 digest of canonical compact JSON containing:

- policy file SHA-256;
- resolved policy path for diagnostics;
- forward, lateral, and yaw commands in canonical decimal form;
- duration;
- seed;
- exporter contract version.

Generation targets a temporary sibling file. The exporter already validates
before replacing its output; Blender additionally loads the completed archive
through its strict importer before treating the cache entry as usable. Failed
or cancelled temporary files are removed. A validated exact-key cache hit may
skip policy execution, but it still creates a new Blender action as requested.

The imported action retains the archive's policy and rollout-configuration
hashes. Cache identity is an optimization, not a substitute for archive
validation.

## Background Job Lifecycle

One process-global job registry maps the running job to its armature and scene.
Only one policy-preview child may run at a time because Blender registration,
timers, and file loading are process-global.

Starting a job does not mutate or snapshot Blender animation state. This lets
the user continue inspecting the file without a later cancellation restoring
stale values over newer edits. A Blender-independent reader thread drains the
child's merged stdout/stderr stream so the OS pipe cannot fill and deadlock the
exporter. The thread only updates a locked, fixed-size tail buffer; it never
accesses `bpy`. A Blender timer polls process state and copies status from that
buffer without blocking the UI. The state machine is:

```text
idle -> preflight -> exporting -> validating -> importing -> completed -> idle
                    |             |             |
                    +---------- failed/cancelled +----------> idle
```

On successful process exit, the timer validates the archive without mutation,
then invokes the existing transactional importer. The importer snapshots the
current action, frame range, current frame, armature transform, pose, FK/IK
state, and mouth value immediately before its first mutation. Success activates
the new action, selects frame 1, and reports its name and frame count; failure
restores that just-in-time snapshot.

**Cancel**, add-on unregister, file reload, or Blender shutdown terminates the
exact child process. The implementation first requests normal termination,
then escalates only if the child does not exit within a short bounded grace
period. It unregisters its timer, closes pipes, joins the reader thread with a
bounded timeout, deletes an incomplete temporary file, clears the job registry,
and reports cancellation when the UI still exists.

Captured logs use a fixed byte ceiling and retain the tail, where CLI errors
normally appear. A noisy child cannot consume unbounded Blender memory.

## Transactional Failure Behavior

Preflight failures do not start a process. Export, validation, cancellation,
and import failures preserve:

- the active action and all existing actions;
- scene frame start, end, and current frame;
- armature object location and rotation;
- the mouth property;
- all existing cache entries and user-selected paths.

An incomplete action is removed if Blender raises after action creation.
Errors are translated to concise sidebar messages. The bounded detail field
distinguishes missing `uv`, missing repositories or files, invalid numeric
input, incompatible ONNX shape or metadata, nonzero child exit, invalid archive,
and cancellation.

## Code Structure

The implementation adds a focused, Blender-independent job module for:

- validated preview configuration;
- canonical cache keys and action names;
- safe argv construction;
- process launch, polling, log bounds, and termination;
- immutable completion and failure results.

`open_duck_tools.addon` owns Blender properties, operators, timer registration,
scene snapshots, archive import, UI drawing, and cleanup hooks. Process logic
does not reach into Blender data, and UI code does not duplicate rollout or
archive validation.

The new module is embedded into generated `.blend` files alongside the existing
add-on modules. No ONNX, MuJoCo, policy, or rollout archive is embedded beyond
the already shipped milestone assets.

## Testing

Blender-independent tests cover:

- finite input and integral-frame validation;
- missing path and executable diagnostics;
- safe argument-array construction for paths containing spaces and shell
  metacharacters;
- deterministic cache keys and readable action names;
- cache hit and miss behavior;
- bounded log capture;
- process state transitions and termination escalation.

Blender tests cover:

- operator and property registration;
- single-job exclusion;
- timer polling and UI state;
- successful archive import and collision-safe action creation;
- preservation of action, frame range, transform, and mouth state on every
  failure path;
- Cancel, unregister, and file-load cleanup;
- embedding the new module into the release artifact.

End-to-end acceptance runs the real
`~/MyCode/microduck/policies/alpha_walking.onnx` policy through the sibling
`microduck_rl` checkout, waits without blocking Blender's main loop, validates
the generated archive, imports a 200-frame action, re-exports it, and validates
the round trip. The complete Open Duck Blender suite runs in Blender 4.3.2 and
5.2.1. Final MCP inspection confirms grounded playback and the expected root
motion.

## Acceptance Criteria

The milestone is complete when:

1. A beginner can generate and import the default four-second walking preview
   from the Open Duck sidebar without using a terminal.
2. Blender stays responsive and Cancel reliably returns the UI to idle.
3. Each success preserves existing actions and activates a newly named action.
4. Exact configurations may reuse validated cache output without weakening
   provenance checks.
5. Every failure is transactional and actionable.
6. The shipped `.blend` remains self-contained apart from the explicitly
   configured sibling repositories and `uv` needed for new generation.
7. Blender 4.3.2 and 5.2.1 tests and the real-policy acceptance run pass.
