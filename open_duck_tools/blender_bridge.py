"""Translate Blender's evaluated body matrices into robot state samples."""

from __future__ import annotations

import math

from mathutils import Matrix, Quaternion, Vector
import numpy as np


MATRIX_RESIDUAL_TOLERANCE = 1e-5


def matrix_residual(actual: Matrix, expected: Matrix) -> tuple[float, float, float]:
    """Return translation, sign-invariant rotation, and affine residuals."""
    position_m = float((actual.to_translation() - expected.to_translation()).length)
    actual_rotation = actual.to_quaternion().normalized()
    expected_rotation = expected.to_quaternion().normalized()
    relative = expected_rotation.conjugated() @ actual_rotation
    relative.normalize()
    vector_length = math.sqrt(
        relative.x * relative.x
        + relative.y * relative.y
        + relative.z * relative.z
    )
    rotation_rad = 2.0 * math.atan2(vector_length, abs(relative.w))

    def deformation(matrix: Matrix, rotation: Quaternion) -> float:
        basis = np.asarray(matrix.to_3x3(), dtype=np.float64)
        rigid = np.asarray(rotation.to_matrix(), dtype=np.float64)
        return float(np.max(np.abs(basis - rigid)))

    affine = max(
        deformation(actual, actual_rotation),
        deformation(expected, expected_rotation),
    )
    return position_m, rotation_rad, affine


def force_fk(armature) -> None:
    """Disable Duck IK directly without baking the evaluated IK pose."""
    for pose_bone in armature.pose.bones:
        for constraint in pose_bone.constraints:
            if constraint.name.startswith("DUCK_IK"):
                constraint.influence = 0.0
    armature["fk_ik"] = 0.0


def _rest_matrix(body) -> Matrix:
    return Matrix.Translation(body.position) @ Quaternion(body.quaternion_wxyz).to_matrix().to_4x4()


def _twist_angle(quaternion: Quaternion, axis: tuple[float, float, float]) -> float:
    quaternion.normalize()
    direction = Vector(axis).normalized()
    signed_vector = quaternion.x * direction.x + quaternion.y * direction.y + quaternion.z * direction.z
    angle = 2.0 * math.atan2(signed_vector, quaternion.w)
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def joint_angles_from_body_matrices(
    matrices, profile, rest_matrices=None
) -> tuple[float, ...]:
    bodies = {body.name: body for body in profile.bodies}
    result = []
    for joint in profile.joints:
        relative = matrices[joint.parent_body].inverted_safe() @ matrices[joint.child_body]
        if rest_matrices is None:
            rest_relative = _rest_matrix(bodies[joint.child_body])
            axis = joint.axis
        else:
            rest_relative = (
                rest_matrices[joint.parent_body].inverted_safe()
                @ rest_matrices[joint.child_body]
            )
            # Generated rigs deliberately map every canonical hinge to the
            # pose bone's local Z rotation channel.
            axis = (0.0, 0.0, 1.0)
        dynamic = rest_relative.inverted_safe() @ relative
        result.append(_twist_angle(dynamic.to_quaternion(), axis))
    return tuple(result)


def body_samples(matrices, profile) -> tuple[np.ndarray, np.ndarray]:
    positions = []
    quaternions = []
    for name in profile.body_names:
        matrix = matrices[name]
        location = matrix.to_translation()
        rotation = matrix.to_quaternion().normalized()
        positions.append(tuple(location))
        quaternions.append((rotation.w, rotation.x, rotation.y, rotation.z))
    return np.asarray(positions, dtype=np.float64), np.asarray(quaternions, dtype=np.float64)


def canonical_body_matrices(root_world: Matrix, joint_angles, profile) -> dict[str, Matrix]:
    """Rebuild MJCF body frames from the Blender-calibrated root and joint state."""
    angles = dict(zip(profile.joint_names, joint_angles))
    joint_by_child = {joint.child_body: joint for joint in profile.joints}
    result = {}
    for body in profile.bodies:
        if body.parent is None:
            result[body.name] = root_world
            continue
        local = _rest_matrix(body)
        joint = joint_by_child.get(body.name)
        if joint is not None:
            local = local @ Matrix.Rotation(
                angles[joint.name], 4, Vector(joint.axis).normalized()
            )
        result[body.name] = result[body.parent] @ local
    return result
