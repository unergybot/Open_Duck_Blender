"""Translate Blender's evaluated body matrices into robot state samples."""

from __future__ import annotations

import math

from mathutils import Matrix, Quaternion, Vector
import numpy as np


def _rest_matrix(body) -> Matrix:
    return Matrix.Translation(body.position) @ Quaternion(body.quaternion_wxyz).to_matrix().to_4x4()


def _twist_angle(quaternion: Quaternion, axis: tuple[float, float, float]) -> float:
    quaternion.normalize()
    direction = Vector(axis).normalized()
    signed_vector = quaternion.x * direction.x + quaternion.y * direction.y + quaternion.z * direction.z
    angle = 2.0 * math.atan2(signed_vector, quaternion.w)
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def joint_angles_from_body_matrices(matrices, profile) -> tuple[float, ...]:
    bodies = {body.name: body for body in profile.bodies}
    result = []
    for joint in profile.joints:
        relative = matrices[joint.parent_body].inverted_safe() @ matrices[joint.child_body]
        dynamic = _rest_matrix(bodies[joint.child_body]).inverted_safe() @ relative
        result.append(_twist_angle(dynamic.to_quaternion(), joint.axis))
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
