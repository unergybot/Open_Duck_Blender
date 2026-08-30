import unittest

import bpy
from mathutils import Matrix, Vector

from open_duck_tools import addon


class AddonRegistrationTests(unittest.TestCase):
    def tearDown(self):
        addon.unregister()

    def test_registers_tools_panel_and_is_idempotent(self):
        addon.register()
        addon.register()
        self.assertTrue(hasattr(bpy.types, "DUCK_PT_tools"))
        self.assertTrue(hasattr(bpy.types.Object, "duck_colorway"))

    def test_ik_target_uses_ankle_tail_without_changing_orientation(self):
        armature_world = Matrix.Translation((1.0, 2.0, 3.0))
        ankle_matrix = Matrix.Rotation(0.3, 4, "Z")
        ankle_matrix.translation = (0.1, 0.2, 0.3)
        ankle_tail = Vector((0.1, 0.215, 0.3))
        target = addon._ankle_target_matrix(armature_world, ankle_matrix, ankle_tail)
        for actual, expected in zip(target.translation, (1.1, 2.215, 3.3)):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertLess(
            target.to_quaternion().rotation_difference(
                (armature_world @ ankle_matrix).to_quaternion()
            ).angle,
            1e-7,
        )


if __name__ == "__main__":
    unittest.main()
