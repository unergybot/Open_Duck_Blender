import unittest

import bpy

from open_duck_tools import addon


class AddonRegistrationTests(unittest.TestCase):
    def tearDown(self):
        addon.unregister()

    def test_registers_tools_panel_and_is_idempotent(self):
        addon.register()
        addon.register()
        self.assertTrue(hasattr(bpy.types, "DUCK_PT_tools"))
        self.assertTrue(hasattr(bpy.types.Object, "duck_colorway"))


if __name__ == "__main__":
    unittest.main()
