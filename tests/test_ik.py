import math
from pathlib import Path
import unittest

import numpy as np

from open_duck_tools.ik import leg_kinematics, solve_leg_ik
from open_duck_tools.profile import build_microduck_profile


ROBOT_ROOT = Path.home() / "MyCode/microduck_rl/src/mjlab_microduck/robot/microduck"
MJCF = ROBOT_ROOT / "robot_walk.xml"
RUNTIME = Path.home() / "MyCode/microduck/duck-control/src/model.rs"
CONTRACT = Path.home() / "MyCode/microduck/duck-ipc-proto/src/lib.rs"


class PhysicalLegIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = build_microduck_profile(
            MJCF, RUNTIME, None, CONTRACT
        )

    def test_analytic_position_jacobian_matches_central_difference(self):
        for side in ("left", "right"):
            model = leg_kinematics(self.profile, side)
            angles = np.asarray([0.08, -0.31, 0.64, -0.22])
            analytic = model.position_jacobian(angles)
            numeric = np.empty((3, 4))
            epsilon = 1e-7
            for column in range(4):
                before = angles.copy()
                after = angles.copy()
                before[column] -= epsilon
                after[column] += epsilon
                numeric[:, column] = (
                    model.forward(after).position - model.forward(before).position
                ) / (2.0 * epsilon)
            np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=0.0)

    def test_reachable_target_solves_position_and_sagittal_pitch(self):
        for side in ("left", "right"):
            model = leg_kinematics(self.profile, side)
            expected = np.asarray([0.1, -0.35, 0.72, -0.18])
            target = model.forward(expected)
            result = solve_leg_ik(
                model,
                target.position,
                target.pitch,
                initial_angles=np.asarray(model.home_angles),
                pole_sign=1.0,
            )
            actual = model.forward(result.angles)
            self.assertTrue(result.reached)
            self.assertFalse(result.clamped)
            self.assertLess(np.linalg.norm(actual.position - target.position), 1e-5)
            self.assertLess(
                abs(math.remainder(actual.pitch - target.pitch, 2.0 * math.pi)),
                1e-5,
            )

    def test_unreachable_target_retains_best_finite_bounded_solution(self):
        for side in ("left", "right"):
            model = leg_kinematics(self.profile, side)
            result = solve_leg_ik(
                model,
                np.asarray((2.0, -2.0, 2.0)),
                0.0,
                initial_angles=np.asarray(model.home_angles),
                pole_sign=-1.0,
            )
            self.assertFalse(result.reached)
            self.assertTrue(result.clamped)
            self.assertTrue(np.isfinite(result.angles).all())
            self.assertTrue(math.isfinite(result.objective))
            for angle, limits in zip(result.angles, model.limits, strict=True):
                self.assertGreaterEqual(angle, limits[0])
                self.assertLessEqual(angle, limits[1])

    def test_nonfinite_target_is_rejected(self):
        model = leg_kinematics(self.profile, "left")
        with self.assertRaisesRegex(ValueError, "finite"):
            solve_leg_ik(model, (math.nan, 0.0, 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
