"""Synthetic geometry checks; these do not assert real-camera accuracy."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from armor_pnp import load_pnp_config, pose_for_detection, solve_corners_camera


class PoseTests(unittest.TestCase):
    def setUp(self):
        # Synthetic dimensions and calibration, not measurements for the user's camera.
        self.points = np.array([[-.07, -.03, 0], [-.07, .03, 0],
                                [.07, .03, 0], [.07, -.03, 0]], dtype=np.float64)
        self.matrix = np.array([[1400., 0, 960], [0, 1380, 540], [0, 0, 1]])
        self.dist = np.array([.08, -.03, .001, -.002, .005])
        self.rvec = np.array([.12, -.35, .08])
        self.translation = np.array([.2, -.1, 2.4])
        image, _ = cv2.projectPoints(self.points, self.rvec, self.translation,
                                     self.matrix, self.dist)
        self.image = image.reshape(4, 2)

    def test_recovers_known_pose_and_camera_corners_with_distortion(self):
        result = solve_corners_camera(self.image, self.points, self.matrix, self.dist)
        rotation, _ = cv2.Rodrigues(self.rvec)
        expected = (rotation @ self.points.T).T + self.translation
        np.testing.assert_allclose(result["corners_camera_m"], expected, atol=1e-7)
        np.testing.assert_allclose(result["t_object_to_camera_m"], self.translation, atol=1e-7)
        self.assertLess(result["reprojection_error_px"], 1e-7)

    def test_pixel_error_is_euclidean_rmse(self):
        noisy = self.image + np.array([[.4, -.2], [-.3, .1], [.2, .5], [-.2, -.4]])
        result = solve_corners_camera(noisy, self.points, self.matrix, self.dist)
        delta = np.asarray(result["reprojected_corners_px"]) - noisy
        self.assertAlmostEqual(result["reprojection_error_px"],
                               float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1)))))
        self.assertTrue(np.all(np.asarray(result["corners_camera_m"])[:, 2] > 0))

    def test_calibration_resolution_mismatch_is_rejected(self):
        data = {"camera_model": "opencv_pinhole", "image_size_wh": [1920, 1080],
                "camera_matrix": self.matrix.tolist(), "dist_coeffs": self.dist.tolist(),
                "corners_object_m": self.points.tolist()}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "camera.json"
            path.write_text(json.dumps(data))
            load_pnp_config(path, (1920, 1080))
            with self.assertRaisesRegex(ValueError, "differs"):
                load_pnp_config(path, (640, 640))

    def test_degenerate_and_crossed_corners_fail(self):
        for bad in (np.zeros((4, 2)), self.image[[0, 2, 1, 3]]):
            with self.assertRaises(ValueError):
                solve_corners_camera(bad, self.points, self.matrix, self.dist)

    def test_per_label_geometry_is_selected_and_missing_geometry_fails(self):
        config = {"camera_matrix": self.matrix, "dist_coeffs": self.dist,
                  "corners_object_m": None, "corners_object_m_by_label": {"1": self.points}}
        result = pose_for_detection(self.image, "1", config)
        self.assertEqual(result["pnp_status"], "ok")
        missing = pose_for_detection(self.image, "3", config)
        self.assertEqual(missing["pnp_status"], "failed")
        self.assertIsNone(missing["corners_camera_m"])


if __name__ == "__main__":
    unittest.main()
