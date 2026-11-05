"""Check assignment-specific RGB, letterbox, all colors, dimensions and PnP integration."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from detect_assignment import (BASE, COLORS, PATTERNS, Detector, load_calibration,
                               original_corners, physical_corners, preprocess)


class AssignmentTests(unittest.TestCase):
    def test_rgb_letterbox_and_original_pixel_inverse(self):
        image = np.empty((1080, 1440, 3), np.uint8)
        image[:] = [10, 20, 30]  # BGR
        tensor, transform = preprocess(image)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertEqual(tensor.dtype, np.float16)
        np.testing.assert_allclose(tensor[0, :, 320, 320], np.array([30,20,10])/255,
                                   atol=1e-4)
        np.testing.assert_allclose(tensor[0, :, 0, 0], 114/255, atol=2e-4)
        self.assertEqual(transform[2:], (0, 80))
        points = np.array([[0, 80], [640, 560]], dtype=float)
        np.testing.assert_allclose(original_corners(points, transform), [[0, 0], [1440, 1080]])

    def test_dimensions_follow_pattern(self):
        for pattern in PATTERNS:
            points = np.array(physical_corners(pattern))
            self.assertAlmostEqual(np.ptp(points[:,0]), .230 if pattern in ("1", "base_big") else .135)
            self.assertAlmostEqual(np.ptp(points[:,1]), .056)

    def test_yaml_calibration_and_all_colors_reach_camera_space(self):
        config = load_calibration(BASE / "assets/calibration.yaml", (1440, 1080))
        output = np.zeros((1, 25200, 22), dtype=np.float32)
        output[0,:,8] = -100
        truths = []
        for i, color in enumerate(COLORS):
            pattern = ["1", "3", "base_big", "sentry"][i]
            points = np.array(physical_corners(pattern), dtype=float)
            rvec, tvec = np.array([.15,-.2,.05]), np.array([-.6+i*.4,0,3.])
            px, _ = cv2.projectPoints(points, rvec, tvec,
                                       config["camera_matrix"],config["dist_coeffs"])
            model_px = px.reshape(4,2) * (640/1440) + [0,80]
            row = output[0,i]
            row[:8], row[8], row[9+i], row[13+PATTERNS.index(pattern)] = model_px.reshape(8), 4, 5, 5
            rotation, _ = cv2.Rodrigues(rvec)
            truths.append((rotation @ points.T).T + tvec)

        class FakeSession:
            def run(self, names, feed):
                return [output]

        detector = Detector.__new__(Detector)
        detector.session, detector.input_name = FakeSession(), "images"
        detector.confidence, detector.iou = .5, .5
        detections = detector.detect(np.zeros((1080,1440,3),np.uint8), config)
        self.assertEqual({d["color"] for d in detections}, set(COLORS))
        for detection in detections:
            i = COLORS.index(detection["color"])
            np.testing.assert_allclose(detection["corners_camera_m"], truths[i], atol=2e-5)
            self.assertLess(detection["reprojection_error_px"], .001)


if __name__ == "__main__":
    unittest.main()
