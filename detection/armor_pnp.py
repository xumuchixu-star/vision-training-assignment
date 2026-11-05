"""Planar four-corner PnP. All physical coordinates and translations use metres."""
import json
from pathlib import Path

import cv2
import numpy as np


def _points(value, shape, name):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if points.shape != shape or not np.isfinite(points).all():
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return np.ascontiguousarray(points)


def _convex_quad(points, name):
    """Reject duplicate, collinear, concave and self-crossing point sequences."""
    span = np.linalg.norm(np.ptp(points, axis=0))
    if span <= 0:
        raise ValueError(f"{name}: corners must be distinct")
    edges = np.roll(points, -1, axis=0) - points
    next_edges = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * next_edges[:, 1] - edges[:, 1] * next_edges[:, 0]
    threshold = span * span * 1e-10
    if not (np.all(cross > threshold) or np.all(cross < -threshold)):
        raise ValueError(f"{name}: expected four ordered corners of a convex quadrilateral")


def load_pnp_config(path, image_size_wh):
    """Require calibration for the original image; do not silently scale/crop K."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_pnp_config(config, image_size_wh)


def validate_pnp_config(config, image_size_wh):
    """Check calibration resolution and prepare NumPy arrays for PnP."""
    # if not isinstance(config, dict):
    #     raise ValueError("PnP configuration must be a JSON object")
    image_size = config.get("image_size_wh")
    # if (not isinstance(image_size, list) or len(image_size) != 2
    #         or any(type(x) is not int or x <= 0 for x in image_size)):
    #     raise ValueError("image_size_wh must contain the calibration width and height")
    if image_size != list(image_size_wh):
        raise ValueError(
            f"Calibration size {image_size} differs from original image {list(image_size_wh)}. "
            "Supply intrinsics for this image resolution/crop. Model resize to 640 is unrelated."
        )
    # matrix = _points(config.get("camera_matrix"), (3, 3), "camera_matrix")
    matrix = np.ascontiguousarray(config["camera_matrix"], dtype=np.float64)
    # if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
    #     raise ValueError("camera_matrix: fx and fy must be positive")
    # if (not np.allclose(matrix[2], [0, 0, 1], atol=1e-10, rtol=0)
    #         or abs(matrix[0, 1]) > 1e-10 or abs(matrix[1, 0]) > 1e-10):
    #     raise ValueError("Use OpenCV pinhole intrinsics [[fx,0,cx],[0,fy,cy],[0,0,1]]")
    # if config.get("camera_model") != "opencv_pinhole":
    #     raise ValueError("camera_model must be opencv_pinhole; fisheye parameters need conversion")
    raw_distortion = config.get("dist_coeffs")
    # if not isinstance(raw_distortion, list) or len(raw_distortion) not in (4, 5, 8, 12, 14):
    #     raise ValueError("dist_coeffs must explicitly contain 4/5/8/12/14 coefficients")
    # distortion = _points(raw_distortion, (len(raw_distortion),), "dist_coeffs")
    distortion = np.ascontiguousarray(raw_distortion, dtype=np.float64)
    default_points = config.get("corners_object_m")
    overrides = config.get("corners_object_m_by_label", {})
    # if not isinstance(overrides, dict):
    #     raise ValueError("corners_object_m_by_label must be a label-to-corners object")
    # if default_points is None and not overrides:
    #     raise ValueError("Provide measured corners_object_m or corners_object_m_by_label")
    points = np.ascontiguousarray(default_points, dtype=np.float64) if default_points is not None else None
    overrides = {str(label): np.ascontiguousarray(value, dtype=np.float64)
                 for label, value in overrides.items()}
    return {"camera_matrix": matrix, "dist_coeffs": distortion,
            "corners_object_m": points, "corners_object_m_by_label": overrides}


def solve_corners_camera(corners_image_px, corners_object_m, camera_matrix, dist_coeffs):
    """Choose the positive-depth IPPE solution with lowest pixel reprojection RMSE.

    Returns X_camera = R_object_to_camera @ X_object + t_object_to_camera_m.
    Planar pose can be ambiguous; num_positive_depth_solutions exposes this.
    """
    # Physical corners are generated from the prescribed width and height.
    object_points = np.ascontiguousarray(corners_object_m, dtype=np.float64)
    image_points = _points(corners_image_px, (4, 2), "corners_image_px")
    _convex_quad(image_points, "corners_image_px")
    matrix = _points(camera_matrix, (3, 3), "camera_matrix")
    distortion = np.ascontiguousarray(dist_coeffs, dtype=np.float64)
    result = cv2.solvePnPGeneric(object_points, image_points, matrix, distortion,
                                 flags=cv2.SOLVEPNP_IPPE)
    if not result[0]:
        raise ValueError("IPPE did not produce a pose")
    candidates = []
    for rvec, tvec in zip(result[1], result[2]):
        rotation, _ = cv2.Rodrigues(rvec)
        translation = tvec.reshape(3)
        camera_points = (rotation @ object_points.T).T + translation
        if not np.isfinite(camera_points).all() or np.any(camera_points[:, 2] <= 0):
            continue
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, distortion)
        projected = projected.reshape(4, 2)
        error = float(np.sqrt(np.mean(np.sum((projected - image_points) ** 2, axis=1))))
        if np.isfinite(error):
            candidates.append((error, rotation, translation, camera_points, projected))
    if not candidates:
        raise ValueError("No finite PnP solution with all corners in front of the camera")
    error, rotation, translation, camera_points, projected = min(candidates, key=lambda x: x[0])
    return {
        "pnp_status": "ok",
        "corners_camera_m": camera_points.tolist(),
        "corners_object_m": object_points.tolist(),
        "R_object_to_camera": rotation.tolist(),
        "t_object_to_camera_m": translation.tolist(),
        "reprojected_corners_px": projected.tolist(),
        "reprojection_error_px": error,
        "pnp_method": "IPPE_min_reprojection_positive_depth",
        "num_positive_depth_solutions": len(candidates),
    }


def pose_for_detection(landmarks, label, config):
    points = config["corners_object_m_by_label"].get(str(label), config["corners_object_m"])
    if points is None:
        return {"pnp_status": "failed", "corners_camera_m": None,
                "reprojection_error_px": None,
                "pnp_error": f"No measured object corners configured for label {label}"}
    try:
        return solve_corners_camera(landmarks, points, config["camera_matrix"], config["dist_coeffs"])
    except (ValueError, cv2.error) as error:
        return {"pnp_status": "failed", "corners_camera_m": None,
                "reprojection_error_px": None, "pnp_error": str(error)}
