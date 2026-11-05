"""HKUAstar detection assignment: RGB letterbox, ONNX CPU, YAML calibration, PnP.

Requirements and assets: HKUAstar/vision-training-assignment, commit
c9fa5e9b66b61f5edcc23341ac7dac83a840bcc3, detection/README.md.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from armor_pnp import pose_for_detection, validate_pnp_config

BASE = Path(__file__).resolve().parent
COLORS = ["blue", "red", "neutral", "purple"]
PATTERNS = ["sentry", "1", "2", "3", "4", "5", "outpost", "base_small", "base_big"]


def physical_corners(pattern):
    """Assignment dimensions by decoded pattern, independent of input filename."""
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}")
    w = 0.230 if pattern in ("1", "base_big") else 0.135
    h = 0.056
    return [[-w/2, -h/2, 0], [-w/2, h/2, 0], [w/2, h/2, 0], [w/2, -h/2, 0]]


def load_calibration(path, image_size_wh):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Calibration YAML must contain a mapping")
    return validate_pnp_config({
        "camera_model": "opencv_pinhole",
        "image_size_wh": [raw.get("image_width"), raw.get("image_height")],
        "camera_matrix": raw.get("camera_matrix"),
        "dist_coeffs": raw.get("distort_coeffs"),
        "corners_object_m_by_label": {p: physical_corners(p) for p in PATTERNS},
    }, image_size_wh)


def preprocess(image):
    """Return float16 RGB NCHW and exact inverse scale/padding for letterbox."""
    h, w = image.shape[:2]
    ratio = min(640 / w, 640 / h)
    rw, rh = max(1, round(w * ratio)), max(1, round(h * ratio))
    left, top = (640 - rw) // 2, (640 - rh) // 2
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top+rh, left:left+rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray((rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None],
                                  dtype=np.float16)
    return tensor, (rw / w, rh / h, left, top)


def original_corners(corners, transform):
    sx, sy, left, top = transform
    return (corners.astype(np.float64) - [left, top]) / [sx, sy]


class Detector:
    def __init__(self, model, confidence=0.5, iou=0.5):
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        spec = self.session.get_inputs()[0]
        if spec.shape != [1, 3, 640, 640] or spec.type != "tensor(float16)":
            raise ValueError(f"Assignment expects [1,3,640,640] float16, got {spec.shape} {spec.type}")
        self.input_name = spec.name
        self.confidence, self.iou = confidence, iou

    def detect(self, image, calibration):
        tensor, transform = preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.shape != (1, 25200, 22):
            raise ValueError(f"Unexpected model output: {output.shape}")
        rows = output[0]
        confidences = np.exp(-np.logaddexp(0.0, -rows[:, 8]))
        candidates, boxes, scores = [], [], []
        for index in np.flatnonzero(confidences >= self.confidence):
            row = rows[index]
            if not np.isfinite(row).all():
                continue
            # Keep intrinsic corner order; never re-sort by x/y.
            corners = original_corners(row[:8].reshape(4, 2), transform)
            lo, hi = corners.min(axis=0), corners.max(axis=0)
            if np.any(hi <= lo):
                continue
            color = COLORS[int(np.argmax(row[9:13]))]
            pattern = PATTERNS[int(np.argmax(row[13:22]))]
            confidence = float(confidences[index])
            candidates.append((corners, color, pattern, confidence))
            boxes.append([float(lo[0]), float(lo[1]), float(hi[0]-lo[0]), float(hi[1]-lo[1])])
            scores.append(confidence)
        keep = cv2.dnn.NMSBoxes(boxes, scores, self.confidence, self.iou) if boxes else []
        detections = []
        for index in np.asarray(keep).reshape(-1):
            corners, color, pattern, confidence = candidates[int(index)]
            pose = pose_for_detection(corners, pattern, calibration)
            detections.append({
                "color": color, "pattern": pattern, "confidence": confidence,
                "corners_px": corners.tolist(),
                "corners_camera_m": pose["corners_camera_m"],
                "reprojection_error_px": pose["reprojection_error_px"],
            })
        return detections


def draw_detections(image, detections):
    rendered = image.copy()
    palette = {"blue": (255, 0, 0), "red": (0, 0, 255),
               "neutral": (180, 180, 180), "purple": (255, 0, 255)}
    for detection in detections:
        points = np.rint(detection["corners_px"]).astype(np.int32)
        color = palette[detection["color"]]
        cv2.polylines(rendered, [points], True, color, 2)
        for point in points:
            cv2.circle(rendered, tuple(map(int, point)), 3, (0, 255, 0), -1)
        x, y = points.min(axis=0)
        text = f"{detection['color']} {detection['pattern']} {detection['confidence']:.2f}"
        cv2.putText(rendered, text, (max(0, int(x)), max(20, int(y)-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
    return rendered


def run(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detector = Detector(args.model, args.conf, args.iou)
    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise ValueError(f"Cannot read {args.image}")
        calibration = load_calibration(args.calibration, (image.shape[1], image.shape[0]))
        detections = detector.detect(image, calibration)
        record = {"frame_index": 0, "timestamp_s": 0.0, "detections": detections}
        args.output.write_text(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n",
                               encoding="utf-8")
        annotated = args.output.with_name(args.output.stem + "_detected.jpg")
        if not cv2.imwrite(str(annotated), draw_detections(image, detections)):
            raise OSError(f"Cannot write {annotated}")
        print(f"Saved {len(detections)} detections: {args.output}\nPreview: {annotated}")
        return

    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or abs(fps - 30) > .01:
            raise ValueError(f"Assignment requires 30 FPS, got {fps}")
        calibration, size, index = None, None, 0
        with args.output.open("w", encoding="utf-8") as stream:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_size = (frame.shape[1], frame.shape[0])
                if calibration is None:
                    calibration = load_calibration(args.calibration, frame_size)
                    size = frame_size
                elif frame_size != size:
                    raise ValueError("Frame dimensions changed; calibration no longer matches")
                detections = detector.detect(frame, calibration)
                record = {"frame_index": index, "timestamp_s": index / 30,
                          "detections": detections}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                if index == 0 and args.preview:
                    args.preview.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(args.preview), draw_detections(frame, detections)):
                        raise OSError(f"Cannot write {args.preview}")
                index += 1
                if index % 60 == 0:
                    print(f"Processed {index} frames", flush=True)
        if index == 0:
            raise ValueError("Video contains no decodable frames")
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if expected > 0 and index != expected:
            raise ValueError(f"Decoded {index}/{expected} frames; output is incomplete")
        print(f"Saved {index} frames: {args.output}")
    finally:
        cap.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--model", type=Path, default=BASE / "assets/0526.onnx")
    parser.add_argument("--calibration", type=Path, default=BASE / "assets/calibration.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, help="Optional annotated first frame for video mode")
    parser.add_argument("--conf", type=float, default=.5)
    parser.add_argument("--iou", type=float, default=.5)
    args = parser.parse_args()
    if not (0 < args.conf < 1 and 0 < args.iou < 1):
        parser.error("--conf and --iou must be between 0 and 1")
    run(args)
