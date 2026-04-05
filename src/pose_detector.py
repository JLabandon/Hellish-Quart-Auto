import argparse
import math
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python as mp_python_tasks
from mediapipe.tasks.python import vision as mp_vision_tasks

from capture_camera import start_camera_capture


POSE_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14
LEFT_WRIST = 15
RIGHT_WRIST = 16
LEFT_PINKY = 17
RIGHT_PINKY = 18
LEFT_INDEX = 19
RIGHT_INDEX = 20
LEFT_THUMB = 21
RIGHT_THUMB = 22

ELBOW_EXTENDED_DEG = 138.0
FORWARD_Z_DELTA = 0.07
WRISTS_TOGETHER_RATIO = 0.45
RIGHT_UP_PALM_WRIST_DELTA = 0.4
RIGHT_HAND_FORWARD_Y_THRESHOLD = 0.7
CAMERA_MIRROR_LR = True


def _safe_visibility(lm: object) -> float:
    vis = getattr(lm, "visibility", 1.0)
    return float(vis) if vis is not None else 1.0


def _distance(a: object, b: object) -> float:
    dx = float(a.x) - float(b.x)
    dy = float(a.y) - float(b.y)
    return math.hypot(dx, dy)


def _angle_deg(a: object, b: object, c: object) -> float:
    bax = float(a.x) - float(b.x)
    bay = float(a.y) - float(b.y)
    bcx = float(c.x) - float(b.x)
    bcy = float(c.y) - float(b.y)

    dot = bax * bcx + bay * bcy
    norm1 = math.hypot(bax, bay)
    norm2 = math.hypot(bcx, bcy)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cos_val = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_val))


def _recognize_pose(landmarks: List[object]) -> Tuple[Optional[str], Dict[str, float | str | bool]]:
    debug: Dict[str, float | str | bool] = {}
    needed = [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
    if any(idx >= len(landmarks) for idx in needed):
        debug["reason"] = "missing_landmarks"
        return None, debug

    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    le = landmarks[LEFT_ELBOW]
    re = landmarks[RIGHT_ELBOW]
    lw = landmarks[LEFT_WRIST]
    rw = landmarks[RIGHT_WRIST]

    shoulder_width = max(_distance(ls, rs), 1e-6)
    wrist_distance = _distance(lw, rw)
    wrists_together = wrist_distance <= shoulder_width * WRISTS_TOGETHER_RATIO

    shoulder_mid_x = (float(ls.x) + float(rs.x)) * 0.5
    shoulder_mid_y = (float(ls.y) + float(rs.y)) * 0.5

    wrists_mid_x = (float(lw.x) + float(rw.x)) * 0.5
    wrists_mid_y = (float(lw.y) + float(rw.y)) * 0.5

    x_offset = wrists_mid_x - shoulder_mid_x
    y_offset = wrists_mid_y - shoulder_mid_y

    left_elbow_angle = _angle_deg(ls, le, lw)
    right_elbow_angle = _angle_deg(rs, re, rw)
    left_extended = left_elbow_angle >= ELBOW_EXTENDED_DEG
    right_extended = right_elbow_angle >= ELBOW_EXTENDED_DEG

    left_z_delta = float(ls.z) - float(lw.z)
    right_z_delta = float(rs.z) - float(rw.z)
    left_forward = left_extended and (left_z_delta >= FORWARD_Z_DELTA)
    right_forward = right_extended and (right_z_delta >= FORWARD_Z_DELTA)

    debug.update(
        {
            "shoulder_width": shoulder_width,
            "wrist_distance": wrist_distance,
            "wrists_together": wrists_together,
            "x_offset": x_offset,
            "y_offset": y_offset,
            "left_elbow_angle": left_elbow_angle,
            "right_elbow_angle": right_elbow_angle,
            "left_z_delta": left_z_delta,
            "right_z_delta": right_z_delta,
            "left_forward": left_forward,
            "right_forward": right_forward,
        }
    )

    if wrists_together:
        if min(_safe_visibility(ls), _safe_visibility(rs), _safe_visibility(lw), _safe_visibility(rw)) < 0.30:
            debug["reason"] = "hands_together_low_visibility"
            return None, debug
        x_thr = shoulder_width * 0.40
        y_thr = shoulder_width * 0.40
        if abs(x_offset) >= abs(y_offset):
            if x_offset <= -x_thr:
                return "Hands Together Left", debug
            if x_offset >= x_thr:
                return "Hands Together Right", debug
        if y_offset <= -y_thr:
            return "Hands Together Up", debug
        if y_offset >= y_thr:
            return "Hands Together Down", debug
        return "Hands Together Center", debug

    if not left_forward and not right_forward:
        debug["reason"] = "no_forward"
        return None, debug

    if left_forward != right_forward:

        forward_wrist = lw if left_forward else rw

        if float(forward_wrist.x) <= shoulder_mid_x:
            hand_label = "Right"
        else:
            hand_label = "Left"

        debug["hand_label"] = hand_label

        if hand_label == "Left":
            return "Left Hand Forward", debug

        return "Right Hand Forward", debug

    debug["reason"] = "no_pose_match"
    return None, debug


def _ensure_pose_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    )
    urllib.request.urlretrieve(model_url, str(model_path))
    return model_path


def _draw_pose(frame_bgr: np.ndarray, landmarks: List[object]) -> None:
    h, w = frame_bgr.shape[:2]

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue
        a = landmarks[start_idx]
        b = landmarks[end_idx]
        if a.visibility < 0.2 or b.visibility < 0.2:
            continue
        pt1 = (int(a.x * w), int(a.y * h))
        pt2 = (int(b.x * w), int(b.y * h))
        cv2.line(frame_bgr, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)

    for lm in landmarks:
        if lm.visibility < 0.2:
            continue
        x = int(lm.x * w)
        y = int(lm.y * h)
        cv2.circle(frame_bgr, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MediaPipe Pose on front camera queue and visualize stream.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target camera capture FPS")
    parser.add_argument("--buffer-size", type=int, default=5, help="Front camera queue max frame count")
    parser.add_argument("--camera-index", type=int, default=0, help="Front camera index")
    parser.add_argument("--camera-width", type=int, default=1280, help="Camera capture width")
    parser.add_argument("--camera-height", type=int, default=720, help="Camera capture height")
    parser.add_argument("--num-poses", type=int, default=1, help="Max number of poses to detect")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5, help="MediaPipe min pose detection confidence")
    parser.add_argument("--min-presence-confidence", type=float, default=0.5, help="MediaPipe min pose presence confidence")
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5, help="MediaPipe min tracking confidence")
    parser.add_argument(
        "--pose-model",
        type=str,
        default="",
        help="Optional .task model path; default auto-downloads pose_landmarker_lite.task to ./model",
    )
    return parser.parse_args()




def run_pose_detection(
    frame_buffer,
    stop_event,
    capture_thread,
    model_path: Path,
    controller=None,
    enable_visualization: bool = True,
    num_poses: int = 1,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    output_path: Optional[Path] = None,
    output_fps: float = 30.0,
    frame_callback: Optional[Callable[[np.ndarray], None]] = None,
) -> None:
    """Run pose detection on camera frames and optionally send gestures to controller."""
    model_path = _ensure_pose_model(model_path)

    base_options = mp_python_tasks.BaseOptions(model_asset_path=str(model_path))
    pose_options = mp_vision_tasks.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision_tasks.RunningMode.VIDEO,
        num_poses=num_poses,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    pose = mp_vision_tasks.PoseLandmarker.create_from_options(pose_options)

    window_name = "Front Camera Pose"
    local_visualization = enable_visualization and frame_callback is None
    if local_visualization:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_t = time.perf_counter()
    fps_smoothed = 0.0
    last_stable_pose_label: Optional[str] = None
    pending_pose_label: Optional[str] = None
    pending_pose_count = 0
    stable_required_frames = 4
    writer: Optional[cv2.VideoWriter] = None

    try:
        while not stop_event.is_set():
            frame_rgb = frame_buffer.latest()
            if frame_rgb is None:
                time.sleep(0.005)
                continue

            if CAMERA_MIRROR_LR:
                frame_rgb = cv2.flip(frame_rgb, 1)

            ts_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = pose.detect_for_video(mp_image, ts_ms)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if output_path is not None and writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                h, w = frame_bgr.shape[:2]
                safe_fps = output_fps if output_fps > 0 else 30.0
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    safe_fps,
                    (w, h),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open pose output writer: {output_path}")

            if results.pose_landmarks:
                for pose_landmarks in results.pose_landmarks:
                    _draw_pose(frame_bgr, pose_landmarks)

            pose_label: Optional[str] = None
            pose_debug: Dict[str, float | str | bool] = {}
            if results.pose_landmarks:
                pose_label, pose_debug = _recognize_pose(results.pose_landmarks[0])

            if pose_label is None:
                pending_pose_label = None
                pending_pose_count = 0
            else:
                if pose_label == pending_pose_label:
                    pending_pose_count += 1
                else:
                    pending_pose_label = pose_label
                    pending_pose_count = 1

                if pending_pose_count >= stable_required_frames and pose_label != last_stable_pose_label:
                    # print(f"Pose -> {pose_label}", flush=True)
                    last_stable_pose_label = pose_label
                    if controller is not None:
                        controller.enqueue_gesture(pose_label)

            now = time.perf_counter()
            dt = max(1e-6, now - last_t)
            last_t = now
            inst_fps = 1.0 / dt
            fps_smoothed = inst_fps if fps_smoothed <= 0 else (fps_smoothed * 0.9 + inst_fps * 0.1)

            cv2.putText(
                frame_bgr,
                f"Pose FPS: {fps_smoothed:.1f} | Queue: {frame_buffer.size()} | Poses: {len(results.pose_landmarks)}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if last_stable_pose_label is not None:
                cv2.putText(
                    frame_bgr,
                    f"Gesture: {last_stable_pose_label}",
                    (12, 64),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if local_visualization:
                cv2.imshow(window_name, frame_bgr)
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord("q"):
                    stop_event.set()
                    break

            if frame_callback is not None:
                frame_callback(frame_bgr)

            if writer is not None:
                writer.write(frame_bgr)
    finally:
        stop_event.set()
        capture_thread.join(timeout=1.0)
        pose.close()
        if writer is not None:
            writer.release()
        if local_visualization:
            cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0")

    frame_buffer, stop_event, capture_thread = start_camera_capture(
        fps=args.fps,
        buffer_size=args.buffer_size,
        camera_index=args.camera_index,
        width=args.camera_width,
        height=args.camera_height,
    )

    project_root = Path(__file__).resolve().parents[1]
    default_model_path = project_root / "model" / "pose_landmarker_lite.task"
    model_path = Path(args.pose_model).resolve() if args.pose_model else default_model_path

    run_pose_detection(
        frame_buffer=frame_buffer,
        stop_event=stop_event,
        capture_thread=capture_thread,
        model_path=model_path,
        controller=None,
        enable_visualization=True,
        num_poses=args.num_poses,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        output_path=project_root / "output" / f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
        output_fps=args.fps,
    )


if __name__ == "__main__":
    main()
