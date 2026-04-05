from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import cv2

from capture_screen import start_screen_capture
from controller import Xbox360PIDController
from move_detector import YoloPyTorchDetector


OUTPUT_W, OUTPUT_H = 1440, 900


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame = None

    def update(self, frame) -> None:
        with self._lock:
            self._frame = frame.copy()

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


def _create_writer(output_path: Path, fps: float) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if fps and fps > 0 else 30.0
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        safe_fps,
        (OUTPUT_W, OUTPUT_H),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {output_path}")
    return writer


def _compute_real_fps(frame_count: int, first_ts: float | None, last_ts: float | None, fallback_fps: float) -> float:
    if frame_count <= 1 or first_ts is None or last_ts is None:
        return fallback_fps if fallback_fps > 0 else 30.0
    duration = max(1e-6, last_ts - first_ts)
    fps = (frame_count - 1) / duration
    return fps if fps > 0 else (fallback_fps if fallback_fps > 0 else 30.0)


def _reencode_with_fps(temp_path: Path, final_path: Path, fps: float) -> None:
    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open temp video for re-encode: {temp_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"Invalid temp video size for re-encode: {temp_path}")

    final_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(final_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open final video writer: {final_path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            writer.write(frame)
    finally:
        cap.release()
        writer.release()


def _stack_videos_vertical(move_path: Path, pose_path: Path, out_path: Path, fps: float) -> None:
    move_cap = cv2.VideoCapture(str(move_path))
    pose_cap = cv2.VideoCapture(str(pose_path))
    if not move_cap.isOpened() or not pose_cap.isOpened():
        if move_cap.isOpened():
            move_cap.release()
        if pose_cap.isOpened():
            pose_cap.release()
        raise RuntimeError("Failed to open move/pose video for stacking")

    move_w = int(move_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    move_h = int(move_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pose_w = int(pose_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    pose_h = int(pose_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if min(move_w, move_h, pose_w, pose_h) <= 0:
        move_cap.release()
        pose_cap.release()
        raise RuntimeError("Invalid video size while stacking move/pose outputs")

    out_w = max(move_w, pose_w)
    move_scaled_h = int(round(move_h * (out_w / move_w)))
    pose_scaled_h = int(round(pose_h * (out_w / pose_w)))
    out_h = move_scaled_h + pose_scaled_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 30.0,
        (out_w, out_h),
    )
    if not writer.isOpened():
        move_cap.release()
        pose_cap.release()
        raise RuntimeError(f"Failed to open stacked video writer: {out_path}")

    try:
        while True:
            ok_m, frame_m = move_cap.read()
            ok_p, frame_p = pose_cap.read()
            if not ok_m or frame_m is None or not ok_p or frame_p is None:
                break

            top = cv2.resize(frame_m, (out_w, move_scaled_h), interpolation=cv2.INTER_LINEAR)
            bottom = cv2.resize(frame_p, (out_w, pose_scaled_h), interpolation=cv2.INTER_LINEAR)
            stacked = cv2.vconcat([top, bottom])
            writer.write(stacked)
    finally:
        move_cap.release()
        pose_cap.release()
        writer.release()


def run_capture_mode(
    args,
    window_name: str,
    run_time: str,
    pose_preview_buffer: LatestFrameBuffer | None,
    pose_window_name: str,
    gesture_stop_event,
    shutdown_event: threading.Event,
    auto_enabled: Optional[dict],
    auto_ls_controller: Optional[Xbox360PIDController],
) -> None:
    detector = YoloPyTorchDetector(model_dir="model", model_name="yolov8n")

    frame_buffer, stop_event, capture_thread = start_screen_capture(
        fps=args.fps,
        buffer_size=args.buffer_size,
    )

    output_path = Path("output") / f"move_{run_time}.mp4"
    pose_output_path = Path("output") / f"pose_{run_time}.mp4"
    move_temp_path = Path("output") / f"move_{run_time}.tmp.mp4"
    writer = _create_writer(move_temp_path, args.fps)
    pose_writer = None
    pose_temp_path = None
    if pose_preview_buffer is not None:
        pose_temp_path = Path("output") / f"pose_{run_time}.tmp.mp4"
        pose_writer = _create_writer(pose_temp_path, args.fps)
    frame_count = 0
    first_ts: float | None = None
    last_ts: float | None = None

    try:
        while not stop_event.is_set() and not shutdown_event.is_set():
            if pose_preview_buffer is not None:
                pose_frame = pose_preview_buffer.latest()
                if pose_frame is not None:
                    cv2.imshow(pose_window_name, pose_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                shutdown_event.set()
                stop_event.set()
                if gesture_stop_event is not None:
                    gesture_stop_event.set()
                break

            frame = frame_buffer.latest()
            if frame is None:
                time.sleep(0.005)
                continue

            annotated, detections = detector.infer(frame)
            if auto_enabled is not None and auto_ls_controller is not None:
                if auto_enabled.get("value", False):
                    auto_ls_controller.update_from_detections(detections, frame.shape[1], frame.shape[0])
                else:
                    auto_ls_controller.suspend()
                    auto_ls_controller.neutral()

            cv2.putText(
                annotated,
                f"Source: capture | Detections: {len(detections)} | Buffer: {frame_buffer.size()}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            show = cv2.resize(annotated, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            writer.write(show)
            cv2.imshow(window_name, show)

            if pose_preview_buffer is not None and pose_writer is not None:
                pose_show = show.copy()
                pose_show[:, :] = 0
                pose_frame = pose_preview_buffer.latest()
                if pose_frame is not None:
                    pose_show = cv2.resize(pose_frame, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow(pose_window_name, pose_show)

                pose_writer.write(pose_show)
            ts_now = time.time()
            if first_ts is None:
                first_ts = ts_now
            last_ts = ts_now
            frame_count += 1
    finally:
        if gesture_stop_event is not None:
            gesture_stop_event.set()
        writer.release()
        if pose_writer is not None:
            pose_writer.release()
        stop_event.set()
        capture_thread.join(timeout=1.0)

        real_fps = _compute_real_fps(frame_count, first_ts, last_ts, args.fps)
        _reencode_with_fps(move_temp_path, output_path, real_fps)
        stack_output_path = None
        if pose_temp_path is not None and pose_writer is not None:
            _reencode_with_fps(pose_temp_path, pose_output_path, real_fps)
            stack_output_path = Path("output") / f"stack_{run_time}.mp4"
            _stack_videos_vertical(output_path, pose_output_path, stack_output_path, real_fps)

        if move_temp_path.exists():
            move_temp_path.unlink()
        if pose_temp_path is not None and pose_temp_path.exists():
            pose_temp_path.unlink()
        print(f"Export FPS (real): {real_fps:.3f}", flush=True)
        if stack_output_path is not None:
            print(f"Stacked debug video: {stack_output_path}", flush=True)


def run_video_mode(
    args,
    window_name: str,
    run_time: str,
    pose_preview_buffer: LatestFrameBuffer | None,
    pose_window_name: str,
    gesture_stop_event,
    shutdown_event: threading.Event,
    auto_enabled: Optional[dict],
    auto_ls_controller: Optional[Xbox360PIDController],
) -> None:
    detector = YoloPyTorchDetector(model_dir="model", model_name="yolov8n")

    video_file = Path("video") / args.video_file
    if not video_file.exists() or not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_file.resolve()}")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_file.resolve()}")

    output_path = Path("output") / f"move_{run_time}.mp4"
    pose_output_path = Path("output") / f"pose_{run_time}.mp4"
    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    target_fps = src_fps if src_fps > 0 else args.fps
    move_temp_path = Path("output") / f"move_{run_time}.tmp.mp4"
    writer = _create_writer(move_temp_path, target_fps)
    pose_writer = None
    pose_temp_path = None
    if pose_preview_buffer is not None:
        pose_temp_path = Path("output") / f"pose_{run_time}.tmp.mp4"
        pose_writer = _create_writer(pose_temp_path, target_fps)
    frame_count = 0
    first_ts: float | None = None
    last_ts: float | None = None

    try:
        while not shutdown_event.is_set():
            if pose_preview_buffer is not None:
                pose_frame = pose_preview_buffer.latest()
                if pose_frame is not None:
                    cv2.imshow(pose_window_name, pose_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                shutdown_event.set()
                if gesture_stop_event is not None:
                    gesture_stop_event.set()
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated, detections = detector.infer(frame_rgb)
            if auto_enabled is not None and auto_ls_controller is not None:
                if auto_enabled.get("value", False):
                    auto_ls_controller.update_from_detections(detections, frame_rgb.shape[1], frame_rgb.shape[0])
                else:
                    auto_ls_controller.suspend()
                    auto_ls_controller.neutral()

            cv2.putText(
                annotated,
                f"Source: video | {video_file.name} | Detections: {len(detections)}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            show = cv2.resize(annotated, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
            writer.write(show)
            cv2.imshow(window_name, show)

            if pose_preview_buffer is not None and pose_writer is not None:
                pose_show = show.copy()
                pose_show[:, :] = 0
                pose_frame = pose_preview_buffer.latest()
                if pose_frame is not None:
                    pose_show = cv2.resize(pose_frame, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LINEAR)
                    cv2.imshow(pose_window_name, pose_show)

                pose_writer.write(pose_show)
            ts_now = time.time()
            if first_ts is None:
                first_ts = ts_now
            last_ts = ts_now
            frame_count += 1
    finally:
        if gesture_stop_event is not None:
            gesture_stop_event.set()
        writer.release()
        if pose_writer is not None:
            pose_writer.release()
        cap.release()

        real_fps = _compute_real_fps(frame_count, first_ts, last_ts, target_fps)
        _reencode_with_fps(move_temp_path, output_path, real_fps)
        stack_output_path = None
        if pose_temp_path is not None and pose_writer is not None:
            _reencode_with_fps(pose_temp_path, pose_output_path, real_fps)
            stack_output_path = Path("output") / f"stack_{run_time}.mp4"
            _stack_videos_vertical(output_path, pose_output_path, stack_output_path, real_fps)

        if move_temp_path.exists():
            move_temp_path.unlink()
        if pose_temp_path is not None and pose_temp_path.exists():
            pose_temp_path.unlink()
        print(f"Export FPS (real): {real_fps:.3f}", flush=True)
        if stack_output_path is not None:
            print(f"Stacked debug video: {stack_output_path}", flush=True)