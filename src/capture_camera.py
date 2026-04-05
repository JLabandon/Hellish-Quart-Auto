import argparse
import os
import time
import threading
from collections import deque
from typing import Any, Deque, Optional, Tuple

import cv2
import numpy as np

__all__ = ["start_camera_capture", "CameraFrameBuffer"]


class FrontCameraWrapper:
    def __init__(
        self,
        camera_index: int = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> None:
        backend = cv2.CAP_DSHOW if os.name == "nt" else 0
        self._cap = cv2.VideoCapture(camera_index, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open front camera at index {camera_index}")

        if width is not None and width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None and height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None and fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, float(fps))

    def screenshot(self) -> Optional[np.ndarray]:
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            return None
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


class CameraFrameBuffer:
    def __init__(self, maxlen: int) -> None:
        self._frames: Deque[np.ndarray] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def append(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frames.append(frame)

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def size(self) -> int:
        with self._lock:
            return len(self._frames)


def camera_capture_loop(
    capturer: Any, buffer: CameraFrameBuffer, stop_event: threading.Event, target_fps: float
) -> None:
    frame_interval = 1.0 / target_fps if target_fps > 0 else 0.0

    while not stop_event.is_set():
        start = time.perf_counter()

        frame = capturer.screenshot()
        if frame is not None:
            buffer.append(frame)

        if frame_interval > 0:
            elapsed = time.perf_counter() - start
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def display_loop(buffer: CameraFrameBuffer, stop_event: threading.Event, window_name: str) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    target_w, target_h = 1440, 900
    try:
        cv2.resizeWindow(window_name, target_w, target_h)
    except Exception:
        pass

    while not stop_event.is_set():
        frame = buffer.latest()
        if frame is None:
            time.sleep(0.005)
            continue

        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        overlay = f"Buffered frames: {buffer.size()}"
        cv2.putText(
            bgr,
            overlay,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        try:
            display_img = cv2.resize(bgr, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        except Exception:
            display_img = bgr

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            stop_event.set()
            break

    cv2.destroyAllWindows()


def start_camera_capture(
    fps: float = 30.0,
    buffer_size: int = 5,
    camera_index: int = 0,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tuple[CameraFrameBuffer, threading.Event, threading.Thread]:
    if buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0")

    capturer = FrontCameraWrapper(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
    )
    frame_buffer = CameraFrameBuffer(maxlen=buffer_size)
    stop_event = threading.Event()

    def _run() -> None:
        try:
            camera_capture_loop(capturer, frame_buffer, stop_event, fps)
        finally:
            capturer.release()

    capture_thread = threading.Thread(target=_run, daemon=True)
    capture_thread.start()

    return frame_buffer, stop_event, capture_thread


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture the front camera and keep latest frames in a deque.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target capture FPS (default: 30)")
    parser.add_argument("--buffer-size", type=int, default=5, help="Number of latest frames to keep in deque")
    parser.add_argument("--camera-index", type=int, default=0, help="Front camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera capture width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera capture height (default: 720)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fb, stop_evt, cap_thread = start_camera_capture(
        fps=args.fps,
        buffer_size=args.buffer_size,
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
    )
    try:
        display_loop(fb, stop_evt, "Front Camera Monitor")
    finally:
        stop_evt.set()
        cap_thread.join(timeout=1.0)