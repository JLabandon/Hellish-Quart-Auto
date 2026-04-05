import argparse
import importlib
import threading
import time
from collections import deque
from typing import Deque, Optional, Any, Tuple

import cv2
import numpy as np


__all__ = ["start_screen_capture", "FrameBuffer"]


class BetterCamWrapper:
    def __init__(self, module: Any):
        try:
            self._obj = module.create(capture_output="numpy")
        except TypeError:
            self._obj = module.create()

    def screenshot(self) -> np.ndarray:
        # Try common capture method names on the BetterCam object.
        for name in ("screenshot", "capture", "get_frame", "get_image", "grab", "read"):
            fn = getattr(self._obj, name, None)
            if fn is None:
                continue
            try:
                res = fn()
            except TypeError:
                continue

            if isinstance(res, tuple) and len(res) >= 2:
                frame = res[1]
            else:
                frame = res

            return frame

        raise AttributeError("BetterCam object has no known capture method")


class FrameBuffer:
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


def capture_loop(
    capturer: Any, buffer: FrameBuffer, stop_event: threading.Event, target_fps: float
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


def display_loop(buffer: FrameBuffer, stop_event: threading.Event, window_name: str) -> None:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    TARGET_W, TARGET_H = 1440, 900
    try:
        cv2.resizeWindow(window_name, TARGET_W, TARGET_H)
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
            display_img = cv2.resize(bgr, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
        except Exception:
            display_img = bgr

        cv2.imshow(window_name, display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # q or ESC
            stop_event.set()
            break

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture the screen with BetterCam and keep latest frames in a deque."
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target capture FPS (default: 30)")
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5,
        help="Number of latest frames to keep in deque (default: 5)",
    )
    return parser.parse_args()


def start_screen_capture(
    fps: float = 30.0, buffer_size: int = 5
) -> Tuple[FrameBuffer, threading.Event, threading.Thread]:
    if buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0")

    bc_mod = importlib.import_module("bettercam")
    capturer = BetterCamWrapper(bc_mod)
    frame_buffer = FrameBuffer(maxlen=buffer_size)
    stop_event = threading.Event()

    capture_thread = threading.Thread(
        target=capture_loop,
        args=(capturer, frame_buffer, stop_event, fps),
        daemon=True,
    )
    capture_thread.start()

    return frame_buffer, stop_event, capture_thread


if __name__ == "__main__":
    args = parse_args()
    fb, stop_evt, cap_thread = start_screen_capture(fps=args.fps, buffer_size=args.buffer_size)
    try:
        display_loop(fb, stop_evt, "BetterCam Monitor")
    finally:
        stop_evt.set()
        cap_thread.join(timeout=1.0)
