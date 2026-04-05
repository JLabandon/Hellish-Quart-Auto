import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from capture_camera import start_camera_capture
from controller import KeyboardXbox360Controller, Xbox360PIDController
from pose_detector import run_pose_detection
from video_pipeline import OUTPUT_H, OUTPUT_W, LatestFrameBuffer, run_capture_mode, run_video_mode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full, video-only, or pose-only pipeline")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--buffer-size", type=int, default=5, help="Frame queue max size")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "video", "pose"], help="Run mode")
    parser.add_argument(
        "--video-file",
        type=str,
        default="",
        help="Video filename under ./video when mode is video",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    if args.buffer_size <= 0:
        raise ValueError("--buffer-size must be > 0")
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    shutdown_event = threading.Event()
    project_root = Path(__file__).resolve().parents[1]

    print("=" * 60, flush=True)
    print("Pipeline Configuration:", flush=True)
    print(f"  Mode: {args.mode}", flush=True)
    if args.mode == "video":
        print(f"  Video: {args.video_file}", flush=True)
    print("  Default gesture input: ON", flush=True)
    print("=" * 60, flush=True)

    if args.mode == "full":
        keyboard_controller = KeyboardXbox360Controller(enable_left_stick=True)
        auto_ls_controller = Xbox360PIDController(output_callback=keyboard_controller.enqueue_auto_left_stick)
        auto_enabled = {"value": False}
        gesture_stop_event = None

        def _request_shutdown() -> None:
            shutdown_event.set()
            if gesture_stop_event is not None:
                gesture_stop_event.set()

        def _toggle_auto() -> None:
            auto_enabled["value"] = not auto_enabled["value"]
            state = "ON" if auto_enabled["value"] else "OFF"
            print(f"Auto LS -> {state}", flush=True)
            auto_ls_controller.reset_output_cache()
            auto_ls_controller.suspend()
            if not auto_enabled["value"]:
                auto_ls_controller.neutral()
            else:
                keyboard_controller.clear_left_stick_override()

        keyboard_controller.toggle_auto_callback = _toggle_auto
        keyboard_controller.exit_callback = _request_shutdown
        keyboard_controller.start_listener()
        print("Auto LS -> OFF (default)", flush=True)

        gesture_thread = None
        pose_preview_buffer = None

        try:
            camera_buffer, gesture_stop_event, camera_capture_thread = start_camera_capture(
                fps=args.fps,
                buffer_size=args.buffer_size,
                camera_index=0,
                width=1280,
                height=720,
            )
            model_path = project_root / "model" / "pose_landmarker_lite.task"
            pose_preview_buffer = LatestFrameBuffer()
            gesture_thread = threading.Thread(
                target=run_pose_detection,
                kwargs={
                    "frame_buffer": camera_buffer,
                    "stop_event": gesture_stop_event,
                    "capture_thread": camera_capture_thread,
                    "model_path": model_path,
                    "controller": keyboard_controller,
                    "enable_visualization": False,
                    "num_poses": 1,
                    "min_detection_confidence": 0.5,
                    "min_presence_confidence": 0.5,
                    "min_tracking_confidence": 0.5,
                    "output_path": None,
                    "output_fps": args.fps,
                    "frame_callback": pose_preview_buffer.update,
                },
                daemon=True,
            )
            gesture_thread.start()
            print("Gesture recognition thread started", flush=True)

            window_name = "Screen Capture + Detection"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, OUTPUT_W, OUTPUT_H)
            pose_window_name = "Front Camera Pose"
            cv2.namedWindow(pose_window_name, cv2.WINDOW_NORMAL)

            run_capture_mode(
                args,
                window_name,
                run_time,
                pose_preview_buffer,
                pose_window_name,
                gesture_stop_event,
                shutdown_event,
                auto_enabled,
                auto_ls_controller,
            )
        finally:
            shutdown_event.set()
            keyboard_controller.stop_listener()
            auto_ls_controller.suspend()
            auto_ls_controller.neutral()
            if gesture_stop_event is not None:
                gesture_stop_event.set()
            if gesture_thread is not None:
                gesture_thread.join(timeout=3.0)
            cv2.destroyAllWindows()
            print("Pipeline stopped", flush=True)

    elif args.mode == "video":
        if not args.video_file:
            raise ValueError("--video-file required for video mode")

        window_name = "Screen Capture + Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, OUTPUT_W, OUTPUT_H)
        run_video_mode(
            args,
            window_name,
            run_time,
            None,
            "",
            None,
            shutdown_event,
            None,
            None,
        )
        cv2.destroyAllWindows()

    elif args.mode == "pose":
        frame_buffer, gesture_stop_event, camera_capture_thread = start_camera_capture(
            fps=args.fps,
            buffer_size=args.buffer_size,
            camera_index=0,
            width=1280,
            height=720,
        )
        try:
            run_pose_detection(
                frame_buffer=frame_buffer,
                stop_event=gesture_stop_event,
                capture_thread=camera_capture_thread,
                model_path=project_root / "model" / "pose_landmarker_lite.task",
                controller=None,
                enable_visualization=True,
                num_poses=1,
                min_detection_confidence=0.5,
                min_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_path=project_root / "output" / f"pose_{run_time}.mp4",
                output_fps=args.fps,
                frame_callback=None,
            )
        finally:
            gesture_stop_event.set()



if __name__ == "__main__":
    main()
