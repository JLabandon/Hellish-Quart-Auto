from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import vgamepad as vg
from pynput import keyboard


__all__ = ["Xbox360PIDController", "KeyboardXbox360Controller", "main"]


LS_MAX = 32767
LS_MIN = -32768
LS_NEUTRAL = 0
DEFAULT_TARGET_RATIO = 1.8
DEFAULT_DEADBAND = 0.05
GESTURE_MAX_HOLD_SEC = 0.35


KEY_BINDINGS: Dict[str, str] = {
    "w": "LS_UP",
    "s": "LS_DOWN",
    "a": "LS_LEFT",
    "d": "LS_RIGHT",
    "o": "TOGGLE_AUTO",
    "u": "X",
    "i": "Y",
    "j": "A",
    "k": "B",
    "f": "LB",
    "g": "RB",
    "r": "LT",
    "t": "RT",
}

LS_UP = 32767
LS_DOWN = -32768
LS_LEFT = -32768
LS_RIGHT = 32767


@dataclass
class PIDState:
    kp: float = 8
    ki: float = 0.1
    kd: float = 2
    integral_limit: float = 1.0
    output_limit: float = 1.0
    integral: float = 0.0
    prev_error: float = 0.0
    has_prev: bool = False

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.has_prev = False

    def step(self, error: float, dt: float) -> float:
        dt = max(dt, 1e-4)
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))

        derivative = 0.0
        if self.has_prev:
            derivative = (error - self.prev_error) / dt
        self.prev_error = error
        self.has_prev = True

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(0.0, min(self.output_limit, output))


class Xbox360PIDController:
    def __init__(
        self,
        target_ratio: float = DEFAULT_TARGET_RATIO,
        deadband: float = DEFAULT_DEADBAND,
        pid: Optional[PIDState] = None,
        output_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> None:
        self.gamepad = vg.VX360Gamepad()
        self.target_ratio = target_ratio
        self.deadband = deadband
        self.pid = pid or PIDState()
        self.output_callback = output_callback
        self._last_time: Optional[float] = None
        self._last_command: Optional[Tuple[int, int]] = None

    @staticmethod
    def _extract_tracked_boxes(detections: Sequence[Dict[str, object]]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
        player_bbox: Optional[List[float]] = None
        opponent_bbox: Optional[List[float]] = None

        for detection in detections:
            track = detection.get("track")
            bbox = detection.get("bbox_xyxy")
            if track == "player" and isinstance(bbox, list):
                player_bbox = [float(v) for v in bbox]
            elif track == "opponent" and isinstance(bbox, list):
                opponent_bbox = [float(v) for v in bbox]

        return player_bbox, opponent_bbox

    @staticmethod
    def _center_x(bbox: Sequence[float]) -> float:
        x1, _, x2, _ = bbox
        return (x1 + x2) * 0.5

    def _emit(self, message: str) -> None:
        print(message, flush=True)

    def _set_left_stick(self, x_value: int, y_value: int = 0, reason: str = "") -> None:
        command = (x_value, y_value)
        if self._last_command == command:
            return

        if self.output_callback is not None:
            self.output_callback(x_value, y_value, reason)
        else:
            if reason:
                self._emit(f"LS -> x={x_value}, y={y_value} | {reason}")
            else:
                self._emit(f"LS -> x={x_value}, y={y_value}")
            self.gamepad.left_joystick(x_value=x_value, y_value=y_value)
            self.gamepad.update()
        self._last_command = command

    def neutral(self) -> None:
        self.pid.reset()
        self._set_left_stick(LS_NEUTRAL, LS_NEUTRAL, "neutral")

    def suspend(self) -> None:
        self.pid.reset()
        self._last_time = None

    def reset_output_cache(self) -> None:
        self._last_command = None

    def update_from_detections(
        self,
        detections: Sequence[Dict[str, object]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        player_bbox, opponent_bbox = self._extract_tracked_boxes(detections)
        self.update_from_boxes(player_bbox, opponent_bbox, frame_width, frame_height)

    def update_from_boxes(
        self,
        player_bbox: Optional[List[float]],
        opponent_bbox: Optional[List[float]],
        frame_width: int,
        _frame_height: int,
    ) -> None:
        if player_bbox is None or opponent_bbox is None:
            self.neutral()
            return

        player_x = self._center_x(player_bbox)
        opponent_x = self._center_x(opponent_bbox)

        player_width = player_bbox[2] - player_bbox[0]
        opponent_width = opponent_bbox[2] - opponent_bbox[0]
        avg_width = (player_width + opponent_width) * 0.5

        if avg_width <= 0:
            self.neutral()
            return

        actual_distance = abs(opponent_x - player_x)
        target_distance = self.target_ratio * avg_width
        error = abs(actual_distance - target_distance)
        
        distance_ratio = actual_distance / avg_width

        if error <= self.deadband * avg_width:
            self.neutral()
            return

        now = time.perf_counter()
        dt = 1.0 / 60.0 if self._last_time is None else now - self._last_time
        self._last_time = now

        magnitude = self.pid.step(error, dt)
        if magnitude <= 0.0:
            self.neutral()
            return

        if actual_distance > target_distance:
            direction = 1 if opponent_x >= player_x else -1
            mode = "toward"
        else:
            direction = -1 if opponent_x >= player_x else 1
            mode = "away"

        x_value = int(direction * magnitude * LS_MAX)
        x_value = max(LS_MIN, min(LS_MAX, x_value))

        self._set_left_stick(
            x_value,
            0,
            f"{mode} | dist_ratio={distance_ratio:.3f}x target={self.target_ratio:.3f}x err={error:.1f}px mag={magnitude:.3f}",
        )


class KeyboardXbox360Controller:
    def __init__(
        self,
        gamepad: Optional[vg.VX360Gamepad] = None,
        enable_left_stick: bool = True,
        toggle_auto_callback: Optional[Callable[[], None]] = None,
        exit_callback: Optional[Callable[[], None]] = None,
    ) -> None:
        self.gamepad = gamepad if gamepad is not None else vg.VX360Gamepad()
        self.enable_left_stick = enable_left_stick
        self.toggle_auto_callback = toggle_auto_callback
        self.exit_callback = exit_callback

        # Producer-side state (keyboard listener thread)
        self._pressed_actions: Set[str] = set()
        self._stick_up = False
        self._stick_down = False
        self._stick_left = False
        self._stick_right = False

        # Runtime threads
        self._listener: Optional[keyboard.Listener] = None
        self._command_queue: "queue.Queue[Tuple[str, dict]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Consumer-side merged state (worker thread)
        self._kbd_override_active = False
        self._kbd_left_stick: Tuple[int, int] = (0, 0)
        self._auto_left_stick: Tuple[int, int] = (0, 0)
        self._active_gesture_button: Optional[str] = None
        self._gesture_last_seen_t = 0.0
        self._gesture_max_hold_sec: float = GESTURE_MAX_HOLD_SEC

        self._gesture_button_map: Dict[str, str] = {
            "Left Hand Forward": "LB",
            "Right Hand Forward": "RB",
            "Hands Together Up": "Y",
            "Hands Together Left": "X",
            "Hands Together Down": "A",
            "Hands Together Right": "B",
            # "Hands Together Center": "LT",
        }

    @staticmethod
    def _key_name(key: keyboard.Key | keyboard.KeyCode) -> Optional[str]:
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        if key == keyboard.Key.up:
            return "up"
        if key == keyboard.Key.down:
            return "down"
        if key == keyboard.Key.left:
            return "left"
        if key == keyboard.Key.right:
            return "right"
        return None

    def _emit(self, message: str) -> None:
        print(message, flush=True)

    @staticmethod
    def _key_label(key: keyboard.Key | keyboard.KeyCode) -> str:
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char
        return str(key)

    def _enqueue(self, cmd: str, payload: dict) -> None:
        self._command_queue.put((cmd, payload))

    def _keyboard_stick_state(self) -> Tuple[int, int, bool]:
        if not self.enable_left_stick:
            return 0, 0, False

        x_value = 0
        y_value = 0

        if self._stick_left and not self._stick_right:
            x_value = LS_LEFT
        elif self._stick_right and not self._stick_left:
            x_value = LS_RIGHT

        if self._stick_up and not self._stick_down:
            y_value = LS_UP
        elif self._stick_down and not self._stick_up:
            y_value = LS_DOWN

        active = any((self._stick_up, self._stick_down, self._stick_left, self._stick_right))
        return x_value, y_value, active

    def _enqueue_keyboard_stick(self) -> None:
        x_value, y_value, active = self._keyboard_stick_state()
        self._enqueue(
            "set_keyboard_stick",
            {
                "x": x_value,
                "y": y_value,
                "active": active,
            },
        )

    def _set_left_stick(self) -> None:
        self._enqueue_keyboard_stick()

    def enqueue_auto_left_stick(self, x_value: int, y_value: int = 0, reason: str = "") -> None:
        self._enqueue(
            "set_auto_stick",
            {
                "x": int(x_value),
                "y": int(y_value),
                "reason": reason,
            },
        )

    def enqueue_gesture(self, gesture_label: Optional[str]) -> None:
        self._enqueue("gesture_update", {"label": gesture_label})

    def _check_gesture_timeout(self) -> bool:
        if self._active_gesture_button is None:
            return False
        now = time.perf_counter()
        if (now - self._gesture_last_seen_t) < self._gesture_max_hold_sec:
            return False

        action = self._active_gesture_button
        self._apply_button_release(action)
        self._emit(f"Gesture timeout(worker) -> Release -> {action}")
        self._active_gesture_button = None
        return True

    def _apply_left_stick(self, x_value: int, y_value: int, source: str, reason: str = "") -> None:
        if reason:
            self._emit(f"{source}_LS -> x={x_value}, y={y_value} | {reason}")
        else:
            self._emit(f"{source}_LS -> x={x_value}, y={y_value}")
        self.gamepad.left_joystick(x_value=int(x_value), y_value=int(y_value))

    def _apply_button_press(self, action: str) -> None:
        if action == "X":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        elif action == "Y":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        elif action == "A":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        elif action == "B":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        elif action == "LB":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        elif action == "RB":
            self.gamepad.press_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        elif action == "LT":
            self.gamepad.left_trigger(value=255)
        elif action == "RT":
            self.gamepad.right_trigger(value=255)

    def _apply_button_release(self, action: str) -> None:
        if action == "X":
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_X)
        elif action == "Y":
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_Y)
        elif action == "A":
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
        elif action == "B":
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_B)
        elif action == "LB":
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_SHOULDER)
        elif action == "RB":
            self.gamepad.release_button(button=vg.XUSB_BUTTON.XUSB_GAMEPAD_RIGHT_SHOULDER)
        elif action == "LT":
            self.gamepad.left_trigger(value=0)
        elif action == "RT":
            self.gamepad.right_trigger(value=0)

    def _handle_gesture_update(self, gesture_label: Optional[str]) -> bool:
        next_button = self._gesture_button_map.get(gesture_label) if gesture_label else None

        # Same gesture: refresh keepalive timer, no press/release spam.
        if next_button == self._active_gesture_button and next_button is not None:
            self._gesture_last_seen_t = time.perf_counter()
            return False

        dirty = False
        if self._active_gesture_button is not None and self._active_gesture_button != next_button:
            prev = self._active_gesture_button
            self._emit(f"Gesture changed -> Release -> {prev}")
            self._apply_button_release(prev)
            self._active_gesture_button = None
            dirty = True

        if next_button is not None and next_button != self._active_gesture_button:
            self._emit(f"Gesture -> {gesture_label} | Press -> {next_button}")
            self._apply_button_press(next_button)
            self._active_gesture_button = next_button
            self._gesture_last_seen_t = time.perf_counter()
            dirty = True

        return dirty

    def _press_button(self, action: str) -> None:
        self._emit(f"Press -> {action}")
        self._apply_button_press(action)

    def _release_button(self, action: str) -> None:
        self._emit(f"Release -> {action}")
        self._apply_button_release(action)

    def _process_command(self, cmd: str, payload: dict) -> bool:
        dirty = False

        if cmd == "set_keyboard_stick":
            x_value = int(payload["x"])
            y_value = int(payload["y"])
            active = bool(payload["active"])
            self._kbd_left_stick = (x_value, y_value)
            self._kbd_override_active = active

            if self._kbd_override_active:
                self._apply_left_stick(x_value, y_value, source="KBD")
            else:
                ax, ay = self._auto_left_stick
                self._apply_left_stick(ax, ay, source="AUTO")
            dirty = True

        elif cmd == "set_auto_stick":
            x_value = int(payload["x"])
            y_value = int(payload["y"])
            reason = str(payload.get("reason", ""))
            self._auto_left_stick = (x_value, y_value)

            # Priority: keyboard WASD overrides auto left stick.
            if not self._kbd_override_active:
                self._apply_left_stick(x_value, y_value, source="AUTO", reason=reason)
                dirty = True

        elif cmd == "press_button":
            self._press_button(payload["action"])
            dirty = True

        elif cmd == "release_button":
            self._release_button(payload["action"])
            dirty = True

        elif cmd == "gesture_update":
            dirty = self._handle_gesture_update(payload.get("label")) or dirty

        return dirty

    def on_press(self, key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        if key == keyboard.Key.esc:
            self._emit("Exit -> ESC")
            if self.exit_callback is not None:
                self.exit_callback()
            return False

        key_name = self._key_name(key)
        if key_name is None:
            self._emit(f"Ignored -> {self._key_label(key)}")
            return None

        action = KEY_BINDINGS.get(key_name)
        if action is None:
            self._emit(f"Ignored -> {self._key_label(key)}")
            return None

        if action == "TOGGLE_AUTO":
            if self.toggle_auto_callback is not None:
                self.toggle_auto_callback()
            else:
                self._emit("Toggle auto ignored: no callback")
            return None

        if action == "LS_UP":
            self._stick_up = True
            self._set_left_stick()
            return None
        if action == "LS_DOWN":
            self._stick_down = True
            self._set_left_stick()
            return None
        if action == "LS_LEFT":
            self._stick_left = True
            self._set_left_stick()
            return None
        if action == "LS_RIGHT":
            self._stick_right = True
            self._set_left_stick()
            return None

        if action in self._pressed_actions:
            return None

        self._pressed_actions.add(action)
        self._enqueue("press_button", {"action": action})
        return None

    def on_release(self, key: keyboard.Key | keyboard.KeyCode) -> bool | None:
        if key == keyboard.Key.esc:
            return False

        key_name = self._key_name(key)
        if key_name is None:
            self._emit(f"Ignored(release) -> {self._key_label(key)}")
            return None

        action = KEY_BINDINGS.get(key_name)
        if action is None:
            self._emit(f"Ignored(release) -> {self._key_label(key)}")
            return None

        if action == "TOGGLE_AUTO":
            return None

        if action == "LS_UP":
            self._stick_up = False
            self._set_left_stick()
            return None
        if action == "LS_DOWN":
            self._stick_down = False
            self._set_left_stick()
            return None
        if action == "LS_LEFT":
            self._stick_left = False
            self._set_left_stick()
            return None
        if action == "LS_RIGHT":
            self._stick_right = False
            self._set_left_stick()
            return None

        if action not in self._pressed_actions:
            return None

        self._pressed_actions.remove(action)
        self._enqueue("release_button", {"action": action})
        return None

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                cmd, payload = self._command_queue.get(timeout=0.1)
            except queue.Empty:
                if self._check_gesture_timeout():
                    self.gamepad.update()
                continue

            if cmd == "__stop__":
                break
            dirty = self._process_command(cmd, payload)

            while True:
                try:
                    cmd, payload = self._command_queue.get_nowait()
                except queue.Empty:
                    break

                if cmd == "__stop__":
                    if dirty:
                        self.gamepad.update()
                    return
                dirty = self._process_command(cmd, payload) or dirty

            dirty = self._check_gesture_timeout() or dirty
            if dirty:
                self.gamepad.update()

    def start_listener(self) -> None:
        if self._listener is not None:
            return
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self._listener.start()

    def stop_listener(self) -> None:
        if self._listener is None:
            return
        if self._active_gesture_button is not None:
            self._enqueue("release_button", {"action": self._active_gesture_button})
            self._active_gesture_button = None
        self._listener.stop()
        self._listener = None
        self._stop_event.set()
        self._enqueue("__stop__", {})
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def clear_left_stick_override(self) -> None:
        self._stick_up = False
        self._stick_down = False
        self._stick_left = False
        self._stick_right = False
        self._enqueue_keyboard_stick()

    def is_left_stick_override_active(self) -> bool:
        if not self.enable_left_stick:
            return False
        return self._stick_up or self._stick_down or self._stick_left or self._stick_right


def main() -> None:
    controller = KeyboardXbox360Controller()
    print("Keyboard -> Xbox 360 controller started.", flush=True)
    print("Mappings:", flush=True)
    print("  LS: W / A / S / D", flush=True)
    print("  Toggle Auto LS: O", flush=True)
    print("  X: Y, Y: U, A: H, B: J", flush=True)
    print("  LB: F, RB: G, LT: R, RT: T", flush=True)
    print("Press ESC to quit.", flush=True)

    with keyboard.Listener(on_press=controller.on_press, on_release=controller.on_release) as listener:
        listener.join()


if __name__ == "__main__":
    main()
