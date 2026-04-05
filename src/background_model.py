from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class BackgroundFilterConfig:
    alpha: float = 0.01
    diff_threshold: int = 25
    motion_threshold: int = 8
    stable_frames: int = 1000
    blur_kernel: Tuple[int, int] = (5, 5)


class AdaptiveBackgroundFilter:
    """Adaptive background model + motion stability mask for pose preprocessing.

    Core idea:
    - Maintain a floating-point background model.
    - Only update pixels that remain stable for a long enough time.
    - Use a foreground mask to cut the original color frame before MediaPipe.
    """

    def __init__(self, config: Optional[BackgroundFilterConfig] = None) -> None:
        self.config = config or BackgroundFilterConfig()
        self._background: Optional[np.ndarray] = None
        self._stable_counts: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._background = None
        self._stable_counts = None

    @staticmethod
    def _build_protected_roi_mask(
        shape: Tuple[int, int],
        landmarks: Optional[List[object]],
    ) -> Optional[np.ndarray]:
        if not landmarks:
            return None

        # MediaPipe pose landmark indices used as ROI anchors.
        head_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        left_shoulder_idx, right_shoulder_idx = 11, 12
        left_hip_idx, right_hip_idx = 23, 24

        needed = [left_shoulder_idx, right_shoulder_idx, left_hip_idx, right_hip_idx]
        if any(idx >= len(landmarks) for idx in needed):
            return None

        h, w = shape

        def _valid_point(idx: int, min_vis: float = 0.2) -> Optional[Tuple[float, float]]:
            if idx >= len(landmarks):
                return None
            lm = landmarks[idx]
            vis = float(getattr(lm, "visibility", 1.0) or 1.0)
            if vis < min_vis:
                return None
            x = float(getattr(lm, "x", 0.0))
            y = float(getattr(lm, "y", 0.0))
            return x, y

        shoulder_points = [_valid_point(left_shoulder_idx), _valid_point(right_shoulder_idx)]
        hip_points = [_valid_point(left_hip_idx), _valid_point(right_hip_idx)]
        if any(p is None for p in shoulder_points) or any(p is None for p in hip_points):
            return None

        head_points = [_valid_point(i, min_vis=0.05) for i in head_indices]
        head_points = [p for p in head_points if p is not None]

        body_points = [p for p in shoulder_points + hip_points if p is not None]
        if not body_points:
            return None

        xs = [p[0] for p in body_points]
        ys = [p[1] for p in body_points]

        if head_points:
            xs.extend(p[0] for p in head_points)
            ys.extend(p[1] for p in head_points)

        ls, rs = shoulder_points[0], shoulder_points[1]
        lh, rh = hip_points[0], hip_points[1]
        assert ls is not None and rs is not None and lh is not None and rh is not None

        shoulder_width = abs(rs[0] - ls[0])
        torso_h = abs(((lh[1] + rh[1]) * 0.5) - ((ls[1] + rs[1]) * 0.5))
        expand_x = max(0.10, shoulder_width * 0.9)
        expand_top = max(0.12, torso_h * 1.1)
        expand_bottom = max(0.15, torso_h * 1.25)

        min_x = max(0.0, min(xs) - expand_x)
        max_x = min(1.0, max(xs) + expand_x)
        min_y = max(0.0, min(ys) - expand_top)
        max_y = min(1.0, max(ys) + expand_bottom)

        x1 = int(min_x * w)
        y1 = int(min_y * h)
        x2 = int(max_x * w)
        y2 = int(max_y * h)
        if x2 <= x1 or y2 <= y1:
            return None

        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return mask

    def apply(
        self,
        frame_rgb: np.ndarray,
        protected_landmarks: Optional[List[object]] = None,
    ) -> Tuple[np.ndarray, dict]:
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("AdaptiveBackgroundFilter expects an RGB frame with 3 channels")

        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, self.config.blur_kernel, 0)
        gray_u8 = gray.astype(np.uint8)

        if self._background is None:
            self._background = gray_u8.astype(np.float32)
            self._stable_counts = np.zeros_like(gray_u8, dtype=np.uint16)
            return frame_rgb.copy(), {
                "initialized": True,
                "foreground_ratio": 1.0,
                "stable_ratio": 0.0,
            }

        background_u8 = np.clip(self._background, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(gray_u8, background_u8)

        stable_mask = diff <= self.config.motion_threshold
        unstable_mask = ~stable_mask

        assert self._stable_counts is not None
        self._stable_counts = np.where(
            stable_mask,
            np.minimum(self._stable_counts + 1, self.config.stable_frames),
            0,
        ).astype(np.uint16)

        update_mask = self._stable_counts >= self.config.stable_frames
        if np.any(update_mask):
            bg = self._background
            bg_update = gray_u8.astype(np.float32)
            bg[update_mask] = (
                (1.0 - self.config.alpha) * bg[update_mask]
                + self.config.alpha * bg_update[update_mask]
            )

        foreground_mask = (diff >= self.config.diff_threshold).astype(np.uint8) * 255
        if np.any(foreground_mask):
            kernel = np.ones((3, 3), np.uint8)
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
            foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)

        protected_mask = self._build_protected_roi_mask(gray_u8.shape, protected_landmarks)
        if protected_mask is not None:
            foreground_mask[protected_mask] = 255

        filtered = cv2.bitwise_and(frame_rgb, frame_rgb, mask=foreground_mask)
        foreground_ratio = float(np.count_nonzero(foreground_mask)) / float(foreground_mask.size)
        stable_ratio = float(np.count_nonzero(update_mask)) / float(update_mask.size)

        return filtered, {
            "initialized": False,
            "foreground_ratio": foreground_ratio,
            "stable_ratio": stable_ratio,
            "active_pixels": int(np.count_nonzero(unstable_mask)),
            "protected_roi": protected_mask is not None,
        }