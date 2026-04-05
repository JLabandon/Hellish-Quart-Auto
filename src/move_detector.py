from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from model import YoloPersonModel


__all__ = ["YoloPyTorchDetector", "YoloTensorRTDetector"]


class YoloPyTorchDetector:
    def __init__(
        self,
        model_dir: str = "model",
        model_name: str = "yolov8n",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: int = 0,
    ) -> None:
        self.runtime = "pt"
        self.model = YoloPersonModel(
            model_dir=model_dir,
            model_name=model_name,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
        )

        self.player_bbox: Optional[List[float]] = None
        self.opponent_bbox: Optional[List[float]] = None

    def reset_tracks(self) -> None:
        self.player_bbox = None
        self.opponent_bbox = None

    @staticmethod
    def _bbox_center(bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _dist_sq(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    def _normalize_people(self, people: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for person in people:
            bbox = [float(v) for v in person["bbox_xyxy"]]
            normalized.append(
                {
                    "class_name": "person",
                    "confidence": float(person.get("confidence", 0.0)),
                    "bbox_xyxy": bbox,
                    "center": self._bbox_center(bbox),
                }
            )
        return normalized

    def _initialize_player_opponent(self, people: List[Dict[str, Any]], frame_w: int, frame_h: int) -> None:
        if self.player_bbox is not None and self.opponent_bbox is not None:
            return

        mid_y = frame_h * 0.5
        left_limit = frame_w * 0.4
        right_limit = frame_w * 0.6

        left_candidates = [p for p in people if p["center"][0] <= left_limit]
        right_candidates = [p for p in people if p["center"][0] >= right_limit]

        if left_candidates and self.player_bbox is None:
            best_left = min(left_candidates, key=lambda p: abs(p["center"][1] - mid_y))
            self.player_bbox = list(best_left["bbox_xyxy"])

        if right_candidates and self.opponent_bbox is None:
            best_right = min(right_candidates, key=lambda p: abs(p["center"][1] - mid_y))
            self.opponent_bbox = list(best_right["bbox_xyxy"])

    def _match_nearest(
        self,
        prev_bbox: Optional[List[float]],
        people: List[Dict[str, Any]],
        used_indices: set[int],
    ) -> Optional[int]:
        if prev_bbox is None:
            return None
        if not people:
            return None

        prev_center = self._bbox_center(prev_bbox)
        best_idx: Optional[int] = None
        best_dist = float("inf")

        for idx, person in enumerate(people):
            if idx in used_indices:
                continue
            d2 = self._dist_sq(prev_center, person["center"])
            if d2 < best_dist:
                best_dist = d2
                best_idx = idx

        return best_idx

    def _update_tracks(self, people: List[Dict[str, Any]]) -> None:
        used: set[int] = set()

        player_idx = self._match_nearest(self.player_bbox, people, used)
        if player_idx is not None:
            self.player_bbox = list(people[player_idx]["bbox_xyxy"])
            used.add(player_idx)

        opponent_idx = self._match_nearest(self.opponent_bbox, people, used)
        if opponent_idx is not None:
            self.opponent_bbox = list(people[opponent_idx]["bbox_xyxy"])
            used.add(opponent_idx)

    def infer(self, frame_rgb: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        if frame_rgb is None:
            return frame_rgb, []

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        people = self._normalize_people(self.model.infer_people(frame_bgr))
        if not people:
            self.reset_tracks()
            return frame_bgr.copy(), []

        frame_h, frame_w = frame_bgr.shape[:2]

        self._initialize_player_opponent(people, frame_w, frame_h)
        self._update_tracks(people)

        annotated = frame_bgr.copy()
        detections: List[Dict[str, Any]] = []

        if self.player_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in self.player_bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                "Player",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "track": "player",
                    "class_name": "person",
                    "bbox_xyxy": list(self.player_bbox),
                }
            )

        if self.opponent_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in self.opponent_bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 140, 255), 2)
            cv2.putText(
                annotated,
                "Opponent",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 140, 255),
                2,
                cv2.LINE_AA,
            )
            detections.append(
                {
                    "track": "opponent",
                    "class_name": "person",
                    "bbox_xyxy": list(self.opponent_bbox),
                }
            )

        return annotated, detections


YoloTensorRTDetector = YoloPyTorchDetector
