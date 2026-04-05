from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from ultralytics import YOLO


class YoloPersonModel:
    def __init__(
        self,
        model_dir: str = "model",
        model_name: str = "yolov8n",
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        device: int = 0,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.pt_path = self.model_dir / f"{self.model_name}.pt"

        self._ensure_pt_file()
        self.model = YOLO(str(self.pt_path))

    def _ensure_pt_file(self) -> None:
        if not self.pt_path.exists():
            YOLO(f"{self.model_name}.pt")
            downloaded = Path(f"{self.model_name}.pt")
            if downloaded.exists() and downloaded.resolve() != self.pt_path.resolve():
                downloaded.replace(self.pt_path)

        if not self.pt_path.exists():
            raise FileNotFoundError(f"Missing model file: {self.pt_path}")

    def infer_people(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        names = result.names

        people: List[Dict[str, Any]] = []
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            is_person = cls_name.lower() == "person" or cls_id == 0
            if not is_person:
                continue

            score = float(box.conf[0].item())
            bbox = [float(v) for v in box.xyxy[0].tolist()]
            people.append(
                {
                    "class_id": cls_id,
                    "class_name": "person",
                    "confidence": score,
                    "bbox_xyxy": bbox,
                }
            )

        return people
