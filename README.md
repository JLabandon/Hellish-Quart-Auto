# HellishQuartAuto

这是一个面向 Hellish Quart 的实时自动化项目：通过屏幕人物检测与前置摄像头姿势识别，将识别结果映射到虚拟手柄输入，实现半自动对战辅助与动作控制实验。

This project is a real-time automation system for Hellish Quart. It combines screen-based person detection and front-camera pose recognition, then maps those signals to virtual gamepad inputs for semi-automated combat assistance and control experiments.

## Demo



https://github.com/user-attachments/assets/cca69149-ce87-4f47-aa13-0612dfa6db08



If the embedded player is not available on your platform, open the demo directly: [output/demo.mp4](output/demo.mp4)

## 开发中 / In Progress

当前状态：开发中。

未来开发方向：
1. 协调自动移动和进攻防守的配合。
2. 提高识别准确率。



## Project Structure

```text
HellishQuartAuto/
├─ model/
│  ├─ yolov8n.pt
│  └─ pose_landmarker_lite.task (auto-downloaded if missing)
├─ output/
├─ video/
├─ src/
│  ├─ main.py
│  ├─ capture_screen.py
│  ├─ capture_camera.py
│  ├─ move_detector.py
│  ├─ pose_detector.py
│  ├─ controller.py
│  ├─ video_pipeline.py
│  └─ model.py
├─ requirements.txt
└─ README.md
```

## Runtime Architecture

- **Pipeline A:** `capture_screen.py` -> `move_detector.py` -> OpenCV preview + MP4 output
- **Pipeline B:** `capture_camera.py` -> `pose_detector.run_pose_detection(...)` -> `controller.enqueue_gesture(...)`
- **Controller output:** `controller.py` batches keyboard, AutoLS, and gesture inputs through one queue and one worker

In default `full` mode, both windows are shown at the same time:
- move window: `Screen Capture + Detection`
- pose window: `Front Camera Pose`

## Modes

### Full mode, default

```powershell
python .\src\main.py
```

This mode runs screen capture + YOLO + pose recognition together.

### Video-only mode

```powershell
python .\src\main.py --mode video --video-file demo.mp4
```

This mode only performs person detection on a video file.

### Pose-only mode

```powershell
python .\src\main.py --mode pose
```

This mode only opens the front camera and runs pose recognition.

## Gesture to Controller Mapping

- `Left Hand Forward` -> `LB`
- `Right Hand Forward` -> `RB`
- `Hands Together Up` -> `Y`
- `Hands Together Left` -> `X`
- `Hands Together Down` -> `A`
- `Hands Together Right` -> `B`

## Keyboard to Controller Mapping:
- LS: `W / A / S / D`
- Switch Auto-move: `O`
- `X / Y / A / B`: `U / I / J / K`
- `LB / RB`: `F / G`
- `LT / RT`: `R / T`
- Exit: `Esc`

## Output Files

Output depends on mode:

- `full` mode:
  - `move_<time>.mp4`
  - `pose_<time>.mp4`
  - `stack_<time>.mp4`
- `video` mode:
  - `move_<time>.mp4`
- `pose` mode:
  - `pose_<time>.mp4`

## Installation

```powershell
pip install -r requirements.txt
```

The NVDA and Virtual Controller drivers may need to be installed manually.
