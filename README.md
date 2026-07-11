# Smart Exam Hall Monitoring System

A real-time AI-powered examination monitoring system built with YOLOv8 and MediaPipe.

## Modules
| Module | Status | Technology |
|--------|--------|------------|
| Phone Detection | ✅ Built & Tested | YOLOv8n |
| Gaze Detection | ✅ Built | MediaPipe Face Mesh |
| Posture Analysis | ✅ Built | MediaPipe Pose |
| Object Passing | ⬜ Pending | YOLOv8 + Zone Tracking |
| Alert Manager | ⬜ Pending | Pure Python |
| Flask Dashboard | ⬜ Pending | Flask + SSE |

## Project Structure
```
Smart_Exam_hall_monitoring_system/
├── modules/
│   ├── __init__.py
│   ├── phone_detection.py
│   ├── gaze_detection.py
│   └── posture_analysis.py
├── models/
│   └── phone_detector_best.pt
├── test_videos/
├── config.yaml
├── test_phone_detection.py
├── test_gaze_detection.py
└── test_posture_analysis.py
```

## Setup
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install mediapipe ultralytics opencv-python pyyaml
```

## Testing Each Module in Isolation
```bash
python test_phone_detection.py --source 0
python test_gaze_detection.py --source 0
python test_posture_analysis.py --source 0
```
