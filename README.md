# Smart Exam Hall Monitoring System

A real-time AI-powered examination monitoring system for Nigerian university exam halls, built with YOLOv8n and MediaPipe. It watches candidates through a webcam, flags four types of suspicious behaviour, and alerts a human invigilator through a live web dashboard. It is a decision-support tool, not a replacement for the invigilator — a human always makes the final call.

## Modules

| Module | Status | Technology |
|--------|--------|------------|
| Phone Detection | ✅ Built & Tested | YOLOv8n (fine-tuned) |
| Gaze Detection | ✅ Built & Tested | MediaPipe Face Mesh + PnP |
| Posture Analysis | ✅ Built & Tested | MediaPipe Pose |
| Object Passing | ✅ Built | MediaPipe Hands + YOLOv8n (fine-tuned + generic COCO) |
| Alert Manager | ✅ Built & Tested | Scoring engine + SQLite logging |
| Flask Dashboard | ✅ Built | Flask + Server-Sent Events |

No facial recognition is used anywhere in this system, by design — candidate identity is a stateless 5×4 grid position (e.g. `R2C3`), never a face or name, computed fresh from the current frame with no persistent memory between frames.

## What each module detects

- **Phone Detection** — fires immediately on a single qualifying frame (no waiting period), since phone possession is treated as unambiguous. Confidence-weighted scoring per Chapter 3 Table 3.7.
- **Gaze/Head Detection** — flags sideways looking (yaw beyond ±30°) only. Pitch (up/down head tilt) was tested and removed from scoring after real exam-hall use showed it couldn't distinguish genuine suspicious behaviour from ordinary writing posture — heads naturally tilt down while writing, and looking up has no plausible use for cheating. The pitch estimator and calibration machinery remain in the code for the on-screen debug display, but no longer trigger alerts.
- **Posture Analysis** — personalised baseline per candidate (mean + 2×std of their own natural posture), rather than a one-size-fits-all threshold.
- **Object Passing** — tracks hand movement between grid zones via MediaPipe Hands, combined with YOLO object detection to check if the hand is holding something. Includes a grace window and burst-detection suppression so mass attendance-sheet distribution at the start of an exam isn't flagged as cheating.

## Scoring & alerts

`alert_manager.py` combines base score × duration multiplier × confidence weight × combination bonus (when two different modules flag the same candidate within 30 seconds) into a running per-candidate score, with a high-confidence phone override, score decay for clean behaviour, and separate cooldowns for yellow (60+) and red (75+) alert thresholds. Every alert is logged to SQLite along with the incident that triggered it.

## Dashboard

`app/dashboard/app_real.py` runs the full four-module pipeline plus the real `AlertManager`, and serves a live web dashboard:

- Live MJPEG video feed (with a full-screen toggle — alerts stay visible and audible even in full screen)
- A seat map that highlights flagged candidates by grid position, never showing coordinates to the invigilator directly
- Live-updating candidate score bars
- An alert log backed by SQLite, so history survives a page refresh, with session-boundary dividers separating different testing sessions
- Red alerts: an interrupting popup + audio chime. Yellow alerts: a non-blocking toast notification, shown once per candidate until they either escalate to red or the page reloads
- Snapshot capture at the moment of each alert, with click-to-preview and a download button
- CSV export of the full alert log

**Known limitation:** the dashboard currently has no authentication — anyone with the URL on the same network can view the live feed and alert history. All dashboard routes are read-only (nothing can be deleted or modified through the web interface), but this should not be run on a shared or untrusted network without adding access control first.

## Project structure

```
Smart_Exam_hall_monitoring_system/
├── app/
│   ├── main.py                # Integrated run, terminal status window (no dashboard)
│   ├── alert_manager.py
│   └── dashboard/
│       ├── app_real.py        # Full pipeline + live dashboard (real detection)
│       ├── app_prototype.py   # Earlier fake-data SSE test version
│       ├── snapshots/         # Captured detection snapshots (generated at runtime)
│       └── templates/
│           └── dashboard.html
├── modules/
│   ├── __init__.py
│   ├── phone_detection.py
│   ├── gaze_detection.py
│   ├── posture_analysis.py
│   └── object_passing.py
├── models/
│   ├── phone_detector_best.pt  # Fine-tuned phone detector (Chapter 3)
│   └── yolov8n.pt              # Stock COCO model, generic object fallback (Section 3.9)
├── database/                  # SQLite alert/incident database
├── test_videos/
├── tests/
│   ├── test_phone_detection.py
│   ├── test_gaze_detection.py
│   ├── test_posture_analysis.py
│   ├── test_object_passing.py
│   └── test_alert_manager.py
├── config.yaml
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.11 (MediaPipe is incompatible with newer versions).

```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```


## Testing each module in isolation

Run from the project root (module-mode, so `config.yaml` resolves correctly):
```bash
python -m tests.test_phone_detection --source 0
python -m tests.test_gaze_detection --source 0
python -m tests.test_posture_analysis --source 0
python -m tests.test_object_passing --source 0
python -m tests.test_alert_manager
```

## Running the full system

Terminal-only, no dashboard (lightweight status window):
```bash
python -m app.main --source 0
```

Full live dashboard (run from the project root):
```bash
python -m app.dashboard.app_real --source 0
```
Then open `http://localhost:5000` in a browser.

## Known limitations (see Chapter 5 for full detail)

- Object Passing's object-recognition step still misses many held items (papers, notebooks) under real lighting/distance conditions — hand-crossing detection itself works reliably, but confirming *what* was passed is the current bottleneck, evidenced by real debug-log testing rather than assumed.
- Single camera only — the architecture carries a `camera_id` field throughout in anticipation of multi-camera support, but the pipeline currently only opens one video source.
- No dashboard authentication (see above).
- Video-clip capture (a rolling buffer around each detection) is not yet built — only still-image snapshots are captured currently.
- CPU-only inference on current test hardware (Intel integrated graphics, no NVIDIA/CUDA path available) is the primary constraint on Object Passing's throughput; `frame_skip` values in `config.yaml` are tuned against this reality, not assumed GPU acceleration.
