"""
Step 3: Flask dashboard wired to the REAL detection pipeline.

This replaces the fake_alert_generator() from Step 2 with the actual
four detection modules + AlertManager, exactly as main.py runs them —
the only difference is the on_alert callback now pushes to SSE clients
instead of just logging to the terminal, and the camera loop feeds an
MJPEG stream instead of (or alongside) the cv2.imshow debug window.

This does NOT modify main.py or alert_manager.py's core behaviour —
it reuses them as-is. The one real change made to alert_manager.py
(passing 'behaviour' and 'camera_id' through to on_alert) is required
because the dashboard's alert log needs to show what triggered each
alert, and the previous on_alert payload didn't carry that.

Run:
    python app_real.py --source 0
    python app_real.py --source 1

Then open http://localhost:5000
"""

import argparse
import json
import os
import queue
import threading
import time

import cv2
import numpy as np
import yaml
from flask import Flask, Response, render_template, send_from_directory

from modules.phone_detection import PhoneDetectionModule
from modules.gaze_detection import GazeDetectionModule
from modules.posture_analysis import PostureAnalysisModule
from modules.object_passing import ObjectPassingModule
from alert_manager import AlertManager

app = Flask(__name__)

_frame_lock = threading.Lock()
_latest_frame = None
_running = True

_sse_clients = []
_sse_clients_lock = threading.Lock()
_alert_manager = None

# Snapshots are saved next to this file, named "{candidate_id}_{timestamp}.jpg".
# This filename convention (rather than a DB column) means /history can
# reconstruct the expected filename from data alert_manager already logs
# (candidate_id + timestamp), with no schema change and no second write
# racing against alert_manager's own DB write in _log_alert().
SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def snapshot_filename(candidate_id: str, timestamp: float) -> str:
    return f"{candidate_id}_{timestamp:.3f}.jpg"


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_snapshot(candidate_id: str, timestamp: float, bbox) -> bool:
    """Grabs the current live frame, draws the triggering detection's
    bounding box on it (if available), and saves it to disk. Uses
    whatever frame is live in _latest_frame AT THE MOMENT the alert
    fires — not the exact frame the detection ran on (that frame is
    long gone by the time an alert reaches this callback, given the
    per-module queues and alert_manager's own processing thread). For
    a snapshot's purpose (invigilator context, not forensic-grade
    frame-accuracy), a frame within a fraction of a second of the real
    detection is an acceptable, documented approximation."""
    with _frame_lock:
        frame_bytes = _latest_frame

    if frame_bytes is None:
        return False

    frame = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return False

    # object_passing uses (0,0,0,0) as a placeholder since it isn't a
    # single-box detection by nature (see its DetectionEvent construction)
    # — skip drawing in that case rather than drawing a meaningless box
    # in the frame's corner.
    if bbox and tuple(bbox) != (0, 0, 0, 0):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    filename = snapshot_filename(candidate_id, timestamp)
    path = os.path.join(SNAPSHOT_DIR, filename)
    ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bool(ok)


def broadcast_alert(alert_dict: dict):
    """on_alert callback passed into AlertManager — pushes the real alert
    to every connected SSE client, reformatted to match what the Step 2
    front-end already expects (candidate_id, row, col, behaviour, score,
    level, camera_id, timestamp), plus a snapshot filename if capture
    succeeded."""
    candidate_id = alert_dict["candidate_id"]  # e.g. "R2C3"
    timestamp = alert_dict["timestamp"]
    row, col = None, None
    try:
        # candidate_id format is "R{row}C{col}" per CandidateZoneTracker
        r_part, c_part = candidate_id[1:].split("C")
        row, col = int(r_part), int(c_part)
    except (ValueError, IndexError):
        pass  # if format ever changes, seat map just won't highlight this one

    snapshot_saved = save_snapshot(candidate_id, timestamp, alert_dict.get("bbox"))

    event = {
        "candidate_id": candidate_id,
        "row": row,
        "col": col,
        "behaviour": alert_dict.get("behaviour", "unknown"),
        "score": round(alert_dict["score"], 1),
        "level": alert_dict["level"],
        "camera_id": alert_dict.get("camera_id", "cam_0"),
        "timestamp": time.strftime("%H:%M:%S", time.localtime(timestamp)),
        "snapshot": snapshot_filename(candidate_id, timestamp) if snapshot_saved else None,
    }

    with _sse_clients_lock:
        for client_queue in _sse_clients:
            client_queue.put(event)


def pipeline_loop(source, config):
    """Mirrors main.py's ExamMonitoringSystem.run() camera loop, but feeds
    the Flask MJPEG stream instead of (or alongside) cv2.imshow."""
    global _latest_frame

    is_index = str(source).isdigit()
    src = int(source) if is_index else source
    backend = cv2.CAP_DSHOW if is_index else 0

    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        print(f"[ERROR] Could not open source {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config["camera"]["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config["camera"]["height"])

    alert_queue = queue.Queue(maxsize=config["queues"]["max_size"] * 4)

    phone_module = PhoneDetectionModule(config, alert_queue)
    gaze_module = GazeDetectionModule(config, alert_queue)
    posture_module = PostureAnalysisModule(config, alert_queue)
    object_passing_module = ObjectPassingModule(config, alert_queue)
    alert_manager = AlertManager(config, alert_queue, on_alert=broadcast_alert)
    global _alert_manager
    _alert_manager = alert_manager

    for m in (phone_module, gaze_module, posture_module, object_passing_module):
        m.start()
    alert_manager.start()
    time.sleep(2.0)  # let model loading finish, same as main.py

    frame_number = 0
    print("[INFO] Real detection pipeline started")

    try:
        while _running:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame read failed, retrying...")
                time.sleep(0.1)
                continue

            frame_number += 1
            phone_module.put_frame(frame, frame_number)
            gaze_module.put_frame(frame, frame_number)
            posture_module.put_frame(frame, frame_number)
            object_passing_module.put_frame(frame, frame_number)

            ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with _frame_lock:
                    _latest_frame = buffer.tobytes()
    finally:
        for m in (phone_module, gaze_module, posture_module, object_passing_module):
            m.stop()
        alert_manager.stop()
        cap.release()
        print("[INFO] Pipeline shut down")


def generate_mjpeg():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.05)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/history")
def history():
    """Returns recent alert history from SQLite so the dashboard's alert
    log, score bars, and seat map can repopulate after a page refresh
    instead of starting empty. Uses the same AlertManager instance the
    live pipeline is already writing to."""
    if _alert_manager is None:
        return json.dumps([]), 200, {"Content-Type": "application/json"}

    rows = _alert_manager.get_recent_alerts(limit=50)
    events = []
    for row in reversed(rows):  # oldest first, so front-end prepend logic matches live order
        candidate_id = row["candidate_id"]
        timestamp = row["timestamp"]
        row_num, col_num = None, None
        try:
            r_part, c_part = candidate_id[1:].split("C")
            row_num, col_num = int(r_part), int(c_part)
        except (ValueError, IndexError):
            pass

        # Reconstruct the expected snapshot filename and confirm it
        # actually exists on disk before offering it — older alerts
        # logged before snapshot capture existed won't have one.
        expected_name = snapshot_filename(candidate_id, timestamp)
        has_snapshot = os.path.exists(os.path.join(SNAPSHOT_DIR, expected_name))

        events.append({
            "candidate_id": candidate_id,
            "row": row_num,
            "col": col_num,
            "behaviour": row["behaviour_type"] or "unknown",
            "score": round(row["score"], 1),
            "level": row["alert_level"],
            "camera_id": row["camera_id"] or "cam_0",
            "timestamp": time.strftime("%H:%M:%S", time.localtime(timestamp)),
            "snapshot": expected_name if has_snapshot else None,
        })

    return json.dumps(events), 200, {"Content-Type": "application/json"}


@app.route("/snapshots/<path:filename>")
def snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/video_feed")
def video_feed():
    return Response(generate_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/alerts")
def alerts():
    def stream():
        client_queue = queue.Queue()
        with _sse_clients_lock:
            _sse_clients.append(client_queue)
        try:
            while _running:
                # A timeout here (instead of an unbounded blocking get())
                # is required so this generator periodically wakes up and
                # checks _running. Without it, an open browser tab keeps
                # this thread parked on get() forever, which was preventing
                # the terminal from returning control after Ctrl+C/process
                # shutdown until every connected tab was closed manually.
                try:
                    event = client_queue.get(timeout=15)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    # SSE comment line (ignored by EventSource) — keeps the
                    # connection alive through proxies/timeouts and gives
                    # us a regular checkpoint to re-check _running.
                    yield ": heartbeat\n\n"
        finally:
            with _sse_clients_lock:
                if client_queue in _sse_clients:
                    _sse_clients.remove(client_queue)

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    pipeline_thread = threading.Thread(
        target=pipeline_loop, args=(args.source, config), daemon=True
    )
    pipeline_thread.start()

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        _running = False
