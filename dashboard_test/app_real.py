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
import csv
import io
import json
import os
import queue
import signal
import threading
import time

import cv2
import numpy as np
import yaml
from flask import Flask, Response, render_template, send_from_directory, request

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
        # Live events are always continuous with whatever session is
        # currently running -- session boundaries are only meaningful
        # when replaying /history across potentially many past sessions.
        "new_session": False,
        "session_label": None,
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
    last_stats_print = time.time()
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

            # Periodic status print — same 10s cadence as main.py's
            # _print_status(), so you're not debugging blind on any
            # module (especially object_passing) while running the
            # dashboard instead of the isolation test scripts.
            if time.time() - last_stats_print >= 10.0:
                print_status(
                    phone_module, gaze_module, posture_module,
                    object_passing_module, frame_number
                )
                last_stats_print = time.time()
    finally:
        for m in (phone_module, gaze_module, posture_module, object_passing_module):
            m.stop()
        alert_manager.stop()
        cap.release()
        print("[INFO] Pipeline shut down")


def print_status(phone_module, gaze_module, posture_module,
                  object_passing_module, frame_number):
    print("\n" + "=" * 60)
    print(f"STATUS — Frame: {frame_number}")
    print("-" * 60)
    for name, module in [
        ("phone_detection", phone_module),
        ("gaze_detection", gaze_module),
        ("posture_analysis", posture_module),
        ("object_passing", object_passing_module),
    ]:
        stats = module.get_stats()
        print(f"  {name:<18} queue={stats['queue_size']:<4} "
              f"processed={stats['frames_processed']} fps={stats.get('fps', '?')}")

    # object_passing's extra diagnostic counters — this is the detail
    # that was missing while running the full dashboard, versus
    # test_object_passing.py which already showed this.
    op_stats = object_passing_module.get_stats()
    print("-" * 60)
    print("  object_passing detail:")
    print(f"    hand_observations={op_stats['total_hand_observations']} "
          f"crossings_detected={op_stats['total_crossings_detected']} "
          f"events_emitted={op_stats['events_emitted']}")
    print(f"    suppressed: grace={op_stats['suppressed_grace_window']} "
          f"burst={op_stats['suppressed_burst']} "
          f"zone_sep={op_stats['suppressed_zone_separation']}")
    print(f"    in_grace_window={op_stats['in_grace_window']} "
          f"zone_separation_reliable={op_stats['zone_separation_reliable']}")
    print("=" * 60 + "\n")


def generate_mjpeg():
    while _running:
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
    live pipeline is already writing to.

    Also tags each event with a session boundary marker: if the gap
    since the PREVIOUS alert (chronologically) exceeds SESSION_GAP_SECONDS,
    this event is flagged as the start of a new session. This is a
    heuristic inferred from timestamp gaps, not a real session ID -- no
    DB schema change needed, and it works retroactively on alerts
    logged before this feature existed. A restart of app_real.py with
    no real gap in time (e.g. quick Ctrl+C and re-run within a minute)
    will NOT be treated as a new session; only an actual time gap is."""
    if _alert_manager is None:
        return json.dumps([]), 200, {"Content-Type": "application/json"}

    SESSION_GAP_SECONDS = 900  # 15 minutes

    rows = _alert_manager.get_recent_alerts(limit=50)
    rows = list(reversed(rows))  # oldest first, so front-end prepend logic matches live order

    events = []
    previous_timestamp = None
    for row in rows:
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

        is_new_session = (
            previous_timestamp is None
            or (timestamp - previous_timestamp) > SESSION_GAP_SECONDS
        )
        previous_timestamp = timestamp

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
            "new_session": is_new_session,
            "session_label": time.strftime("%b %d, %Y — session from %H:%M", time.localtime(timestamp)),
        })

    return json.dumps(events), 200, {"Content-Type": "application/json"}


@app.route("/snapshots/<path:filename>")
def snapshot(filename):
    return send_from_directory(SNAPSHOT_DIR, filename)


@app.route("/export/csv")
def export_csv():
    """Exports the full alert history (not just the last 50 shown in
    the dashboard) as a downloadable CSV file, for record-keeping or
    submission as evidence alongside the project. Read-only, same as
    every other route here -- this does not modify or delete anything
    in the database.

    A plain CSV cannot embed an actual image (CSV is a text format,
    not a container format) -- so instead of the photo itself, each
    row includes the snapshot's filename (matching the naming
    convention used everywhere else in this app) AND a full URL to
    it, so a reader can open the corresponding image directly, or a
    script can bulk-download every referenced snapshot from the
    snapshots/ folder. This is what actually ties a logged score back
    to visual proof of which candidate/behaviour it was, not the CSV
    row alone."""
    if _alert_manager is None:
        rows = []
    else:
        rows = _alert_manager.get_recent_alerts(limit=1_000_000)

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Timestamp", "Date/Time", "Candidate", "Behaviour", "Score",
        "Level", "Camera", "Snapshot Filename", "Snapshot URL",
    ])
    for row in rows:
        candidate_id = row["candidate_id"]
        timestamp = row["timestamp"]
        expected_name = snapshot_filename(candidate_id, timestamp)
        has_snapshot = os.path.exists(os.path.join(SNAPSHOT_DIR, expected_name))

        snapshot_name = expected_name if has_snapshot else ""
        snapshot_url = (
            request.host_url.rstrip("/") + "/snapshots/" + expected_name
            if has_snapshot else ""
        )

        writer.writerow([
            timestamp,
            time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp)),
            candidate_id,
            row["behaviour_type"] or "unknown",
            row["score"],
            row["alert_level"],
            row["camera_id"] or "cam_0",
            snapshot_name,
            snapshot_url,
        ])

    csv_data = output.getvalue()
    filename = f"exam_alert_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


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

    def _force_exit_after_delay(delay_seconds=2.0):
        """Safety net for Ctrl+C: if a clean shutdown hasn't actually
        ended the process within this window, force-kill it outright.
        This exists because relying purely on every generator loop
        (video_feed, /alerts SSE) cooperatively checking a _running
        flag turned out not to reliably return terminal control on
        Windows when a browser tab was still open at Ctrl+C time --
        likely Werkzeug's threaded dev server not unblocking an
        in-flight streaming connection's handler thread promptly
        enough. Rather than keep chasing that indefinitely, this
        guarantees Ctrl+C works every time, at the cost of skipping
        the graceful module.stop() calls if the timeout is actually
        hit (acceptable for a local dev/test tool, not a production
        service)."""
        time.sleep(delay_seconds)
        print("[INFO] Graceful shutdown timed out -- forcing exit.")
        os._exit(0)

    def handle_sigint(signum, frame):
        global _running
        print("\n[INFO] Ctrl+C received -- shutting down...")
        _running = False
        # Starts the force-exit countdown immediately; if the normal
        # app.run() -> finally block below completes first (the
        # graceful path), the process exits on its own before this
        # timer ever fires, and this thread being a daemon means it
        # doesn't block that from happening.
        threading.Thread(target=_force_exit_after_delay, daemon=True).start()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        _running = False
