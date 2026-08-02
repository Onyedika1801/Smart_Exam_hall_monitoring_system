"""
Step 1: Flask skeleton + live MJPEG video streaming.

Purpose: prove the dashboard can display a live webcam feed before we
add SSE alerts, seat map, or wire in the real detection modules.

Run:
    python app.py --source 0        (or 1 for external webcam)

Then open http://localhost:5000 in a browser.
"""

import argparse
import json
import queue
import random
import threading
import time

import cv2
from flask import Flask, Response, render_template

app = Flask(__name__)

# Shared state between the camera-reading thread and the Flask response
# generator. A lock protects the frame buffer since two different
# threads touch it (camera thread writes, Flask generator reads).
_frame_lock = threading.Lock()
_latest_frame = None
_camera_source = 0
_running = True

# --- Fake alert generation (dummy data only, for testing the SSE pipe) ---
# This stands in for your real alert_manager.py output. Each SSE client
# gets its own queue so multiple browser tabs can each receive every event.
_sse_clients = []
_sse_clients_lock = threading.Lock()

BEHAVIOUR_TYPES = ["Phone Detected", "Gaze Deviation", "Posture Deviation", "Object Passing"]
GRID_COLS = 5
GRID_ROWS = 4


def fake_alert_generator():
    """Pushes a random fake alert event every few seconds to all connected clients."""
    while _running:
        time.sleep(random.uniform(3, 6))

        col = random.randint(1, GRID_COLS)
        row = random.randint(1, GRID_ROWS)
        score = random.randint(40, 95)
        level = "red" if score >= 75 else "yellow" if score >= 60 else "none"

        event = {
            "candidate_id": f"R{row}C{col}",
            "row": row,
            "col": col,
            "behaviour": random.choice(BEHAVIOUR_TYPES),
            "score": score,
            "level": level,
            "camera_id": 1,
            "timestamp": time.strftime("%H:%M:%S"),
        }

        with _sse_clients_lock:
            for client_queue in _sse_clients:
                client_queue.put(event)


def camera_loop():
    """Continuously read frames from the webcam in a background thread."""
    global _latest_frame

    # On Windows, the default MSMF backend can fail to grab frames from some
    # external/USB webcams (works for opening the device, then fails on read).
    # DirectShow (CAP_DSHOW) tends to be more reliable for these. This flag is
    # ignored harmlessly on non-Windows platforms.
    cap = cv2.VideoCapture(_camera_source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source {_camera_source}")
        return

    print(f"[INFO] Camera source {_camera_source} opened successfully")

    while _running:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to read frame, retrying...")
            time.sleep(0.1)
            continue

        # Encode as JPEG for MJPEG streaming
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        with _frame_lock:
            _latest_frame = buffer.tobytes()

    cap.release()


def generate_mjpeg():
    """Generator that yields frames in the multipart MJPEG format Flask streams."""
    while True:
        with _frame_lock:
            frame = _latest_frame

        if frame is None:
            time.sleep(0.05)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )

        # Cap streaming rate slightly to avoid hammering the browser
        # faster than needed for a first test (~20 fps ceiling)
        time.sleep(0.05)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/alerts")
def alerts():
    """SSE endpoint. Each connecting browser gets its own queue and receives
    every fake alert event as it's generated, formatted as an SSE 'data:' line."""

    def stream():
        client_queue = queue.Queue()
        with _sse_clients_lock:
            _sse_clients.append(client_queue)

        try:
            while True:
                event = client_queue.get()  # blocks until next event
                yield f"data: {json.dumps(event)}\n\n"
        finally:
            with _sse_clients_lock:
                _sse_clients.remove(client_queue)

    return Response(stream(), mimetype="text/event-stream")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=int, default=0, help="Webcam index (0 or 1)"
    )
    args = parser.parse_args()
    _camera_source = args.source

    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()

    alert_thread = threading.Thread(target=fake_alert_generator, daemon=True)
    alert_thread.start()

    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        _running = False
