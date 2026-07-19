"""
main.py
=======
Smart Exam Hall Monitoring System — Main Entry Point (Chapter 3 Section 3.13)

Wires together the full architecture that has, until now, only existed as
separately-tested pieces:

    Camera Thread
        |-- phone_detection.frame_queue    -> PhoneDetectionThread    --\\
        |-- gaze_detection.frame_queue     -> GazeDetectionThread     ---\\
        |-- posture_analysis.frame_queue   -> PostureAnalysisThread   ----+--> alert_queue
        `-- object_passing.frame_queue     -> ObjectPassingThread     --/
                                                                          |
                                                              AlertManagerThread
                                                              (scoring + logging)

Each detection module runs in its own thread with its own private frame
queue (Section 3.13). Only the alert_queue is shared across all of them.

ARCHITECTURAL NOTE — read before relying on this file:
--------------------------------------------------------
phone_detection.py, gaze_detection.py, and posture_analysis.py already
implement their own self-contained threading (start() / stop() /
put_frame() / internal _run() loop that pushes DetectionEvents directly
onto a shared alert_queue passed at construction).

object_passing.py does NOT follow that same pattern — it was built to be
called synchronously (process_frame() returns events rather than
threading itself and pushing to alert_queue internally), and takes a
zone_tracker + frame dimensions directly rather than an alert_queue.
This inconsistency was identified during main.py integration but
deliberately NOT fixed by changing object_passing.py itself, since the
person testing this project was actively mid-way through isolation-
testing that exact module and script when this was built — changing its
interface then would have invalidated testing already in progress.

Instead, ObjectPassingThreadWrapper below gives it the same external
shape (own thread, own queue, pushes to alert_queue) WITHOUT modifying
object_passing.py or test_object_passing.py at all. If you retrofit
object_passing.py later to self-thread like the other three, this
wrapper class can be deleted and construction below simplified to match
the other three modules' pattern.

CANDIDATE ID CONSISTENCY ACROSS MODULES:
------------------------------------------
phone_detection.py, gaze_detection.py, and posture_analysis.py EACH
create their own internal CandidateZoneTracker lazily on first frame
(they do not accept a shared instance). CandidateZoneTracker itself is
stateless — get_candidate_id() is a pure function of
(x, y, frame_width, frame_height, grid_cols, grid_rows) with no memory
between calls — so as long as every module receives frames of identical
dimensions (guaranteed here, since all four are fed from the same
camera read) and use the same default grid_cols=5/grid_rows=4, separate
instances still produce IDENTICAL candidate_id strings for the same
physical position. This has been verified by inspecting each module's
source, not assumed. If any module's grid defaults are ever changed,
they must be changed in ALL FOUR simultaneously or candidate_id values
will silently stop matching across modules, breaking alert_manager's
combination-bonus scoring (Section 3.10.4).

USAGE:
    python main.py --source 0
    python main.py --source test_videos/sample.mp4

Press Q (with a video window focused) or Ctrl+C in the terminal to stop.
"""

import argparse
import logging
import queue
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import yaml

from modules.phone_detection import PhoneDetectionModule, DetectionEvent
from modules.gaze_detection import GazeDetectionModule
from modules.posture_analysis import PostureAnalysisModule, CandidateZoneTracker
from modules.object_passing import ObjectPassingModule
from alert_manager import AlertManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================
# Wrapper: gives object_passing.py the same external shape as the
# other three self-threading modules, without modifying that module.
# See module docstring above for why this exists.
# ============================================================
class ObjectPassingThreadWrapper:
    def __init__(self, config: dict, alert_queue: "queue.Queue[DetectionEvent]",
                 frame_width: int, frame_height: int):
        self.alert_queue = alert_queue

        max_q = config['queues']['max_size']
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=max_q)

        # Uses the SAME default grid (grid_cols=5, grid_rows=4) as the
        # other three modules' internally-created trackers — see the
        # "Candidate ID Consistency" note in this file's module docstring.
        self._zone_tracker = CandidateZoneTracker(frame_width, frame_height)
        self._module = ObjectPassingModule(config, self._zone_tracker,
                                            frame_width, frame_height)

        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="ObjectPassingThread")
        self._stop_event = threading.Event()
        self._start_time: Optional[float] = None
        self.frames_processed = 0

    def start(self):
        self._start_time = time.time()
        self._thread.start()
        logger.info("[ObjectPassingWrapper] Thread started")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info(f"[ObjectPassingWrapper] Stopped. "
                    f"Processed {self.frames_processed} frames.")

    def put_frame(self, frame, frame_number: int):
        try:
            self.frame_queue.put_nowait((frame, frame_number))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()  # drop oldest, stay real-time
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait((frame, frame_number))
            except queue.Full:
                pass

    def get_stats(self) -> dict:
        inner = self._module.get_stats()
        elapsed = time.time() - self._start_time if self._start_time else 0
        inner['fps'] = round(self.frames_processed / elapsed, 2) if elapsed > 0 else 0
        inner['queue_size'] = self.frame_queue.qsize()
        return inner

    def _run(self):
        while not self._stop_event.is_set():
            try:
                frame, frame_number = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                events = self._module.process_frame(frame, frame_number)
                for event in events:
                    try:
                        self.alert_queue.put_nowait(event)
                    except queue.Full:
                        logger.warning("[ObjectPassingWrapper] Alert queue full — event dropped")
            except Exception as e:
                logger.error(f"[ObjectPassingWrapper] Frame {frame_number} error: {e}")
            self.frames_processed += 1


# ============================================================
# Main orchestration
# ============================================================
class ExamMonitoringSystem:
    def __init__(self, config: dict, source):
        self.config = config
        self.source = source
        self.alert_queue: "queue.Queue[DetectionEvent]" = queue.Queue(
            maxsize=config['queues']['max_size'] * 4
        )

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_width = config['camera']['width']
        self.frame_height = config['camera']['height']

        self.phone_module: Optional[PhoneDetectionModule] = None
        self.gaze_module: Optional[GazeDetectionModule] = None
        self.posture_module: Optional[PostureAnalysisModule] = None
        self.object_passing_module: Optional[ObjectPassingThreadWrapper] = None
        self.alert_manager: Optional[AlertManager] = None

        self._running = False
        self._frame_number = 0

    # --------------------------------------------------------
    def _open_camera(self):
        is_index = str(self.source).isdigit()
        src = int(self.source) if is_index else self.source
        backend = cv2.CAP_DSHOW if is_index else 0  # CAP_DSHOW helps on Windows

        self.cap = cv2.VideoCapture(src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera/video source: {self.source}")

        # Match the resolution modules were tuned against where possible
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w and actual_h:
            self.frame_width, self.frame_height = actual_w, actual_h
        logger.info(f"[Main] Camera opened — resolution {self.frame_width}x{self.frame_height}")

    def _init_modules(self):
        logger.info("[Main] Initialising detection modules...")

        self.phone_module = PhoneDetectionModule(self.config, self.alert_queue)
        self.gaze_module = GazeDetectionModule(self.config, self.alert_queue)
        self.posture_module = PostureAnalysisModule(self.config, self.alert_queue)
        self.object_passing_module = ObjectPassingThreadWrapper(
            self.config, self.alert_queue, self.frame_width, self.frame_height
        )

        def on_alert(alert_dict: dict):
            # Placeholder hook — the Flask/SSE dashboard will subscribe
            # here once built. For now, surfaced to the terminal so the
            # system is observable without a dashboard.
            logger.info(
                f"🚨 {alert_dict['level'].upper()} ALERT — "
                f"{alert_dict['candidate_id']} "
                f"(score: {alert_dict['score']:.1f})"
            )

        self.alert_manager = AlertManager(
            self.config, self.alert_queue, on_alert=on_alert
        )

    def _start_all(self):
        logger.info("[Main] Starting all module threads...")
        self.phone_module.start()
        self.gaze_module.start()
        self.posture_module.start()
        self.object_passing_module.start()
        self.alert_manager.start()
        # Small delay to let model-loading finish before the camera loop
        # starts hammering queues — avoids an initial backlog while
        # YOLO/MediaPipe are still loading in each thread.
        time.sleep(2.0)
        logger.info("[Main] All threads started.")

    def _stop_all(self):
        logger.info("[Main] Stopping all threads...")
        # Stop the camera loop first (implicit via self._running = False),
        # then detection modules, then alert_manager last so it can drain
        # any events already in the queue before shutting down.
        for name, module in [
            ("phone_detection", self.phone_module),
            ("gaze_detection", self.gaze_module),
            ("posture_analysis", self.posture_module),
            ("object_passing", self.object_passing_module),
        ]:
            try:
                module.stop()
            except Exception as e:
                logger.error(f"[Main] Error stopping {name}: {e}")

        try:
            self.alert_manager.stop()
        except Exception as e:
            logger.error(f"[Main] Error stopping alert_manager: {e}")

        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("[Main] Shutdown complete.")

    # --------------------------------------------------------
    def run(self):
        self._open_camera()
        self._init_modules()
        self._start_all()

        self._running = True

        def handle_sigint(sig, frame):
            logger.info("[Main] Ctrl+C received — shutting down...")
            self._running = False

        signal.signal(signal.SIGINT, handle_sigint)

        fps_timer = time.time()
        fps_counter = 0
        camera_fps = 0.0
        last_stats_print = time.time()

        try:
            while self._running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("[Main] Frame read failed / end of video.")
                    break

                self._frame_number += 1
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    camera_fps = fps_counter / (time.time() - fps_timer)
                    fps_counter = 0
                    fps_timer = time.time()

                # Distribute this frame to all four modules' own queues.
                # Each module applies its OWN frame_skip internally
                # (Section 3.13) — main.py does not skip on their behalf.
                self.phone_module.put_frame(frame, self._frame_number)
                self.gaze_module.put_frame(frame, self._frame_number)
                self.posture_module.put_frame(frame, self._frame_number)
                self.object_passing_module.put_frame(frame, self._frame_number)

                # Lightweight status window — NOT the real dashboard.
                # Useful for confirming the pipeline is alive without
                # needing the Flask dashboard built yet.
                display = frame.copy()
                cv2.putText(display, f"Camera FPS: {camera_fps:.1f}", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display, f"Frame: {self._frame_number}", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Exam Monitoring — Integrated Run (Q to quit)", display)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False

                # Periodic terminal summary — candidate scores, queue
                # health, per-module throughput. Stands in for the
                # dashboard until it exists.
                if time.time() - last_stats_print >= 10.0:
                    self._print_status(camera_fps)
                    last_stats_print = time.time()

        finally:
            self._stop_all()

    # --------------------------------------------------------
    def _print_status(self, camera_fps: float):
        print("\n" + "=" * 60)
        print(f"STATUS — Camera FPS: {camera_fps:.1f} | Frame: {self._frame_number}")
        print("-" * 60)
        for name, module in [
            ("phone_detection", self.phone_module),
            ("gaze_detection", self.gaze_module),
            ("posture_analysis", self.posture_module),
            ("object_passing", self.object_passing_module),
        ]:
            stats = module.get_stats()
            print(f"  {name:<18} queue={stats.get('queue_size', '?'):<4} "
                  f"processed={stats.get('frames_processed', stats.get('frames_actually_processed', '?'))}")

        candidates = self.alert_manager.get_all_candidate_states()
        if candidates:
            print("-" * 60)
            print("  Candidate scores:")
            for cid, state in sorted(candidates.items()):
                marker = {"red": "🔴", "yellow": "🟡", "none": "⚪"}.get(
                    state['alert_level'], "⚪"
                )
                print(f"    {marker} {cid}: {state['score']:.1f} ({state['alert_level']})")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Smart Exam Hall Monitoring — Integrated Run")
    parser.add_argument("--source", default="0", help="Camera index or video file path")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    print("=" * 60)
    print("SMART EXAM HALL MONITORING SYSTEM — INTEGRATED RUN")
    print("=" * 60)
    print("This runs all four detection modules + alert_manager together")
    print("for the first time. Expect this to reveal issues that isolated")
    print("module testing could not — CPU load under concurrency, thread")
    print("timing, and shared-resource behaviour in particular.")
    print("Press Q (video window focused) or Ctrl+C (terminal) to stop.")
    print("=" * 60 + "\n")

    config = load_config(args.config)
    system = ExamMonitoringSystem(config, args.source)
    system.run()


if __name__ == "__main__":
    main()
