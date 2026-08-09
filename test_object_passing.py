"""
test_object_passing.py
=======================
Isolation test for the Object Passing Detection Module.
Run this BEFORE integrating into the main pipeline.

Usage:
    python test_object_passing.py --source 0
    python test_object_passing.py --source test_videos/test.mp4

IMPORTANT — READ BEFORE TESTING:
    This module has real, documented limitations (see the docstring at
    the top of modules/object_passing.py). In particular:

    - It reuses the phone-detection model as a placeholder object
      detector. It was NOT trained on paper/pen/answer-sheet classes,
      so it will under-detect anything that isn't phone-shaped. Expect
      weak results testing with paper — this is expected, not a bug.
    - It needs at least two people (or one person moving between two
      positions) in frame to produce a meaningful zone-crossing test,
      since it only fires on hand movement BETWEEN adjacent grid zones.
    - The grace window suppresses ALL scored events for the first few
      minutes (see config.yaml: grace_window_seconds) to avoid flagging
      attendance-sheet passing. Crossings during this window are still
      counted internally but will not appear as red boxes or terminal
      prints — check the "Suppressed (grace)" stat to confirm this is
      what's happening rather than nothing being detected at all.

NOTE ON THIS MODULE'S ARCHITECTURE (updated):
    ObjectPassingModule now self-threads exactly like phone_detection.py,
    gaze_detection.py, and posture_analysis.py — it owns its own thread
    and frame_queue, and pushes DetectionEvents directly onto a shared
    alert_queue rather than returning them from a synchronous call. This
    test script therefore constructs its own local alert_queue (used by
    nothing else, since this is an isolated single-module test) and
    drains it each loop to print/log events — this exactly mirrors how
    the real system observes this module's output once integrated via
    main.py, rather than being a special isolation-only code path.
"""

import argparse
import queue
import time
import sys
import os

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.object_passing import ObjectPassingModule
from modules.phone_detection import DetectionEvent


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_event(event: DetectionEvent):
    print(
        f"  🔁 OBJECT PASS DETECTED | "
        f"To candidate: {event.candidate_id} | "
        f"Score: {event.weighted_score:.1f}/{event.base_score} | "
        f"Frame: {event.frame_number}"
    )


def run_isolation_test(source, config_path: str = "config.yaml"):
    print("=" * 60)
    print("OBJECT PASSING MODULE — ISOLATION TEST")
    print("=" * 60)

    try:
        config = load_config(config_path)
        print(f"✅ Config loaded from {config_path}")
    except FileNotFoundError:
        print(f"❌ config.yaml not found at {config_path}")
        return

    op_cfg = config['object_passing']
    print("Active thresholds (from config.yaml):")
    print(f"  Object confidence:   {op_cfg['object_confidence_threshold']}")
    print(f"  Base score:          {op_cfg['base_score']}")
    print(f"  Frame skip:          1-in-{op_cfg['frame_skip']}")
    print(f"  Grace window:        {op_cfg['grace_window_seconds']}s")
    print(f"  Burst window:        {op_cfg['burst_detection_window_seconds']}s "
          f"(>= {op_cfg['burst_min_concurrent_crossings']} concurrent pairs = suppressed)")
    print()

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source,
                            cv2.CAP_DSHOW if str(source).isdigit() else 0)
    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        return

    # Local alert_queue — nothing else consumes it in this isolated test.
    # In the real system (main.py), this same queue is shared across all
    # four modules and drained by alert_manager instead of this script.
    alert_queue: "queue.Queue[DetectionEvent]" = queue.Queue(maxsize=100)

    module = ObjectPassingModule(config, alert_queue, camera_id="isolation_test")
    module.start()

    print("Loading YOLO + MediaPipe Hands (in background thread)...")
    print("✅ ObjectPassingModule started")
    print("\nMove a hand holding an object across a grid boundary to test.")
    print("Press Q to quit.\n")

    frame_number = 0
    fps_timer = time.time()
    fps_counter = 0
    camera_fps = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame read failed / end of video.")
                break

            frame_number += 1
            fps_counter += 1
            if time.time() - fps_timer >= 1.0:
                camera_fps = fps_counter / (time.time() - fps_timer)
                fps_counter = 0
                fps_timer = time.time()

            module.put_frame(frame, frame_number)

            # Drain any events the background thread pushed since last
            # loop iteration — this is how the real system (alert_manager)
            # would consume them too.
            while True:
                try:
                    event = alert_queue.get_nowait()
                except queue.Empty:
                    break
                print_event(event)

            # get_last_frame_events() is an isolation-test convenience —
            # lets us draw boxes for whatever the module's background
            # thread most recently processed, without needing frame-exact
            # synchronisation with the queue drain above.
            display_frame = module.draw_detections(
                frame, module.get_last_frame_events()
            )

            stats = module.get_stats()
            overlay = [
                f"Camera FPS: {camera_fps:.1f} | Module FPS: {stats['fps']}",
                f"Frames processed: {stats['frames_processed']} "
                f"(actually ran models: {stats['frames_actually_processed']}, "
                f"skipped: {stats['frames_skipped']})",
                f"Queue size: {stats['queue_size']}",
                f"Hand observations: {stats['total_hand_observations']}",
                f"Crossings detected: {stats['total_crossings_detected']}",
                f"Events emitted: {stats['events_emitted']}",
                f"Suppressed (grace): {stats['suppressed_grace_window']}",
                f"Suppressed (burst): {stats['suppressed_burst']}",
                f"Suppressed (zone sep): {stats['suppressed_zone_separation']}",
                f"Zone separation OK: {stats['zone_separation_reliable']}",
                f"In grace window: {stats['in_grace_window']}",
            ]
            for i, line in enumerate(overlay):
                cv2.putText(display_frame, line, (10, 20 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            cv2.imshow("Object Passing — Isolation Test (Q to quit)", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        module.stop()
        cap.release()
        cv2.destroyAllWindows()

    stats = module.get_stats()
    print("\n" + "=" * 60)
    print("SESSION SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nStructural checks:")
    checks = [
        ("Ran without crashing", stats['frames_processed'] > 0),
        ("Detected >= 1 hand crossing", stats['total_crossings_detected'] > 0),
        ("Grace/burst suppression logic executed",
         stats['suppressed_grace_window'] > 0 or stats['suppressed_burst'] > 0
         or stats['events_emitted'] > 0),
    ]
    for label, ok in checks:
        print(f"  {'✅' if ok else '⬜'} {label}")

    print("\nReminder: weak detection of non-phone objects (paper, pens) is")
    print("EXPECTED at this stage — see module docstring limitations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Object Passing isolation test")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()
    run_isolation_test(args.source, args.config)
