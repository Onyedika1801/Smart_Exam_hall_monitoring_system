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
      logged to the terminal but will not appear as red boxes.
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
from modules.posture_analysis import CandidateZoneTracker
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
    print(f"  Boundary margin:     {op_cfg['boundary_margin_pixels']}px")
    print(f"  Grace window:        {op_cfg['grace_window_seconds']}s")
    print(f"  Burst window:        {op_cfg['burst_detection_window_seconds']}s "
          f"(>= {op_cfg['burst_min_concurrent_crossings']} concurrent pairs = suppressed)")
    print()

    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source,
                            cv2.CAP_DSHOW if str(source).isdigit() else 0)
    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or config['camera']['width']
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or config['camera']['height']

    zone_tracker = CandidateZoneTracker(frame_width, frame_height)
    module = ObjectPassingModule(config, zone_tracker, frame_width, frame_height)

    print("Loading YOLO + MediaPipe Hands...")
    print("✅ ObjectPassingModule started")
    print("\nMove a hand holding an object across a grid boundary to test.")
    print("Press Q to quit.\n")

    frame_number = 0
    fps_timer = time.time()
    fps_counter = 0
    camera_fps = 0.0
    last_events: list = []

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

        new_events = module.process_frame(frame, frame_number)
        for event in new_events:
            print_event(event)
        if new_events:
            last_events = new_events  # only draw THIS frame's events —
                                       # see gaze/phone modules' fix for
                                       # why we don't accumulate history
        else:
            last_events = []

        display_frame = module.draw_detections(frame, last_events)

        stats = module.get_stats()
        overlay = [
            f"Camera FPS: {camera_fps:.1f}",
            f"Frames seen: {stats['total_frames_seen']} "
            f"(processed: {stats['frames_actually_processed']}, "
            f"skipped: {stats['frames_skipped']})",
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
        ("Ran without crashing", stats['total_frames_seen'] > 0),
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
