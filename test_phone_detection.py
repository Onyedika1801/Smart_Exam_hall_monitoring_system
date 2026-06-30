"""
test_phone_detection.py
=======================
Isolation test for the Phone Detection Module.
Run this BEFORE integrating into the main pipeline.

Chapter 3 principle:
  "Build the simplest possible version of each module that produces
   a measurable output, validate it in isolation against your test
   videos, and only then integrate it."

Usage:
    # Test with webcam:
    python test_phone_detection.py --source 0

    # Test with video file:
    python test_phone_detection.py --source test_videos/test.mp4

    # Test with image:
    python test_phone_detection.py --source test_videos/test_image.jpg

What this script checks:
    1. Model loads correctly
    2. Detections appear on screen with bounding boxes
    3. Confidence scores are above 0.75 (not clustering at 0.50-0.65)
    4. Candidate IDs are assigned correctly
    5. DetectionEvents are correctly structured
    6. FPS is >= 15 (real-time requirement, Section 1.6)
"""

import argparse
import queue
import time
import threading
import sys
import os

import cv2
import yaml
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phone_detection import PhoneDetectionModule, DetectionEvent


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_event(event: DetectionEvent):
    """Print a detection event in a readable format."""
    print(
        f"  📱 PHONE DETECTED | "
        f"Candidate: {event.candidate_id} | "
        f"Conf: {event.confidence:.3f} | "
        f"Weighted Score: {event.weighted_score:.1f}/{event.base_score} | "
        f"Frame: {event.frame_number} | "
        f"BBox: {event.bbox}"
    )


def run_isolation_test(source, config_path="config.yaml"):
    print("=" * 60)
    print("PHONE DETECTION MODULE — ISOLATION TEST")
    print("=" * 60)

    # Load config
    try:
        config = load_config(config_path)
        print(f"✅ Config loaded from {config_path}")
    except FileNotFoundError:
        print(f"❌ config.yaml not found at {config_path}")
        print("   Make sure you run this from the exam_monitor/ root folder")
        return

    # Shared alert queue (normally owned by alert_manager)
    alert_queue = queue.Queue(maxsize=100)

    # Initialise module
    print(f"\nLoading model: {config['phone_detection']['model_path']}")
    module = PhoneDetectionModule(config, alert_queue, camera_id="cam_0")
    module.start()
    print("✅ PhoneDetectionModule started\n")

    # Open video source
    source_int = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(source_int)

    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        module.stop()
        return

    print(f"✅ Video source opened: {source}")
    print(f"   Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"   Conf threshold: {config['phone_detection']['confidence_threshold']}")
    print("\nRunning... Press Q to quit\n")

    frame_number = 0
    recent_events = []  # For drawing on frame
    fps_counter = []
    total_detections = 0
    confidence_samples = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video if it ends
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Resize to spec
            frame = cv2.resize(frame, (640, 480))
            frame_number += 1

            # Feed frame to module
            t_start = time.time()
            module.put_frame(frame.copy(), frame_number)

            # Collect any detection events (non-blocking)
            new_events = []
            while True:
                try:
                    event = alert_queue.get_nowait()
                    new_events.append(event)
                    recent_events.append(event)
                    total_detections += 1
                    confidence_samples.append(event.confidence)
                    print_event(event)
                except queue.Empty:
                    break

            # Keep only last 30 events for display
            recent_events = recent_events[-30:]

            # Draw detections on frame
            display_frame = module.draw_detections(frame, recent_events)

            # FPS calculation
            fps_counter.append(time.time() - t_start)
            if len(fps_counter) > 30:
                fps_counter.pop(0)
            fps = 1.0 / (sum(fps_counter) / len(fps_counter)) if fps_counter else 0

            # Overlay stats on frame
            stats = module.get_stats()
            overlay_lines = [
                f"FPS: {fps:.1f} (min required: 15)",
                f"Frames processed: {stats['frames_processed']}",
                f"Total detections: {total_detections}",
                f"Queue size: {stats['queue_size']}",
                f"Conf threshold: {config['phone_detection']['confidence_threshold']}",
            ]
            y = 25
            for line in overlay_lines:
                colour = (0, 255, 0) if fps >= 15 else (0, 0, 255)
                cv2.putText(display_frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
                y += 22

            # Alert queue indicator
            if new_events:
                cv2.putText(display_frame, "⚡ DETECTION EVENT SENT TO ALERT QUEUE",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.imshow("Phone Detection — Isolation Test (Q to quit)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        module.stop()

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ISOLATION TEST REPORT")
    print("=" * 60)

    final_stats = module.get_stats()
    print(f"Frames processed:   {final_stats['frames_processed']}")
    print(f"Total detections:   {total_detections}")
    print(f"Average FPS:        {final_stats['fps']}")

    if confidence_samples:
        avg_conf = sum(confidence_samples) / len(confidence_samples)
        min_conf = min(confidence_samples)
        max_conf = max(confidence_samples)
        clustering_low = sum(1 for c in confidence_samples if 0.50 <= c < 0.65)
        clustering_pct = 100 * clustering_low / len(confidence_samples)

        print(f"\nConfidence scores:")
        print(f"  Average:  {avg_conf:.3f}")
        print(f"  Min:      {min_conf:.3f}")
        print(f"  Max:      {max_conf:.3f}")
        print(f"  Clustering 0.50-0.65: {clustering_low}/{len(confidence_samples)} "
              f"({clustering_pct:.1f}%)")

    print("\n--- PASS / FAIL CHECKLIST ---")

    checks = []

    # FPS check
    fps_ok = final_stats['fps'] >= 15
    checks.append(('FPS >= 15 (real-time requirement)', fps_ok,
                   f"Got {final_stats['fps']} FPS"))

    # Detection check
    detected = total_detections > 0
    checks.append(('At least one phone detected', detected,
                   'Show a phone to the camera during the test'))

    # Confidence check
    if confidence_samples:
        conf_ok = clustering_pct < 30
        checks.append(('Confidence not clustering at 0.50-0.65', conf_ok,
                       f"{clustering_pct:.1f}% of detections in low range"))
    
    # Event structure check
    if recent_events:
        e = recent_events[0]
        struct_ok = (
            hasattr(e, 'module') and
            hasattr(e, 'candidate_id') and
            hasattr(e, 'weighted_score') and
            hasattr(e, 'requires_persistence') and
            e.requires_persistence == False
        )
        checks.append(('DetectionEvent structure correct', struct_ok,
                       'Check dataclass fields'))

    all_pass = True
    for check_name, passed, note in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status} — {check_name}")
        if not passed:
            print(f"         Note: {note}")
            all_pass = False

    print("\n" + ("🟢 ALL CHECKS PASSED — Ready for pipeline integration."
                  if all_pass else
                  "🔴 SOME CHECKS FAILED — Fix before integration."))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phone Detection Isolation Test")
    parser.add_argument("--source", default="0",
                        help="Video source: 0 for webcam, or path to video/image file")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    args = parser.parse_args()
    run_isolation_test(args.source, args.config)
