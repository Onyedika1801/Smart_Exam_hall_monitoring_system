"""
test_phone_detection.py
=======================
Isolation test for the Phone Detection Module.
Run this BEFORE integrating into the main pipeline.

Usage:
    python test_phone_detection.py --source 0         # webcam
    python test_phone_detection.py --source test_videos/test.mp4

IMPORTANT NOTE ON FPS:
    The full pipeline camera runs at 30 FPS but the detection module
    does NOT need to process every frame. A phone stays visible for
    several seconds — catching it at 2-5 detections/sec on CPU is
    more than sufficient for the scoring system to accumulate a red alert.
"""

import argparse
import queue
import time
import sys
import os

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.phone_detection import PhoneDetectionModule, DetectionEvent


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_event(event: DetectionEvent):
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

    try:
        config = load_config(config_path)
        print(f"✅ Config loaded from {config_path}")
    except FileNotFoundError:
        print(f"❌ config.yaml not found at {config_path}")
        return

    alert_queue = queue.Queue(maxsize=100)

    print(f"\nLoading model: {config['phone_detection']['model_path']}")
    module = PhoneDetectionModule(config, alert_queue, camera_id="cam_0")
    module.start()
    print("✅ PhoneDetectionModule started\n")

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
    recent_events = []
    total_detections = 0
    confidence_samples = []
    camera_fps_times = []

    # Send every 3rd frame to detector — keeps display smooth
    # while giving detector enough time per frame on CPU
    FRAME_SKIP = 10  # Increased from 3 — CPU-only inference needs fewer frames/sec

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (640, 480))
            frame_number += 1

            # Only send every Nth frame to detector
            if frame_number % FRAME_SKIP == 0:
                module.put_frame(frame.copy(), frame_number)

            # Collect detection events
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

            # Only draw THIS frame's detections — drawing accumulated history
            # causes stale boxes to stack up on screen (cosmetic bug only,
            # never affected the real detection/scoring pipeline)
            display_frame = module.draw_detections(frame, new_events)

            # Camera FPS
            camera_fps_times.append(time.time() - t_start)
            if len(camera_fps_times) > 30:
                camera_fps_times.pop(0)
            camera_fps = 1.0 / (sum(camera_fps_times) / len(camera_fps_times)) if camera_fps_times else 0

            stats = module.get_stats()

            lines = [
                f"Camera FPS: {camera_fps:.1f}",
                f"Module throughput FPS: {stats['fps']:.2f}",
                f"Frames sent to detector: {stats['frames_processed']}",
                f"Total phone detections: {total_detections}",
                f"Queue size: {stats['queue_size']}",
                f"Frame skip: 1 in every {FRAME_SKIP} frames",
            ]
            y = 25
            for line in lines:
                cv2.putText(display_frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
                y += 20

            if new_events:
                cv2.putText(display_frame, "DETECTION → ALERT QUEUE",
                            (10, 460), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

            cv2.imshow("Phone Detection — Isolation Test (Q to quit)", display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        module.stop()

    # Final report
    print("\n" + "=" * 60)
    print("ISOLATION TEST REPORT")
    print("=" * 60)
    final_stats = module.get_stats()
    print(f"Frames sent to module:  {final_stats['frames_processed']}")
    print(f"Total detections:       {total_detections}")
    print(f"Module throughput FPS:  {final_stats['fps']}")

    if confidence_samples:
        avg_conf = sum(confidence_samples) / len(confidence_samples)
        clustering_low = sum(1 for c in confidence_samples if 0.50 <= c < 0.65)
        clustering_pct = 100 * clustering_low / len(confidence_samples)
        print(f"\nConfidence scores:")
        print(f"  Average:  {avg_conf:.3f}")
        print(f"  Min:      {min(confidence_samples):.3f}")
        print(f"  Max:      {max(confidence_samples):.3f}")
        print(f"  Clustering 0.50-0.65: {clustering_low}/{len(confidence_samples)} ({clustering_pct:.1f}%)")

    print("\n--- PASS / FAIL CHECKLIST ---")
    checks = []

    fps_ok = final_stats['fps'] >= 1.0
    checks.append(('Module processing frames (throughput > 1 FPS)', fps_ok,
                   f"Got {final_stats['fps']} FPS"))

    checks.append(('At least one phone detected', total_detections > 0,
                   'Hold phone clearly visible to camera during test'))

    if confidence_samples:
        clustering_pct = 100 * sum(1 for c in confidence_samples if 0.50 <= c < 0.65) / len(confidence_samples)
        checks.append(('Confidence not clustering at 0.50-0.65', clustering_pct < 30,
                       f"{clustering_pct:.1f}% in low range"))

    if recent_events:
        e = recent_events[0]
        struct_ok = all([hasattr(e, attr) for attr in
                         ['module', 'candidate_id', 'weighted_score', 'requires_persistence']])
        struct_ok = struct_ok and e.requires_persistence == False
        checks.append(('DetectionEvent structure correct', struct_ok, 'Check dataclass'))

    all_pass = True
    for name, passed, note in checks:
        print(f"  {'✅ PASS' if passed else '❌ FAIL'} — {name}")
        if not passed:
            print(f"         Note: {note}")
            all_pass = False

    print("\n" + ("🟢 ALL CHECKS PASSED — Ready for pipeline integration."
                  if all_pass else "🔴 SOME CHECKS FAILED — Fix before integration."))

    print("\n--- WHY LOW FPS IS ACCEPTABLE ---")
    print("  • Phone stays visible for seconds, not single frames")
    print("  • Scoring needs sustained detection — not every frame")
    print("  • At 1 detection/sec, a 3-second phone = 3 events = 35–105 pts (red alert)")
    print("  • Chapter 3.4.1 targets 15-20 FPS SYSTEM-WIDE across all modules")
    print("  • Individual CPU module throughput is always lower — this is expected")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_isolation_test(args.source, args.config)
