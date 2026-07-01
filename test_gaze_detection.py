"""
test_gaze_detection.py
======================
Isolation test for the Gaze Detection Module.
Completely independent of phone_detection.py.

Usage:
    python test_gaze_detection.py --source 0
    python test_gaze_detection.py --source test_videos/test.mp4

What to test:
    1. Sit normally → should show GREEN box, no alerts
    2. Turn head left/right beyond 30° → should turn ORANGE then RED after 3s
    3. Look down sharply → should trigger after 3s
    4. Brief glance (under 3s) → should NOT trigger a sustained alert
    5. Multiple people in frame → each should have independent state
"""

import argparse
import queue
import time
import sys
import os

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.gaze_detection import GazeDetectionModule
from modules.phone_detection import DetectionEvent


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_event(event: DetectionEvent):
    duration_str = f"{event.duration_seconds:.1f}s" if event.duration_seconds else "N/A"
    print(
        f"  👁  GAZE EVENT | "
        f"Candidate: {event.candidate_id} | "
        f"Type: {event.behaviour_type} | "
        f"Score: {event.weighted_score:.0f} pts | "
        f"Duration: {duration_str} | "
        f"Frame: {event.frame_number}"
    )


def run_isolation_test(source, config_path="config.yaml"):
    print("=" * 60)
    print("GAZE DETECTION MODULE — ISOLATION TEST")
    print("=" * 60)

    try:
        config = load_config(config_path)
        print(f"✅ Config loaded from {config_path}")
    except FileNotFoundError:
        print(f"❌ config.yaml not found. Run from exam_monitor/ root.")
        return

    # Print active thresholds so you can verify
    gaze_cfg = config['gaze_detection']
    print(f"\nActive thresholds (from config.yaml):")
    print(f"  Yaw threshold:   ±{gaze_cfg['yaw_threshold']}°")
    print(f"  Pitch down:       {gaze_cfg['pitch_down_threshold']}°")
    print(f"  Pitch up:        +{gaze_cfg['pitch_up_threshold']}°")
    print(f"  Persistence:      {gaze_cfg['persistence_seconds']}s")
    print(f"  Score (brief):    {gaze_cfg['base_score_brief']} pts")
    print(f"  Score (sustained):{gaze_cfg['base_score_sustained']} pts")

    alert_queue = queue.Queue(maxsize=100)

    print(f"\nLoading MediaPipe Face Mesh...")
    module = GazeDetectionModule(config, alert_queue, camera_id="cam_0")
    module.start()
    time.sleep(2)  # Give MediaPipe time to initialise
    print("✅ GazeDetectionModule started\n")

    source_int = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(source_int)

    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        module.stop()
        return

    print(f"✅ Video source opened: {source}")
    print("\nTest guide:")
    print("  GREEN box  = normal orientation")
    print("  ORANGE box = deviation detected (timer running)")
    print("  RED box    = alert triggered (sent to alert queue)")
    print("\nRunning... Press Q to quit\n")

    frame_number = 0
    total_events = 0
    event_log = []
    camera_fps_times = []
    FRAME_SKIP = 3

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame = cv2.resize(frame, (640, 480))
            frame_number += 1

            if frame_number % FRAME_SKIP == 0:
                module.put_frame(frame.copy(), frame_number)

            # Collect events
            while True:
                try:
                    event = alert_queue.get_nowait()
                    total_events += 1
                    event_log.append(event)
                    print_event(event)
                except queue.Empty:
                    break

            # Draw detections
            display_frame = module.draw_detections(frame)

            # FPS
            camera_fps_times.append(time.time() - t_start)
            if len(camera_fps_times) > 30:
                camera_fps_times.pop(0)
            camera_fps = 1.0 / (sum(camera_fps_times) / len(camera_fps_times)) \
                         if camera_fps_times else 0

            stats = module.get_stats()

            # Overlay stats
            lines = [
                f"Camera FPS: {camera_fps:.1f}",
                f"Module FPS: {stats['fps']:.2f}",
                f"Tracked candidates: {stats['tracked_candidates']}",
                f"Total gaze events: {total_events}",
                f"Yaw threshold: ±{gaze_cfg['yaw_threshold']}°",
                f"Pitch: {gaze_cfg['pitch_down_threshold']}° to "
                f"+{gaze_cfg['pitch_up_threshold']}°",
                f"Persistence: {gaze_cfg['persistence_seconds']}s",
            ]
            y = 25
            for line in lines:
                cv2.putText(display_frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y += 18

            # Legend
            cv2.putText(display_frame, "GREEN=normal  ORANGE=deviating  RED=alert",
                        (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            cv2.imshow("Gaze Detection — Isolation Test (Q to quit)", display_frame)

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
    print(f"Frames processed:     {final_stats['frames_processed']}")
    print(f"Total gaze events:    {total_events}")
    print(f"Module FPS:           {final_stats['fps']}")
    print(f"Candidates tracked:   {final_stats['tracked_candidates']}")

    if event_log:
        print(f"\nEvent breakdown:")
        lateral = sum(1 for e in event_log if 'lateral' in e.behaviour_type)
        down    = sum(1 for e in event_log if 'down' in e.behaviour_type)
        up      = sum(1 for e in event_log if 'up' in e.behaviour_type)
        print(f"  Lateral (neighbour):  {lateral}")
        print(f"  Downward (notes):     {down}")
        print(f"  Upward (signalling):  {up}")

        scores = [e.weighted_score for e in event_log]
        print(f"\nScores:")
        print(f"  Brief events (8 pts):     "
              f"{sum(1 for e in event_log if e.weighted_score == 8)}")
        print(f"  Sustained events (15 pts):"
              f"{sum(1 for e in event_log if e.weighted_score == 15)}")

    print("\n--- PASS / FAIL CHECKLIST ---")
    checks = []

    checks.append(('Module processed frames', final_stats['frames_processed'] > 0,
                   'Module may not have started correctly'))

    checks.append(('At least one face tracked',
                   final_stats['tracked_candidates'] > 0,
                   'Sit in front of camera — face must be visible'))

    if event_log:
        # Check persistence is working — events should have duration >= 3s
        sustained = [e for e in event_log if e.duration_seconds >= 3.0]
        checks.append(('Persistence counter working (events have duration >= 3s)',
                       len(sustained) > 0,
                       'Turn head and hold for 3+ seconds to trigger'))

        # Check event structure
        e = event_log[0]
        struct_ok = (
            e.module == "gaze_detection" and
            e.requires_persistence == True and
            hasattr(e, 'duration_seconds')
        )
        checks.append(('DetectionEvent structure correct', struct_ok,
                       'Check module name and requires_persistence flag'))

        checks.append(('Score values correct (8 or 15 pts)',
                       all(e.weighted_score in [8, 15] for e in event_log),
                       'Check base_score_brief and base_score_sustained in config'))
    else:
        checks.append(('Gaze events generated',
                       False,
                       'Turn head beyond threshold and hold for 3+ seconds'))

    all_pass = True
    for name, passed, note in checks:
        print(f"  {'✅ PASS' if passed else '❌ FAIL'} — {name}")
        if not passed:
            print(f"         Note: {note}")
            all_pass = False

    print("\n" + ("🟢 ALL CHECKS PASSED — Gaze module ready for pipeline integration."
                  if all_pass else
                  "🔴 SOME CHECKS FAILED — Review notes above."))

    print("\n--- TUNING GUIDE ---")
    print("If too many false alerts (normal behaviour being flagged):")
    print("  → Increase yaw_threshold from 30 to 35 in config.yaml")
    print("  → Increase persistence_seconds from 3.0 to 4.0")
    print("If missing real deviations:")
    print("  → Decrease yaw_threshold from 30 to 25")
    print("  → Decrease persistence_seconds from 3.0 to 2.0")
    print("All changes go in config.yaml only — never in the module code")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_isolation_test(args.source, args.config)
