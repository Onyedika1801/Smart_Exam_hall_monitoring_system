"""
test_posture_analysis.py
========================
Isolation test for the Posture Analysis Module.
Completely independent of all other modules.

Usage:
    python test_posture_analysis.py --source 0
    python test_posture_analysis.py --source test_videos/test.mp4

What to test:
    PHASE 1 — Calibration (first 60 seconds):
        Sit normally and stay still — let the system record your baseline
        Watch the BLUE box and calibration percentage

    PHASE 2 — Detection (after 60 seconds):
        1. Sit normally        → GREEN box, no alerts
        2. Lean left/right     → ORANGE box, timer starts
        3. Hold lean for 3s+   → RED box, event printed
        4. Lean forward        → triggers forward_lean after 3s
        5. Return to normal    → resets immediately
"""

import argparse
import queue
import time
import sys
import os

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.posture_analysis import PostureAnalysisModule
from modules.phone_detection import DetectionEvent


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_event(event: DetectionEvent):
    print(
        f"  🧍 POSTURE EVENT | "
        f"Candidate: {event.candidate_id} | "
        f"Type: {event.behaviour_type} | "
        f"Score: {event.weighted_score:.0f} pts | "
        f"Duration: {event.duration_seconds:.1f}s | "
        f"Frame: {event.frame_number}"
    )


def run_isolation_test(source, config_path="config.yaml"):
    print("=" * 60)
    print("POSTURE ANALYSIS MODULE — ISOLATION TEST")
    print("=" * 60)

    try:
        config = load_config(config_path)
        print(f"✅ Config loaded from {config_path}")
    except FileNotFoundError:
        print(f"❌ config.yaml not found. Run from exam_monitor/ root.")
        return

    posture_cfg = config['posture_analysis']
    print(f"\nActive settings (from config.yaml):")
    print(f"  Calibration window: {posture_cfg['calibration_window_seconds']}s")
    print(f"  Std dev multiplier: {posture_cfg['deviation_std_multiplier']}")
    print(f"  Persistence:        {posture_cfg['persistence_seconds']}s")
    print(f"  Score (brief):      {posture_cfg['base_score_brief']} pts")
    print(f"  Score (sustained):  {posture_cfg['base_score_sustained']} pts")

    alert_queue = queue.Queue(maxsize=100)

    print(f"\nLoading MediaPipe Pose...")
    module = PostureAnalysisModule(config, alert_queue, camera_id="cam_0")
    module.start()
    time.sleep(2)
    print("✅ PostureAnalysisModule started\n")

    source_int = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(source_int)

    if not cap.isOpened():
        print(f"❌ Could not open source: {source}")
        module.stop()
        return

    print(f"✅ Video source opened: {source}")
    print(f"\nTest phases:")
    print(f"  PHASE 1 (0-{posture_cfg['calibration_window_seconds']}s): "
          f"Sit normally — BLUE box = calibrating")
    print(f"  PHASE 2 (after {posture_cfg['calibration_window_seconds']}s): "
          f"Test deviations — GREEN/ORANGE/RED")
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

            display_frame = module.draw_detections(frame)

            # FPS
            camera_fps_times.append(time.time() - t_start)
            if len(camera_fps_times) > 30:
                camera_fps_times.pop(0)
            camera_fps = 1.0 / (sum(camera_fps_times) / len(camera_fps_times)) \
                         if camera_fps_times else 0

            stats = module.get_stats()

            lines = [
                f"Camera FPS: {camera_fps:.1f}",
                f"Module FPS: {stats['fps']:.2f}",
                f"Tracked: {stats['tracked_candidates']} | "
                f"Calibrated: {stats['calibrated_candidates']} | "
                f"Calibrating: {stats['calibrating_candidates']}",
                f"Total posture events: {total_events}",
            ]
            y = 25
            for line in lines:
                cv2.putText(display_frame, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y += 18

            cv2.putText(display_frame,
                        "BLUE=calibrating  GREEN=normal  ORANGE=deviating  RED=alert",
                        (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

            cv2.imshow("Posture Analysis — Isolation Test (Q to quit)", display_frame)

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
    print(f"Frames processed:      {final_stats['frames_processed']}")
    print(f"Total posture events:  {total_events}")
    print(f"Module FPS:            {final_stats['fps']}")
    print(f"Candidates tracked:    {final_stats['tracked_candidates']}")
    print(f"Calibrated:            {final_stats['calibrated_candidates']}")

    if event_log:
        print(f"\nEvent breakdown:")
        asymmetry = sum(1 for e in event_log if 'asymmetry' in e.behaviour_type)
        lean      = sum(1 for e in event_log if 'lean' in e.behaviour_type)
        print(f"  Shoulder asymmetry: {asymmetry}")
        print(f"  Forward lean:       {lean}")
        print(f"\nScores:")
        print(f"  Brief (5 pts):     {sum(1 for e in event_log if e.weighted_score == 5)}")
        print(f"  Sustained (10 pts):{sum(1 for e in event_log if e.weighted_score == 10)}")

    print("\n--- PASS / FAIL CHECKLIST ---")
    checks = []

    checks.append(('Module processed frames',
                   final_stats['frames_processed'] > 0,
                   'Module may not have started correctly'))

    checks.append(('At least one candidate tracked',
                   final_stats['tracked_candidates'] > 0,
                   'Stand/sit in front of camera — body must be visible'))

    checks.append(('Calibration completed for at least one candidate',
                   final_stats['calibrated_candidates'] > 0,
                   f"Stay in frame for {posture_cfg['calibration_window_seconds']}s"))

    if event_log:
        sustained = [e for e in event_log if e.duration_seconds >= 3.0]
        checks.append(('Persistence counter working (duration >= 3s)',
                       len(sustained) > 0,
                       'Lean and hold for 3+ seconds after calibration'))

        e = event_log[0]
        struct_ok = (
            e.module == "posture_analysis" and
            e.requires_persistence == True and
            e.weighted_score in [5, 10]
        )
        checks.append(('DetectionEvent structure correct', struct_ok,
                       'Check module name, requires_persistence, score values'))
    else:
        checks.append(('Posture events generated', False,
                       'Wait for calibration, then lean sideways and hold 3s'))

    all_pass = True
    for name, passed, note in checks:
        print(f"  {'✅ PASS' if passed else '❌ FAIL'} — {name}")
        if not passed:
            print(f"         Note: {note}")
            all_pass = False

    print("\n" + ("🟢 ALL CHECKS PASSED — Posture module ready for integration."
                  if all_pass else
                  "🔴 SOME CHECKS FAILED — Review notes above."))

    print("\n--- TUNING GUIDE ---")
    print("Too many false alerts after calibration:")
    print("  → Increase deviation_std_multiplier from 2.0 to 2.5 in config.yaml")
    print("Missing real deviations:")
    print("  → Decrease deviation_std_multiplier from 2.0 to 1.5")
    print("Calibration too short (not enough baseline):")
    print("  → Increase calibration_window_seconds from 60 to 90")
    print("All changes in config.yaml only — never in module code")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_isolation_test(args.source, args.config)
