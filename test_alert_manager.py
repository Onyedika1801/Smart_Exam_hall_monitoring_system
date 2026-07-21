"""
test_alert_manager.py
======================
Standalone test for the Alert Manager — Section 3.10 & 3.11.

UNLIKE the other four test scripts, this one needs NO CAMERA. All
inputs are synthetic DetectionEvent objects constructed directly in
this file. This tests the SCORING MATH and ALERT LOGIC in isolation
from whether any detection module correctly perceives a phone, a
gaze angle, a posture deviation, or a hand crossing — those are
already covered by their own isolation tests.

Usage:
    python test_alert_manager.py

Each test prints PASS/FAIL. A summary is printed at the end.
"""

import time
import os
import sqlite3

import yaml

from modules.phone_detection import DetectionEvent
from alert_manager import AlertManager


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def make_event(module: str, candidate_id: str, base_score: float,
               weighted_score: float = None, duration_seconds: float = 0.0,
               confidence: float = 0.9, timestamp: float = None) -> DetectionEvent:
    return DetectionEvent(
        module=module,
        candidate_id=candidate_id,
        behaviour_type=f"{module}_test",
        confidence=confidence,
        base_score=base_score,
        weighted_score=weighted_score if weighted_score is not None else base_score,
        bbox=(0, 0, 0, 0),
        frame_number=0,
        timestamp=timestamp if timestamp is not None else time.time(),
        duration_seconds=duration_seconds,
    )


class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def check(self, label: str, condition: bool, detail: str = ""):
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status} — {label}" + (f" ({detail})" if detail else ""))
        if condition:
            self.passed += 1
        else:
            self.failed += 1


def fresh_manager(config: dict, db_path: str) -> AlertManager:
    if os.path.exists(db_path):
        os.remove(db_path)
    import queue
    return AlertManager(config, queue.Queue(), db_path=db_path)


def test_basic_scoring(config, results: TestResults):
    print("\n--- Test 1: Basic scoring (single event, under 3s duration) ---")
    am = fresh_manager(config, "database/test_alert_manager_1.db")

    # Uses gaze_detection rather than phone_detection deliberately -- this
    # test checks the generic duration/bonus formula in isolation, and
    # phone_detection events now carry a high-confidence auto-alert
    # override (Test 11) that would otherwise interfere here.
    event = make_event("gaze_detection", "R2C3", base_score=35,
                        weighted_score=35, duration_seconds=0.0)
    result = am.process_event(event)

    # under_3s multiplier = 1.0, no combination bonus (only one module)
    expected_score = 35 * 1.0 * 1.0
    results.check(
        "Score matches base_score x 1.0 duration x 1.0 bonus",
        abs(result['score_after'] - expected_score) < 0.01,
        f"got {result['score_after']:.2f}, expected {expected_score:.2f}"
    )
    results.check("No combination bonus applied (only one module)",
                   result['combination_bonus_applied'] is False)
    results.check("No alert yet (35 < yellow_threshold 60)",
                   result['alert_level'] == "none")


def test_duration_multiplier(config, results: TestResults):
    print("\n--- Test 2: Duration multiplier tiers ---")
    am = fresh_manager(config, "database/test_alert_manager_2.db")

    cases = [
        (2.0, 1.0, "under_3s"),
        (4.0, 1.5, "3_to_6s"),
        (8.0, 2.0, "6_to_10s"),
        (12.0, 3.0, "over_10s"),
    ]
    for duration, expected_mult, label in cases:
        am2 = fresh_manager(config, f"database/test_alert_manager_2_{label}.db")
        event = make_event("gaze_detection", "R1C1", base_score=15,
                            weighted_score=15, duration_seconds=duration)
        result = am2.process_event(event)
        expected = 15 * expected_mult
        results.check(
            f"Duration {duration}s -> {label} multiplier ({expected_mult}x)",
            abs(result['contribution'] - expected) < 0.01,
            f"got {result['contribution']:.2f}, expected {expected:.2f}"
        )


def test_combination_bonus(config, results: TestResults):
    print("\n--- Test 3: Combination bonus (2 modules, same candidate, within window) ---")
    am = fresh_manager(config, "database/test_alert_manager_3.db")
    now = time.time()

    event1 = make_event("gaze_detection", "R3C2", base_score=15,
                         weighted_score=15, duration_seconds=0, timestamp=now)
    result1 = am.process_event(event1)
    results.check("First event (gaze): no bonus (nothing to corroborate yet)",
                   result1['combination_bonus_applied'] is False)

    event2 = make_event("posture_analysis", "R3C2", base_score=10,
                         weighted_score=10, duration_seconds=0, timestamp=now + 5)
    result2 = am.process_event(event2)
    results.check("Second event (posture, 5s later, same candidate): bonus applied",
                   result2['combination_bonus_applied'] is True)

    expected_contribution = 10 * 1.0 * 1.20  # base x duration_mult x combo_bonus
    results.check(
        "Bonus contribution = base x 1.0 duration x 1.20 combo",
        abs(result2['contribution'] - expected_contribution) < 0.01,
        f"got {result2['contribution']:.2f}, expected {expected_contribution:.2f}"
    )


def test_combination_bonus_different_candidates(config, results: TestResults):
    print("\n--- Test 4: No combination bonus across DIFFERENT candidates ---")
    am = fresh_manager(config, "database/test_alert_manager_4.db")
    now = time.time()

    am.process_event(make_event("gaze_detection", "R1C1", base_score=15,
                                 weighted_score=15, timestamp=now))
    result = am.process_event(make_event("posture_analysis", "R4C4", base_score=10,
                                          weighted_score=10, timestamp=now + 2))
    results.check(
        "Different candidate does NOT trigger combination bonus",
        result['combination_bonus_applied'] is False
    )


def test_combination_bonus_same_module(config, results: TestResults):
    print("\n--- Test 5: No combination bonus from the SAME module twice ---")
    am = fresh_manager(config, "database/test_alert_manager_5.db")
    now = time.time()

    am.process_event(make_event("phone_detection", "R2C2", base_score=35,
                                 weighted_score=35, timestamp=now))
    result = am.process_event(make_event("phone_detection", "R2C2", base_score=35,
                                          weighted_score=35, timestamp=now + 3))
    results.check(
        "Same module flagging twice does NOT count as corroboration",
        result['combination_bonus_applied'] is False
    )


def test_alert_thresholds(config, results: TestResults):
    print("\n--- Test 6: Yellow/Red alert thresholds ---")
    am = fresh_manager(config, "database/test_alert_manager_6.db")
    now = time.time()

    # Push score up gradually using non-phone modules -- phone_detection
    # events now carry a high-confidence auto-alert override (Test 11)
    # that would short-circuit this gradual-climb scenario.
    r1 = am.process_event(make_event("posture_analysis", "R1C1", base_score=35,
                                      weighted_score=35, timestamp=now))
    results.check("Score 35: no alert", r1['alert_level'] == "none")

    r2 = am.process_event(make_event("gaze_detection", "R1C1", base_score=30,
                                      weighted_score=30, timestamp=now + 1))
    # 35 + (30 * combo_bonus 1.2 since gaze corroborates within window) = 35 + 36 = 71
    results.check(f"Score ~71 (>= yellow 60): YELLOW alert",
                   r2['alert_level'] == "yellow",
                   f"score={r2['score_after']:.1f}")

    r3 = am.process_event(make_event("posture_analysis", "R1C1", base_score=20,
                                      weighted_score=20, timestamp=now + 2))
    results.check(f"Score should now exceed red 75: RED alert",
                   r3['alert_level'] == "red",
                   f"score={r3['score_after']:.1f}")
    results.check("Red alert was dispatched (cooldown allows first alert)",
                   r3['alert_dispatched'] is True)


def test_alert_cooldown(config, results: TestResults):
    print("\n--- Test 7: Alert cooldown prevents repeat alerts ---")
    am = fresh_manager(config, "database/test_alert_manager_7.db")
    now = time.time()

    # Force straight to red with one big event
    r1 = am.process_event(make_event("phone_detection", "R5C5", base_score=80,
                                      weighted_score=80, timestamp=now))
    results.check("First red alert dispatched", r1['alert_dispatched'] is True)

    # Immediately after — still red, but within cooldown (30s)
    r2 = am.process_event(make_event("phone_detection", "R5C5", base_score=80,
                                      weighted_score=80, timestamp=now + 2))
    results.check("Second alert within cooldown window is suppressed",
                   r2['alert_dispatched'] is False,
                   f"alert_level={r2['alert_level']}")

    # After cooldown expires
    r3 = am.process_event(make_event("phone_detection", "R5C5", base_score=80,
                                      weighted_score=80, timestamp=now + 35))
    results.check("Alert after cooldown expires is dispatched again",
                   r3['alert_dispatched'] is True)


def test_score_decay(config, results: TestResults):
    print("\n--- Test 8: Idle score decay (-10% per 10s of no new events) ---")
    am = fresh_manager(config, "database/test_alert_manager_8.db")
    now = time.time()

    am.process_event(make_event("phone_detection", "R3C3", base_score=100,
                                 weighted_score=100, timestamp=now))
    score_immediately = am.get_candidate_score("R3C3")

    # Simulate 10s of no new events by directly checking decayed score
    # (get_candidate_score applies lazy decay based on wall-clock time,
    # so we fast-forward by manipulating the state's last_updated)
    state = am._candidates["R3C3"]
    state.last_updated = now - 10  # pretend 10s have passed
    score_after_10s = am.get_candidate_score("R3C3")

    expected = score_immediately * 0.90
    results.check(
        "Score decays by ~10% after 10 idle seconds",
        abs(score_after_10s - expected) < 0.5,
        f"got {score_after_10s:.2f}, expected ~{expected:.2f}"
    )


def test_incident_logging(config, results: TestResults):
    print("\n--- Test 9: Incidents are logged to SQLite ---")
    db_path = "database/test_alert_manager_9.db"
    am = fresh_manager(config, db_path)

    am.process_event(make_event("phone_detection", "R2C2", base_score=35,
                                 weighted_score=35))

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM incidents").fetchall()
    conn.close()

    results.check("Exactly one incident row logged", len(rows) == 1,
                   f"found {len(rows)} rows")


def test_alert_logging_and_callback(config, results: TestResults):
    print("\n--- Test 10: Alert callback fires + alerts table populated ---")
    db_path = "database/test_alert_manager_10.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    callback_fired = {'count': 0, 'last': None}

    def on_alert(alert_dict):
        callback_fired['count'] += 1
        callback_fired['last'] = alert_dict

    import queue
    am = AlertManager(config, queue.Queue(), db_path=db_path, on_alert=on_alert)
    am.process_event(make_event("phone_detection", "R4C1", base_score=80,
                                 weighted_score=80))

    results.check("on_alert callback fired exactly once", callback_fired['count'] == 1)
    results.check("Callback received correct candidate_id",
                   callback_fired['last'] is not None and
                   callback_fired['last']['candidate_id'] == "R4C1")

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT * FROM alerts").fetchall()
    conn.close()
    results.check("Alert row logged to alerts table", len(rows) == 1)


def cleanup_test_dbs():
    import glob
    for f in glob.glob("database/test_alert_manager_*.db"):
        try:
            os.remove(f)
        except OSError:
            pass


def test_phone_high_confidence_override(config, results: TestResults):
    print("\n--- Test 11: High-confidence phone detection forces immediate red alert ---")
    am = fresh_manager(config, "database/test_alert_manager_11.db")

    # A single phone event at 0.85 confidence, low base weighted_score,
    # no duration, no combination bonus -- would normally contribute far
    # less than red_threshold (75) on its own.
    event = make_event("phone_detection", "R1C1", base_score=35,
                        weighted_score=35, duration_seconds=0, confidence=0.85)
    result = am.process_event(event)

    results.check("High-confidence override applied",
                   result['phone_high_confidence_override'] is True)
    results.check(f"Score forced to >= red_threshold (75) on a single event",
                   result['score_after'] >= 75,
                   f"score={result['score_after']:.1f}")
    results.check("Alert level is red", result['alert_level'] == "red")
    results.check("Alert dispatched immediately", result['alert_dispatched'] is True)

    # Confirm a LOW-confidence phone event does NOT trigger the override
    am2 = fresh_manager(config, "database/test_alert_manager_11b.db")
    low_conf_event = make_event("phone_detection", "R2C2", base_score=35,
                                 weighted_score=21, duration_seconds=0, confidence=0.60)
    result2 = am2.process_event(low_conf_event)
    results.check("Low-confidence phone event does NOT trigger override",
                   result2['phone_high_confidence_override'] is False)
    results.check("Low-confidence event alone does not reach red",
                   result2['alert_level'] != "red",
                   f"score={result2['score_after']:.1f}")


def main():
    print("=" * 60)
    print("ALERT MANAGER — STANDALONE TEST (no camera required)")
    print("=" * 60)

    config = load_config("config.yaml")
    os.makedirs("database", exist_ok=True)
    results = TestResults()

    test_basic_scoring(config, results)
    test_duration_multiplier(config, results)
    test_combination_bonus(config, results)
    test_combination_bonus_different_candidates(config, results)
    test_combination_bonus_same_module(config, results)
    test_alert_thresholds(config, results)
    test_alert_cooldown(config, results)
    test_score_decay(config, results)
    test_incident_logging(config, results)
    test_alert_logging_and_callback(config, results)
    test_phone_high_confidence_override(config, results)

    cleanup_test_dbs()

    print("\n" + "=" * 60)
    print(f"RESULTS: {results.passed} passed, {results.failed} failed")
    print("=" * 60)
    if results.failed == 0:
        print("✅ All scoring/alert logic behaves as specified in Chapter 3.")
    else:
        print("❌ Some checks failed — see above for details.")


if __name__ == "__main__":
    main()
