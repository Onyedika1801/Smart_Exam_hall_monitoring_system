"""
alert_manager.py
=================
Alert Manager — Chapter 3 Section 3.10 & 3.11

Consumes DetectionEvent objects (from phone_detection, gaze_detection,
posture_analysis, object_passing — via the shared alert_queue) and turns
them into per-candidate suspicion scores, alert levels, and logged
incidents. This is the ONLY place scoring logic lives — no detection
module computes or stores a candidate's cumulative score itself
(Section 3.13 architectural rule).

SCORING FORMULA (Section 3.10):
    contribution = event.weighted_score
                   × duration_multiplier(event.duration_seconds)
                   × combination_bonus (1.20 if another module flagged
                     the same candidate within the last 30s, else 1.0)

    candidate.score = candidate.score × idle_decay + contribution

Where idle_decay reduces the running score by 10% for every 10 full
seconds since the candidate's score was last updated (Section 3.10.5),
applied lazily whenever the score is touched (no background timer
needed) — this is mathematically equivalent to a periodic decay tick
without the overhead of running one.

Events older than score_window_seconds no longer count toward
combination-bonus detection (Section 3.10.4) — the per-candidate event
log is pruned on every update.

ALERT THRESHOLDS (Chapter 3 Table 3.8):
    score >= red_threshold (75)    -> RED alert
    score >= yellow_threshold (60) -> YELLOW alert
    else                            -> no alert

Cooldowns (Section 3.11) prevent repeat alerts of the same level for
the same candidate from flooding the dashboard/log.

THIS MODULE DOES NOT REQUIRE A CAMERA TO TEST. All inputs are
DetectionEvent objects — see test_alert_manager.py, which feeds
synthetic events and asserts the scoring math behaves as specified,
independent of any live detection module.
"""

import time
import sqlite3
import logging
import threading
import queue
import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable

from modules.phone_detection import DetectionEvent

logger = logging.getLogger(__name__)


# ============================================================
# Per-candidate running state
# ============================================================

@dataclass
class CandidateScoreState:
    candidate_id: str
    score: float = 0.0
    last_updated: float = field(default_factory=time.time)

    # (timestamp, module) pairs — used for combination-bonus detection
    # and pruned to score_window_seconds on every update
    recent_events: List[tuple] = field(default_factory=list)

    last_red_alert_time: Optional[float] = None
    last_yellow_alert_time: Optional[float] = None
    current_alert_level: str = "none"  # "none" | "yellow" | "red"


# ============================================================
# Alert Manager
# ============================================================

class AlertManager:
    def __init__(self, config: dict, alert_queue: "queue.Queue[DetectionEvent]",
                 db_path: Optional[str] = None,
                 on_alert: Optional[Callable[[dict], None]] = None):
        """
        alert_queue: shared queue that every detection module puts
                     DetectionEvent objects onto (Section 3.13 architecture)
        on_alert:    optional callback fired whenever a new alert is
                     raised — the Flask dashboard will hook in here via
                     SSE once it exists. Not required for this module
                     to function or be tested standalone.
        """
        scoring_cfg = config['scoring']
        alerts_cfg = config['alerts']

        self._duration_multipliers = scoring_cfg['duration_multipliers']
        self._combination_bonus = scoring_cfg['combination_bonus']
        self._combination_window = scoring_cfg['combination_window_seconds']
        self._score_window = scoring_cfg['score_window_seconds']
        self._decay_percent_per_10s = scoring_cfg['decay_percent_per_10s']
        self._yellow_threshold = scoring_cfg['yellow_threshold']
        self._red_threshold = scoring_cfg['red_threshold']
        self._max_score = scoring_cfg.get('max_score', 200)

        # High-confidence phone override: a candidate confidently and
        # visibly holding a phone in the open is judged a strong enough
        # signal on its own to warrant an immediate red alert, without
        # waiting for repeated detections or a corroborating signal from
        # another module. Rationale: phone possession is already treated
        # as an unambiguous violation (Section 3.7.3) with no persistence
        # requirement; this extends that reasoning to scoring, since a
        # student risking open phone use in front of a working camera is
        # a rarer and more deliberate act than a brief, low-confidence
        # glimpse of a phone-like object.
        phone_cfg = config.get('phone_detection', {})
        self._phone_high_conf_threshold = phone_cfg.get(
            'high_confidence_auto_alert_threshold', 0.70
        )

        self._red_cooldown = alerts_cfg['red_cooldown_seconds']
        self._yellow_cooldown = alerts_cfg['yellow_cooldown_seconds']

        self._alert_queue = alert_queue
        self._on_alert = on_alert

        self._candidates: Dict[str, CandidateScoreState] = {}
        self._lock = threading.Lock()

        self._running = False
        self._thread: Optional[threading.Thread] = None

        db_path = db_path or config['database']['path']
        self._db_path = db_path
        self._init_database()

        # Stats for isolation testing / debugging
        self.total_events_processed = 0
        self.total_alerts_raised = 0

    # --------------------------------------------------------
    # Database
    # --------------------------------------------------------
    def _init_database(self):
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                module TEXT NOT NULL,
                behaviour_type TEXT,
                confidence REAL,
                contribution REAL,
                candidate_score_after REAL,
                alert_level TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                alert_level TEXT NOT NULL,
                score REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _log_incident(self, event: DetectionEvent, contribution: float,
                       score_after: float, alert_level: str):
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO incidents "
            "(candidate_id, timestamp, module, behaviour_type, confidence, "
            " contribution, candidate_score_after, alert_level) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event.candidate_id, event.timestamp, event.module,
             event.behaviour_type, event.confidence, contribution,
             score_after, alert_level)
        )
        conn.commit()
        conn.close()

    def _log_alert(self, candidate_id: str, level: str, score: float, now: float):
        conn = sqlite3.connect(self._db_path)
        conn.execute(
            "INSERT INTO alerts (candidate_id, timestamp, alert_level, score) "
            "VALUES (?, ?, ?, ?)",
            (candidate_id, now, level, score)
        )
        conn.commit()
        conn.close()

    # --------------------------------------------------------
    # Scoring
    # --------------------------------------------------------
    def _get_duration_multiplier(self, duration_seconds: float) -> float:
        if duration_seconds >= 10:
            return self._duration_multipliers['over_10s']
        elif duration_seconds >= 6:
            return self._duration_multipliers['from_6_to_10s']
        elif duration_seconds >= 3:
            return self._duration_multipliers['from_3_to_6s']
        else:
            return self._duration_multipliers['under_3s']

    def _apply_idle_decay(self, state: CandidateScoreState, now: float):
        """
        Reduce score by decay_percent_per_10s for every full 10s elapsed
        since last_updated, applied lazily (no background timer needed —
        mathematically equivalent to a periodic tick).
        """
        elapsed = now - state.last_updated
        if elapsed <= 0 or state.score <= 0:
            return
        decay_steps = elapsed / 10.0
        decay_factor = (1 - self._decay_percent_per_10s / 100.0) ** decay_steps
        state.score *= decay_factor

    def _prune_recent_events(self, state: CandidateScoreState, now: float):
        state.recent_events = [
            (ts, mod) for (ts, mod) in state.recent_events
            if (now - ts) <= self._score_window
        ]

    def _has_corroborating_signal(self, state: CandidateScoreState,
                                    event: DetectionEvent, now: float) -> bool:
        """True if a DIFFERENT module flagged this candidate within the
        combination window — Section 3.10.4."""
        for ts, mod in state.recent_events:
            if mod != event.module and (now - ts) <= self._combination_window:
                return True
        return False

    def _get_or_create_state(self, candidate_id: str) -> CandidateScoreState:
        if candidate_id not in self._candidates:
            self._candidates[candidate_id] = CandidateScoreState(candidate_id)
        return self._candidates[candidate_id]

    # --------------------------------------------------------
    def process_event(self, event: DetectionEvent) -> dict:
        """
        Process a single DetectionEvent and return a summary dict.
        Public + synchronous so it can be called directly in tests
        without needing the background thread/queue.
        """
        now = event.timestamp or time.time()

        with self._lock:
            state = self._get_or_create_state(event.candidate_id)

            # 1. Decay existing score for idle time since last update
            self._apply_idle_decay(state, now)

            # 2. Prune old events out of the combination-bonus window
            self._prune_recent_events(state, now)

            # 3. Compute this event's contribution
            duration_mult = self._get_duration_multiplier(event.duration_seconds)
            bonus = (self._combination_bonus
                     if self._has_corroborating_signal(state, event, now)
                     else 1.0)
            contribution = event.weighted_score * duration_mult * bonus

            # 3b. High-confidence phone override (see __init__ for
            # rationale) — guarantees THIS SINGLE event alone is enough
            # to push the candidate's score to at least red_threshold,
            # regardless of duration multiplier, combination bonus, or
            # decayed score history. Does not replace the normal
            # formula above — only raises the floor when triggered.
            phone_override_applied = False
            if (event.module == "phone_detection" and
                    event.confidence >= self._phone_high_conf_threshold):
                floor_needed = self._red_threshold - state.score
                if floor_needed > contribution:
                    contribution = floor_needed
                    phone_override_applied = True

            # 4. Update score + event log
            state.score += contribution
            state.score = min(state.score, self._max_score)
            state.recent_events.append((now, event.module))
            state.last_updated = now

            # 5. Determine alert level
            alert_level = self._classify_alert(state.score)

            # 6. Log the incident regardless of alert level (Section 3.11 —
            # all incidents are logged, not just ones that cross a threshold)
            self._log_incident(event, contribution, state.score, alert_level)

            # 7. Dispatch alert if threshold crossed and cooldown allows
            alert_dispatched = self._maybe_dispatch_alert(state, alert_level, now)

            self.total_events_processed += 1

            return {
                'candidate_id': event.candidate_id,
                'module': event.module,
                'contribution': contribution,
                'combination_bonus_applied': bonus > 1.0,
                'phone_high_confidence_override': phone_override_applied,
                'score_after': state.score,
                'alert_level': alert_level,
                'alert_dispatched': alert_dispatched,
            }

    def _classify_alert(self, score: float) -> str:
        if score >= self._red_threshold:
            return "red"
        elif score >= self._yellow_threshold:
            return "yellow"
        return "none"

    def _maybe_dispatch_alert(self, state: CandidateScoreState,
                                alert_level: str, now: float) -> bool:
        state.current_alert_level = alert_level

        if alert_level == "red":
            if (state.last_red_alert_time is None or
                    (now - state.last_red_alert_time) >= self._red_cooldown):
                state.last_red_alert_time = now
                self._fire_alert(state.candidate_id, "red", state.score, now)
                return True

        elif alert_level == "yellow":
            if (state.last_yellow_alert_time is None or
                    (now - state.last_yellow_alert_time) >= self._yellow_cooldown):
                state.last_yellow_alert_time = now
                self._fire_alert(state.candidate_id, "yellow", state.score, now)
                return True

        return False

    def _fire_alert(self, candidate_id: str, level: str, score: float, now: float):
        self.total_alerts_raised += 1
        self._log_alert(candidate_id, level, score, now)
        logger.info(f"[AlertManager] {level.upper()} ALERT — "
                    f"{candidate_id} score={score:.1f}")
        if self._on_alert:
            self._on_alert({
                'candidate_id': candidate_id,
                'level': level,
                'score': score,
                'timestamp': now,
            })

    # --------------------------------------------------------
    # Background thread — consumes the shared alert_queue
    # --------------------------------------------------------
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[AlertManager] Started, consuming shared alert_queue")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _run_loop(self):
        while self._running:
            try:
                event = self._alert_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.process_event(event)
            except Exception as e:
                logger.exception(f"[AlertManager] Error processing event: {e}")

    # --------------------------------------------------------
    # Introspection helpers (for dashboard + tests)
    # --------------------------------------------------------
    def get_candidate_score(self, candidate_id: str) -> Optional[float]:
        with self._lock:
            state = self._candidates.get(candidate_id)
            if state is None:
                return None
            self._apply_idle_decay(state, time.time())
            return state.score

    def get_all_candidate_states(self) -> Dict[str, dict]:
        now = time.time()
        with self._lock:
            result = {}
            for cid, state in self._candidates.items():
                self._apply_idle_decay(state, now)
                result[cid] = {
                    'score': round(state.score, 1),
                    'alert_level': state.current_alert_level,
                    'last_updated': state.last_updated,
                }
            return result
