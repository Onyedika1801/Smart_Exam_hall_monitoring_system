"""
posture_analysis.py
===================
Smart Exam Hall Monitoring System — Posture Analysis Module
Chapter 3 Reference: Section 3.9

Responsibilities:
- Runs as its own dedicated thread
- Reads frames from its own private frame queue
- Uses MediaPipe Pose to extract 33 body keypoints per person
- Records personalised baseline for each candidate during first 60 seconds
- Detects shoulder asymmetry and forward lean deviations
- Persistence counter is PER CANDIDATE — never global
- Only raises DetectionEvent after sustained deviation beyond personal baseline
- Brief deviation (< 3s) = base_score_brief (5 pts)
- Sustained deviation (>= 3s) = base_score_sustained (10 pts)
- Outputs DetectionEvent objects to the shared alert queue
- NO scoring logic here — belongs to alert_manager only

Architecture note (Section 3.13):
  Camera thread → [posture_frame_queue] → PostureAnalysisModule → [alert_queue]

Key design feature (Section 3.9.2):
  Personalised baseline calibration — each candidate's threshold is derived
  from their OWN natural posture during the first 60 seconds, NOT a universal
  fixed value. This substantially reduces false positives.
"""

import threading
import time
import logging
import queue
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phone_detection import DetectionEvent

logger = logging.getLogger(__name__)


# ============================================================
# MediaPipe Pose Landmark Indices
# Section 3.9.1 — relevant landmarks for posture analysis
# ============================================================
# Left shoulder:  11
# Right shoulder: 12
# Left hip:       23
# Right hip:      24
# Nose:           0
# Left ear:       7
# Right ear:      8

LANDMARK = {
    'nose':            0,
    'left_ear':        7,
    'right_ear':       8,
    'left_shoulder':   11,
    'right_shoulder':  12,
    'left_hip':        23,
    'right_hip':       24,
}


# ============================================================
# Per-Candidate Baseline and State
# Section 3.9.2 — Personalised Baseline Calibration
# ============================================================
class CandidatePostureState:
    """
    Tracks posture baseline and deviation state for a single candidate.

    During the first 60 seconds (calibration window):
      - Records shoulder asymmetry and head height values each frame
      - Computes mean and std dev of each measurement

    After calibration:
      - Flags deviations beyond mean + (std_multiplier * std_dev)
      - This is personal to each candidate — not a universal threshold
    """

    def __init__(self, candidate_id: str,
                 calibration_seconds: float,
                 std_multiplier: float,
                 persistence_seconds: float):
        self.candidate_id = candidate_id
        self.calibration_seconds = calibration_seconds
        self.std_multiplier = std_multiplier
        self.persistence_seconds = persistence_seconds

        # Calibration
        self.calibration_start: Optional[float] = None
        self.is_calibrated = False
        self.calibration_asymmetry_values: List[float] = []
        self.calibration_head_height_values: List[float] = []

        # Personal thresholds (set after calibration)
        self.asymmetry_mean = 0.0
        self.asymmetry_std = 0.0
        self.asymmetry_threshold = 0.0

        self.head_height_mean = 0.0
        self.head_height_std = 0.0
        self.head_height_threshold = 0.0

        # Deviation tracking
        self.is_deviating = False
        self.deviation_start_time: Optional[float] = None
        self.deviation_type: Optional[str] = None

        # Alert cooldown
        self.last_alert_time: Optional[float] = None
        self.alert_cooldown = 8.0

        # Current measurements (for display)
        self.current_asymmetry = 0.0
        self.current_head_height = 0.0

    def start_calibration(self):
        """Begin recording baseline values."""
        self.calibration_start = time.time()
        logger.debug(f"[Posture] Calibration started for {self.candidate_id}")

    def is_in_calibration(self) -> bool:
        """Check if still in calibration window."""
        if self.calibration_start is None:
            return False
        return (time.time() - self.calibration_start) < self.calibration_seconds

    def add_calibration_sample(self, asymmetry: float, head_height: float):
        """Record a measurement during calibration window."""
        self.calibration_asymmetry_values.append(asymmetry)
        self.calibration_head_height_values.append(head_height)

    def finalise_calibration(self):
        """
        Compute personal thresholds from calibration data.
        Threshold = mean + (std_multiplier * std_dev)
        Called automatically when calibration window ends.
        """
        if len(self.calibration_asymmetry_values) < 10:
            # Not enough clean frames — use conservative defaults
            logger.warning(
                f"[Posture] {self.candidate_id}: insufficient calibration frames "
                f"({len(self.calibration_asymmetry_values)}), using defaults"
            )
            self.asymmetry_threshold = 0.15
            self.head_height_threshold = 0.20
        else:
            # Shoulder asymmetry thresholds
            self.asymmetry_mean = float(np.mean(self.calibration_asymmetry_values))
            self.asymmetry_std = float(np.std(self.calibration_asymmetry_values))
            self.asymmetry_threshold = (
                self.asymmetry_mean + self.std_multiplier * self.asymmetry_std
            )

            # Head height thresholds
            self.head_height_mean = float(np.mean(self.calibration_head_height_values))
            self.head_height_std = float(np.std(self.calibration_head_height_values))
            self.head_height_threshold = (
                self.head_height_mean - self.std_multiplier * self.head_height_std
            )

        self.is_calibrated = True
        logger.info(
            f"[Posture] {self.candidate_id} calibrated — "
            f"asymmetry threshold: {self.asymmetry_threshold:.3f}, "
            f"head height threshold: {self.head_height_threshold:.3f}"
        )

    def get_calibration_progress(self) -> float:
        """Returns calibration progress 0.0 to 1.0."""
        if self.calibration_start is None:
            return 0.0
        elapsed = time.time() - self.calibration_start
        return min(elapsed / self.calibration_seconds, 1.0)

    def get_deviation_duration(self) -> float:
        """
        Total elapsed time since this deviation began. Used to decide
        WHETHER a deviation has persisted long enough to alert at all
        (checked against persistence_seconds) — NOT used for the
        duration value reported to alert_manager, see
        get_segment_duration() below.
        """
        if self.deviation_start_time is None:
            return 0.0
        return time.time() - self.deviation_start_time

    def get_segment_duration(self) -> float:
        """
        Duration value reported to alert_manager for THIS SPECIFIC alert.

        Uses time since the LAST alert (if this deviation has already
        alerted before), not time since the deviation began. Without
        this distinction, a single ongoing deviation that keeps
        re-alerting every alert_cooldown seconds would report an
        ever-growing cumulative duration (8s, 16s, 24s, 32s...) on every
        repeat — and since alert_manager's duration multiplier maxes out
        at ×3.0 for anything over 10s, every repeat past the first would
        be scored as a fresh maximum-duration incident indefinitely,
        even though each repeat really only represents ~alert_cooldown
        seconds of additional sustained behaviour. This caused runaway
        score growth in live integrated testing (main.py) — a candidate
        who deviated once and stayed deviating (or whose deviation was
        still being processed from a backlogged frame queue well after
        they'd already stopped) accumulated an escalating score with no
        ceiling, reaching 200+ in one observed run.

        This method makes each repeat alert report a roughly constant,
        bounded duration (~alert_cooldown) instead, so repeat alerts on
        one ongoing deviation land in a stable multiplier tier rather
        than compounding forever.
        """
        if self.last_alert_time is None:
            return self.get_deviation_duration()
        return time.time() - self.last_alert_time

    def can_alert(self) -> bool:
        if self.last_alert_time is None:
            return True
        return (time.time() - self.last_alert_time) >= self.alert_cooldown

    def start_deviation(self, deviation_type: str):
        if not self.is_deviating:
            self.is_deviating = True
            self.deviation_start_time = time.time()
            self.deviation_type = deviation_type

    def end_deviation(self):
        self.is_deviating = False
        self.deviation_start_time = None
        self.deviation_type = None

    def record_alert(self):
        self.last_alert_time = time.time()


# ============================================================
# Candidate Zone Tracker
# ============================================================
class CandidateZoneTracker:
    def __init__(self, frame_width: int, frame_height: int,
                 grid_cols: int = 5, grid_rows: int = 4):
        self.cell_w = frame_width / grid_cols
        self.cell_h = frame_height / grid_rows
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

    def get_candidate_id(self, x_centre: float, y_centre: float) -> str:
        col = min(int(x_centre / self.cell_w), self.grid_cols - 1)
        row = min(int(y_centre / self.cell_h), self.grid_rows - 1)
        return f"R{row+1}C{col+1}"


# ============================================================
# Posture Measurement Functions
# Section 3.9.1
# ============================================================
def compute_shoulder_asymmetry(landmarks, frame_height: int) -> Optional[float]:
    """
    Compute normalised vertical difference between shoulders.
    Returns value between 0 and 1.
    0 = perfectly level shoulders
    Higher = more asymmetry (leaning to one side)

    Normalised by inter-shoulder distance so it's scale-independent
    and consistent across different body sizes and camera distances.
    """
    left_shoulder  = landmarks[LANDMARK['left_shoulder']]
    right_shoulder = landmarks[LANDMARK['right_shoulder']]

    if left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5:
        return None  # Landmarks not reliable enough

    left_y  = left_shoulder.y * frame_height
    right_y = right_shoulder.y * frame_height
    left_x  = left_shoulder.x
    right_x = right_shoulder.x

    # Vertical difference between shoulders
    vertical_diff = abs(left_y - right_y)

    # Inter-shoulder distance for normalisation
    inter_shoulder = abs(left_x - right_x)
    if inter_shoulder < 0.01:
        return None  # Candidate facing sideways — unreliable

    # Normalised asymmetry
    asymmetry = vertical_diff / (inter_shoulder * frame_height)
    return float(asymmetry)


def compute_head_height(landmarks, frame_height: int) -> Optional[float]:
    """
    Compute normalised vertical position of nose relative to shoulder midpoint.
    Section 3.9.1:
      'In neutral seated posture, the nose sits consistently above the
       shoulder midpoint. When this distance decreases beyond a threshold,
       the module flags a forward lean event.'

    Returns normalised ratio — higher = head further above shoulders (normal)
    Lower = head dropping toward desk (forward lean = suspicious)
    """
    nose           = landmarks[LANDMARK['nose']]
    left_shoulder  = landmarks[LANDMARK['left_shoulder']]
    right_shoulder = landmarks[LANDMARK['right_shoulder']]

    if (nose.visibility < 0.5 or
            left_shoulder.visibility < 0.5 or
            right_shoulder.visibility < 0.5):
        return None

    nose_y = nose.y * frame_height
    shoulder_mid_y = ((left_shoulder.y + right_shoulder.y) / 2) * frame_height

    # Positive = nose above shoulders (normal upright posture)
    # Negative or very small = nose at or below shoulders (forward lean)
    head_height = (shoulder_mid_y - nose_y) / frame_height
    return float(head_height)


# ============================================================
# Posture Analysis Module
# ============================================================
class PostureAnalysisModule:
    """
    MediaPipe Pose based posture analysis module.

    Usage:
        module = PostureAnalysisModule(config, alert_queue, camera_id="cam_0")
        module.start()
        module.put_frame(frame, frame_number)
        module.stop()
    """

    def __init__(self, config: dict, alert_queue: queue.Queue,
                 camera_id: str = "cam_0"):
        self.config = config
        self.alert_queue = alert_queue
        self.camera_id = camera_id

        # Posture config (Section 3.9 and config.yaml)
        posture_cfg = config['posture_analysis']
        self._calibration_seconds = posture_cfg['calibration_window_seconds']  # 60s
        self._std_multiplier      = posture_cfg['deviation_std_multiplier']    # 2.0
        self._persistence_secs    = posture_cfg['persistence_seconds']         # 3.0
        self._base_sustained      = posture_cfg['base_score_sustained']        # 10 pts
        self._base_brief          = posture_cfg['base_score_brief']            # 5 pts

        # Each module has its OWN frame queue (Section 3.13)
        max_q = config['queues']['max_size']
        self.frame_queue = queue.Queue(maxsize=max_q)

        # Threading
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PostureAnalysisThread"
        )
        self._stop_event = threading.Event()

        # Per-candidate state
        self._candidate_states: Dict[str, CandidatePostureState] = {}

        # MediaPipe — initialised in thread
        self._pose = None
        self._zone_tracker: Optional[CandidateZoneTracker] = None

        # Stats
        self.frames_processed = 0
        self.detections_total = 0
        self._start_time = None

        # Last frame data for draw_detections()
        self._last_pose_data = []

        logger.info(f"[PostureAnalysis] Module initialised — camera: {camera_id}")

    def start(self):
        self._start_time = time.time()
        self._thread.start()
        logger.info("[PostureAnalysis] Thread started")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info(
            f"[PostureAnalysis] Stopped. "
            f"Processed {self.frames_processed} frames, "
            f"{self.detections_total} detections total."
        )

    def put_frame(self, frame: np.ndarray, frame_number: int):
        """Add frame to queue, drop oldest if full."""
        try:
            self.frame_queue.put_nowait((frame, frame_number))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait((frame, frame_number))
            except queue.Full:
                pass

    def get_stats(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        calibrated = sum(1 for s in self._candidate_states.values()
                         if s.is_calibrated)
        calibrating = sum(1 for s in self._candidate_states.values()
                          if not s.is_calibrated and s.calibration_start)
        return {
            'frames_processed': self.frames_processed,
            'detections_total': self.detections_total,
            'fps': round(fps, 2),
            'queue_size': self.frame_queue.qsize(),
            'tracked_candidates': len(self._candidate_states),
            'calibrated_candidates': calibrated,
            'calibrating_candidates': calibrating,
        }

    # --------------------------------------------------------
    # Internal — runs in dedicated thread
    # --------------------------------------------------------
    def _run(self):
        self._load_mediapipe()
        while not self._stop_event.is_set():
            try:
                frame, frame_number = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._process_frame(frame, frame_number)
            except Exception as e:
                logger.error(f"[PostureAnalysis] Frame {frame_number} error: {e}")
            self.frames_processed += 1

    def _load_mediapipe(self):
        """Load MediaPipe Pose. Called once when thread starts."""
        try:
            import mediapipe as mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,          # 0=fast, 1=balanced, 2=accurate
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("[PostureAnalysis] MediaPipe Pose loaded")
        except Exception as e:
            logger.error(f"[PostureAnalysis] Failed to load MediaPipe: {e}")
            self._stop_event.set()

    def _process_frame(self, frame: np.ndarray, frame_number: int):
        """
        Process a single frame:
        1. Detect body pose landmarks with MediaPipe Pose
        2. Compute shoulder asymmetry and head height
        3. During calibration: record baseline values
        4. After calibration: compare to personal threshold
        5. Raise DetectionEvent if deviation persists beyond threshold
        """
        if self._pose is None:
            return

        h, w = frame.shape[:2]

        if self._zone_tracker is None:
            self._zone_tracker = CandidateZoneTracker(w, h)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        pose_data = []

        if not results.pose_landmarks:
            self._last_pose_data = []
            return

        landmarks = results.pose_landmarks.landmark

        # Get candidate position from shoulder midpoint
        left_shoulder  = landmarks[LANDMARK['left_shoulder']]
        right_shoulder = landmarks[LANDMARK['right_shoulder']]

        if (left_shoulder.visibility < 0.5 or right_shoulder.visibility < 0.5):
            self._last_pose_data = []
            return

        shoulder_mid_x = ((left_shoulder.x + right_shoulder.x) / 2) * w
        shoulder_mid_y = ((left_shoulder.y + right_shoulder.y) / 2) * h
        candidate_id = self._zone_tracker.get_candidate_id(
            shoulder_mid_x, shoulder_mid_y
        )

        # Get or create candidate state
        if candidate_id not in self._candidate_states:
            state = CandidatePostureState(
                candidate_id,
                self._calibration_seconds,
                self._std_multiplier,
                self._persistence_secs
            )
            state.start_calibration()
            self._candidate_states[candidate_id] = state
        state = self._candidate_states[candidate_id]

        # Compute posture measurements
        asymmetry   = compute_shoulder_asymmetry(landmarks, h)
        head_height = compute_head_height(landmarks, h)

        if asymmetry is None or head_height is None:
            self._last_pose_data = []
            return

        state.current_asymmetry   = asymmetry
        state.current_head_height = head_height

        # Bounding box from shoulder landmarks
        nose = landmarks[LANDMARK['nose']]
        x1 = int(min(left_shoulder.x, right_shoulder.x) * w) - 20
        y1 = int(nose.y * h) - 20
        x2 = int(max(left_shoulder.x, right_shoulder.x) * w) + 20
        y2 = int(max(left_shoulder.y, right_shoulder.y) * h) + 20
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        pose_data.append((candidate_id, asymmetry, head_height,
                          state, (x1, y1, x2, y2)))

        # ---- Calibration phase ----
        if not state.is_calibrated:
            if state.is_in_calibration():
                state.add_calibration_sample(asymmetry, head_height)
            else:
                state.finalise_calibration()
            self._last_pose_data = pose_data
            return

        # ---- Detection phase (after calibration) ----
        deviation_type = self._classify_deviation(state, asymmetry, head_height)

        if deviation_type:
            state.start_deviation(deviation_type)
            duration = state.get_deviation_duration()

            if duration >= self._persistence_secs and state.can_alert():
                is_sustained = duration >= self._persistence_secs
                base_score = (self._base_sustained if is_sustained
                              else self._base_brief)

                # Computed BEFORE record_alert() below updates
                # last_alert_time — see get_segment_duration() docstring
                # for why this must NOT be the cumulative duration.
                reported_duration = state.get_segment_duration()

                event = DetectionEvent(
                    module="posture_analysis",
                    candidate_id=candidate_id,
                    behaviour_type=f"posture_{deviation_type}",
                    confidence=1.0,
                    base_score=base_score,
                    weighted_score=float(base_score),
                    bbox=(x1, y1, x2, y2),
                    frame_number=frame_number,
                    timestamp=time.time(),
                    camera_id=self.camera_id,
                    duration_seconds=reported_duration,
                    requires_persistence=True
                )

                try:
                    self.alert_queue.put_nowait(event)
                    state.record_alert()
                    self.detections_total += 1
                    logger.debug(
                        f"[PostureAnalysis] posture_{deviation_type} — "
                        f"candidate: {candidate_id}, "
                        f"duration: {duration:.1f}s, "
                        f"score: {base_score}"
                    )
                except queue.Full:
                    logger.warning("[PostureAnalysis] Alert queue full")
        else:
            state.end_deviation()

        self._last_pose_data = pose_data

    def _classify_deviation(self, state: CandidatePostureState,
                             asymmetry: float,
                             head_height: float) -> Optional[str]:
        """
        Compare current measurements to personal baseline thresholds.
        Returns deviation type or None if within normal range.
        """
        if asymmetry > state.asymmetry_threshold:
            return "shoulder_asymmetry"   # Leaning to one side
        if head_height < state.head_height_threshold:
            return "forward_lean"         # Head dropping toward desk
        return None

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """Draw posture indicators on frame for display."""
        frame_out = frame.copy()

        for candidate_id, asymmetry, head_height, state, bbox in self._last_pose_data:
            x1, y1, x2, y2 = bbox
            deviation = self._classify_deviation(state, asymmetry, head_height) \
                        if state.is_calibrated else None

            # Colour coding
            if not state.is_calibrated:
                progress = state.get_calibration_progress()
                colour = (255, 165, 0)   # Blue — calibrating
                label = f"{candidate_id} CAL {int(progress*100)}%"
            elif deviation is None:
                colour = (0, 255, 0)     # Green — normal
                label = (f"{candidate_id} "
                         f"A:{asymmetry:.2f} H:{head_height:.2f}")
            elif state.get_deviation_duration() >= self._persistence_secs:
                colour = (0, 0, 255)     # Red — alert
                label = f"{candidate_id} ALERT: {deviation}"
            else:
                colour = (0, 165, 255)   # Orange — deviating
                label = (f"{candidate_id} DEV:{state.get_deviation_duration():.1f}s"
                         f" {deviation}")

            cv2.rectangle(frame_out, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame_out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1)

            # Show thresholds if calibrated
            if state.is_calibrated:
                thresh_label = (f"A_thresh:{state.asymmetry_threshold:.2f} "
                                f"H_thresh:{state.head_height_threshold:.2f}")
                cv2.putText(frame_out, thresh_label, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        return frame_out
