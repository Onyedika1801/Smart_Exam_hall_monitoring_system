"""
gaze_detection.py
=================
Smart Exam Hall Monitoring System — Gaze & Head Detection Module
Chapter 3 Reference: Section 3.6

Responsibilities:
- Runs as its own dedicated thread
- Reads frames from its own private frame queue
- Uses MediaPipe Face Mesh to extract 468 3D facial landmarks
- Estimates head pose (yaw, pitch, roll) via PnP algorithm (cv2.solvePnP)
- Maintains a persistence counter PER CANDIDATE (not global)
- Only raises a DetectionEvent after 3 continuous seconds outside range
- Brief deviations (< 3s) = base_score_brief (8 pts)
- Sustained deviations (>= 3s) = base_score_sustained (15 pts)
- Outputs DetectionEvent objects to the shared alert queue
- NO scoring logic here — that belongs to alert_manager only

Architecture note (Section 3.13):
  Camera thread → [gaze_frame_queue] → GazeDetectionModule → [alert_queue]

Head orientation thresholds (Chapter 3 Table 3.3):
  Yaw:   outside ±30°  → looking at neighbour (always active)
  Pitch: below personal baseline - 15°  → reading hidden notes
         (personal baseline calibrated over first 180s per candidate;
          "down" is NOT flagged at all during calibration — normal
          question-reading/writing posture would otherwise false-positive)
  Pitch: above +15°   → signalling / looking back (always active)
  Roll:  monitored only, does NOT trigger alert independently
"""

import threading
import time
import logging
import queue
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import cv2
import numpy as np

# Import the shared DetectionEvent from phone_detection
# This ensures all modules output the same contract to alert_manager
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phone_detection import DetectionEvent

logger = logging.getLogger(__name__)


# ============================================================
# 3D Face Model Reference Points
# Used by PnP algorithm to estimate head pose
# These are canonical 3D coordinates of key facial landmarks
# in a normalised face model coordinate system
# ============================================================
FACE_3D_MODEL_POINTS = np.array([
    [0.0,      0.0,      0.0],     # Nose tip (landmark 1)
    [0.0,     -330.0,   -65.0],    # Chin (landmark 152)
    [-225.0,   170.0,  -135.0],    # Left eye left corner (landmark 263)
    [225.0,    170.0,  -135.0],    # Right eye right corner (landmark 33)
    [-150.0,  -150.0,  -125.0],    # Left mouth corner (landmark 287)
    [150.0,   -150.0,  -125.0],    # Right mouth corner (landmark 57)
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
FACE_LANDMARK_INDICES = [1, 152, 263, 33, 287, 57]


# ============================================================
# Per-Candidate State Tracker
# Persistence counter is per candidate — NEVER global
# (Screenshot warning: "Build the persistence counter correctly
#  from the start — per candidate, not global")
# ============================================================
class CandidateGazeState:
    """
    Tracks gaze deviation state for a single candidate.
    Each detected face gets its own instance.
    """
    def __init__(self, candidate_id: str, persistence_seconds: float,
                 calibration_window_seconds: float = 180.0):
        self.candidate_id = candidate_id
        self.persistence_seconds = persistence_seconds

        # Deviation tracking
        self.is_deviating = False
        self.deviation_start_time: Optional[float] = None
        self.deviation_type: Optional[str] = None  # "lateral", "down", "up"

        # Alert cooldown — prevent flooding alert queue
        self.last_alert_time: Optional[float] = None
        self.alert_cooldown = 5.0  # seconds between repeated alerts for same candidate

        # Current angles (for display)
        self.current_yaw = 0.0
        self.current_pitch = 0.0
        self.current_roll = 0.0

        # --- Personalised downward-pitch calibration ---
        # Captures each candidate's natural "reading/writing" pitch over
        # the first N seconds, so normal head-down writing posture isn't
        # mistaken for looking at hidden notes.
        self.created_at = time.time()
        self.calibration_window_seconds = calibration_window_seconds
        self.pitch_samples: list = []
        self.baseline_pitch: Optional[float] = None

    def is_calibrating(self) -> bool:
        """True while this candidate's personal pitch baseline is still open."""
        return (self.baseline_pitch is None and
                (time.time() - self.created_at) < self.calibration_window_seconds)

    def add_pitch_sample(self, pitch: float):
        """Record a pitch reading during the calibration window."""
        self.pitch_samples.append(pitch)
        if len(self.pitch_samples) > 5000:  # safety cap, ~ a few minutes at high FPS
            self.pitch_samples = self.pitch_samples[-5000:]

    def try_finalize_calibration(self, fallback_pitch: float):
        """
        Once the calibration window has elapsed, compute the candidate's
        personal 'natural writing pitch' baseline from collected samples.
        Uses the lower half (more downward) of samples, since candidates
        also glance up/around while reading questions early on — we want
        the baseline to represent their writing angle, not the average of
        everything they did in the window.
        """
        if self.baseline_pitch is not None:
            return  # already calibrated
        if (time.time() - self.created_at) < self.calibration_window_seconds:
            return  # window not finished yet

        if self.pitch_samples:
            sorted_samples = sorted(self.pitch_samples)  # most negative (down) first
            cutoff = sorted_samples[:max(1, len(sorted_samples) // 2)]
            self.baseline_pitch = sum(cutoff) / len(cutoff)
        else:
            # No samples somehow (e.g. candidate rarely faced camera) —
            # fall back to the universal threshold rather than leaving
            # this candidate permanently uncalibrated.
            self.baseline_pitch = fallback_pitch

    def get_deviation_duration(self) -> float:
        """Returns how long current deviation has been ongoing."""
        if self.deviation_start_time is None:
            return 0.0
        return time.time() - self.deviation_start_time

    def can_alert(self) -> bool:
        """Check if enough time has passed since last alert."""
        if self.last_alert_time is None:
            return True
        return (time.time() - self.last_alert_time) >= self.alert_cooldown

    def start_deviation(self, deviation_type: str):
        """Called when head enters suspicious orientation."""
        if not self.is_deviating:
            self.is_deviating = True
            self.deviation_start_time = time.time()
            self.deviation_type = deviation_type

    def end_deviation(self):
        """Called when head returns to normal orientation."""
        self.is_deviating = False
        self.deviation_start_time = None
        self.deviation_type = None

    def record_alert(self):
        """Called after an alert is issued."""
        self.last_alert_time = time.time()


# ============================================================
# Candidate Zone Tracker (same grid logic as phone_detection)
# ============================================================
class CandidateZoneTracker:
    def __init__(self, frame_width: int, frame_height: int,
                 grid_cols: int = 5, grid_rows: int = 4):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.cell_w = frame_width / grid_cols
        self.cell_h = frame_height / grid_rows
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows

    def get_candidate_id(self, x_centre: float, y_centre: float) -> str:
        col = min(int(x_centre / self.cell_w), self.grid_cols - 1)
        row = min(int(y_centre / self.cell_h), self.grid_rows - 1)
        return f"R{row+1}C{col+1}"


# ============================================================
# Head Pose Estimator
# Implements Section 3.6.2 — PnP algorithm via cv2.solvePnP
# ============================================================
class HeadPoseEstimator:
    """
    Estimates yaw, pitch, roll from MediaPipe Face Mesh landmarks
    using the Perspective-n-Point (PnP) algorithm.
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height

        # Camera matrix — approximated from frame dimensions
        # Focal length approximated as frame width (reasonable for webcams)
        focal_length = frame_width
        centre = (frame_width / 2, frame_height / 2)
        self.camera_matrix = np.array([
            [focal_length, 0,            centre[0]],
            [0,            focal_length, centre[1]],
            [0,            0,            1         ]
        ], dtype=np.float64)

        # Assuming no lens distortion (standard for webcams)
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def estimate(self, landmarks, frame_width: int,
                 frame_height: int) -> Tuple[float, float, float]:
        """
        Estimate yaw, pitch, roll from Face Mesh landmarks.

        Returns:
            (yaw, pitch, roll) in degrees
            yaw:   negative = left, positive = right
            pitch: negative = down, positive = up
            roll:  negative = tilt left, positive = tilt right
        """
        # Extract 2D image points for the 6 reference landmarks
        image_points_2d = []
        for idx in FACE_LANDMARK_INDICES:
            lm = landmarks[idx]
            x = lm.x * frame_width
            y = lm.y * frame_height
            image_points_2d.append([x, y])
        image_points_2d = np.array(image_points_2d, dtype=np.float64)

        # Solve PnP — find rotation and translation vectors
        success, rotation_vector, translation_vector = cv2.solvePnP(
            FACE_3D_MODEL_POINTS,
            image_points_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix (Rodrigues transformation)
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        # Decompose rotation matrix into Euler angles.
        #
        # NOTE: We deliberately do NOT use cv2.RQDecomp3x3 here. It has a
        # known ambiguity at steep angles — during testing, a sustained
        # downward head tilt (real pitch around -40°) was instead reported
        # as pitch ≈ +47° with roll snapping to ≈176° (i.e. flipping to an
        # equivalent-but-differently-signed decomposition branch). That
        # caused genuine "down" deviations to be misread as "up" and
        # bypass the personalised down-calibration entirely.
        #
        # This atan2-based extraction stays continuous and correctly
        # signed across the full pitch range, including steep down tilts.
        yaw, pitch, roll = self._rotation_matrix_to_euler(rotation_matrix)

        return yaw, pitch, roll

    @staticmethod
    def _rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert a rotation matrix to (yaw, pitch, roll) in degrees using
        direct atan2 extraction. More robust than cv2.RQDecomp3x3 for
        head-pose estimation since it doesn't jump between equivalent
        Euler-angle branches at steep angles.
        """
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        else:
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            yaw   = 0.0
            roll  = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))

        return float(yaw), float(pitch), float(roll)


# ============================================================
# Gaze Detection Module
# ============================================================
class GazeDetectionModule:
    """
    MediaPipe Face Mesh based head pose and gaze detection module.

    Usage:
        module = GazeDetectionModule(config, alert_queue, camera_id="cam_0")
        module.start()
        module.put_frame(frame, frame_number)
        module.stop()
    """

    def __init__(self, config: dict, alert_queue: queue.Queue,
                 camera_id: str = "cam_0"):
        self.config = config
        self.alert_queue = alert_queue
        self.camera_id = camera_id

        # Gaze config (Section 3.6.3 and config.yaml)
        gaze_cfg = config['gaze_detection']
        self._yaw_threshold    = gaze_cfg['yaw_threshold']        # 30.0 degrees
        self._pitch_down_fallback = gaze_cfg['pitch_down_threshold']  # -20.0 fallback only
        self._pitch_up         = gaze_cfg['pitch_up_threshold']    # +15.0 degrees
        self._persistence_secs = gaze_cfg['persistence_seconds']   # 3.0 seconds
        self._base_sustained   = gaze_cfg['base_score_sustained']  # 15 pts
        self._base_brief       = gaze_cfg['base_score_brief']      # 8 pts

        # Personalised pitch-down calibration
        self._pitch_calibration_secs = gaze_cfg['pitch_calibration_window_seconds']  # 180.0
        self._pitch_down_margin = gaze_cfg['pitch_down_margin']    # 15.0 degrees

        # Each module has its OWN frame queue (Section 3.13)
        max_q = config['queues']['max_size']
        self.frame_queue = queue.Queue(maxsize=max_q)

        # Threading
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="GazeDetectionThread"
        )
        self._stop_event = threading.Event()

        # Per-candidate state — persistence counter is per candidate, not global
        self._candidate_states: Dict[str, CandidateGazeState] = {}

        # MediaPipe and pose estimator — initialised in thread
        self._face_mesh = None
        self._pose_estimator: Optional[HeadPoseEstimator] = None
        self._zone_tracker: Optional[CandidateZoneTracker] = None

        # Stats
        self.frames_processed = 0
        self.detections_total = 0
        self._start_time = None

        # Store last frame's detection results for draw_detections()
        self._last_face_data = []  # List of (candidate_id, yaw, pitch, roll, bbox)

        logger.info(f"[GazeDetection] Module initialised — camera: {camera_id}")

    def start(self):
        self._start_time = time.time()
        self._thread.start()
        logger.info("[GazeDetection] Thread started")

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info(f"[GazeDetection] Stopped. "
                    f"Processed {self.frames_processed} frames, "
                    f"{self.detections_total} detections total.")

    def put_frame(self, frame: np.ndarray, frame_number: int):
        """Add frame to this module's queue, dropping oldest if full."""
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
        return {
            'frames_processed': self.frames_processed,
            'detections_total': self.detections_total,
            'fps': round(fps, 2),
            'queue_size': self.frame_queue.qsize(),
            'tracked_candidates': len(self._candidate_states)
        }

    # --------------------------------------------------------
    # Internal — runs in dedicated thread
    # --------------------------------------------------------
    def _run(self):
        """Main detection loop."""
        self._load_mediapipe()

        while not self._stop_event.is_set():
            try:
                frame, frame_number = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_frame(frame, frame_number)
            except Exception as e:
                logger.error(f"[GazeDetection] Frame {frame_number} error: {e}")

            self.frames_processed += 1

    def _load_mediapipe(self):
        """Load MediaPipe Face Mesh. Called once when thread starts."""
        try:
            import mediapipe as mp
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,      # Video mode — tracks across frames
                max_num_faces=15,             # Up to 15 candidates per frame
                refine_landmarks=True,        # Better iris landmarks
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            logger.info("[GazeDetection] MediaPipe Face Mesh loaded")
        except Exception as e:
            logger.error(f"[GazeDetection] Failed to load MediaPipe: {e}")
            self._stop_event.set()

    def _process_frame(self, frame: np.ndarray, frame_number: int):
        """
        Process a single frame:
        1. Detect all faces with MediaPipe Face Mesh
        2. Estimate head pose for each face via PnP
        3. Update per-candidate persistence counters
        4. Raise DetectionEvent if persistence threshold exceeded
        """
        if self._face_mesh is None:
            return

        h, w = frame.shape[:2]

        # Initialise helpers on first frame
        if self._pose_estimator is None:
            self._pose_estimator = HeadPoseEstimator(w, h)
        if self._zone_tracker is None:
            self._zone_tracker = CandidateZoneTracker(w, h)

        # MediaPipe requires RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._face_mesh.process(frame_rgb)
        frame_rgb.flags.writeable = True

        face_data = []  # For draw_detections()

        if not results.multi_face_landmarks:
            # No faces detected — end any active deviations
            # (candidates may have left frame)
            self._last_face_data = []
            return

        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Get face bounding box from landmarks
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            x_centre = (x1 + x2) / 2
            y_centre = (y1 + y2) / 2

            # Assign candidate ID from grid position
            candidate_id = self._zone_tracker.get_candidate_id(x_centre, y_centre)

            # Get or create per-candidate state
            if candidate_id not in self._candidate_states:
                self._candidate_states[candidate_id] = CandidateGazeState(
                    candidate_id, self._persistence_secs,
                    calibration_window_seconds=self._pitch_calibration_secs
                )
            state = self._candidate_states[candidate_id]

            # Estimate head pose
            yaw, pitch, roll = self._pose_estimator.estimate(landmarks, w, h)

            # Update state for display
            state.current_yaw = yaw
            state.current_pitch = pitch
            state.current_roll = roll

            # Feed the personal pitch-down calibration
            if state.is_calibrating():
                state.add_pitch_sample(pitch)
            state.try_finalize_calibration(fallback_pitch=self._pitch_down_fallback)

            # Store for draw_detections
            face_data.append((candidate_id, yaw, pitch, roll,
                              (x1, y1, x2, y2)))

            # Determine if head is in suspicious orientation
            deviation_type = self._classify_deviation(yaw, pitch, state)

            if deviation_type:
                # Head is outside normal range
                state.start_deviation(deviation_type)
                duration = state.get_deviation_duration()

                # Only raise event after persistence threshold (3 seconds)
                # OR immediately for very large deviations
                should_alert = (
                    duration >= self._persistence_secs and state.can_alert()
                )

                if should_alert:
                    # Determine score based on duration (Table 3.5)
                    is_sustained = duration >= self._persistence_secs
                    base_score = (self._base_sustained if is_sustained
                                  else self._base_brief)

                    behaviour_type = self._get_behaviour_type(deviation_type)

                    event = DetectionEvent(
                        module="gaze_detection",
                        candidate_id=candidate_id,
                        behaviour_type=behaviour_type,
                        confidence=1.0,       # MediaPipe doesn't give conf score
                        base_score=base_score,
                        weighted_score=float(base_score),  # No conf weighting for MediaPipe
                        bbox=(x1, y1, x2, y2),
                        frame_number=frame_number,
                        timestamp=time.time(),
                        camera_id=self.camera_id,
                        duration_seconds=duration,
                        requires_persistence=True
                    )

                    try:
                        self.alert_queue.put_nowait(event)
                        state.record_alert()
                        self.detections_total += 1
                        logger.debug(
                            f"[GazeDetection] {behaviour_type} — "
                            f"candidate: {candidate_id}, "
                            f"yaw: {yaw:.1f}°, pitch: {pitch:.1f}°, "
                            f"duration: {duration:.1f}s, "
                            f"score: {base_score}"
                        )
                    except queue.Full:
                        logger.warning("[GazeDetection] Alert queue full")

            else:
                # Head back in normal range — reset persistence counter
                state.end_deviation()

        self._last_face_data = face_data

    def _classify_deviation(self, yaw: float, pitch: float,
                             state: CandidateGazeState) -> Optional[str]:
        """
        Check if head orientation is outside normal range.
        Returns deviation type string or None if normal.

        Chapter 3 Table 3.3 (updated after false-positive testing):
          Yaw outside ±30°  → "lateral"   — always active, incl. calibration
          Pitch above +15°  → "up"        — always active, incl. calibration
          Pitch down        → "down"      — ONLY checked once this candidate's
                               personal baseline is calibrated. During the
                               calibration window, downward pitch is never
                               flagged — that's the whole point of learning
                               each candidate's natural writing angle first.
          Roll not checked here (monitored but doesn't trigger)
        """
        if abs(yaw) > self._yaw_threshold:
            return "lateral"
        if pitch > self._pitch_up:
            return "up"

        # Downward pitch — personalised, and exempt during calibration
        if state.baseline_pitch is not None:
            effective_pitch_down = state.baseline_pitch - self._pitch_down_margin
            if pitch < effective_pitch_down:
                return "down"
        # else: still calibrating — intentionally not checking "down" at all

        return None

    def _get_behaviour_type(self, deviation_type: str) -> str:
        """Map deviation type to behaviour description for database logging."""
        mapping = {
            "lateral": "gaze_lateral_deviation",   # Looking at neighbour
            "down":    "gaze_downward_deviation",   # Reading hidden notes
            "up":      "gaze_upward_deviation",     # Signalling
        }
        return mapping.get(deviation_type, "gaze_deviation")

    def draw_detections(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw head pose indicators on frame for dashboard/test display.
        Call this in the display thread, not the detection thread.
        """
        frame_out = frame.copy()

        for candidate_id, yaw, pitch, roll, bbox in self._last_face_data:
            x1, y1, x2, y2 = bbox
            state = self._candidate_states.get(candidate_id)
            deviation_type = (self._classify_deviation(yaw, pitch, state)
                               if state else None)

            calibrating = state.is_calibrating() if state else False

            # Colour priority: an ACTIVE deviation (lateral/up — these stay
            # fully active during calibration) always takes precedence over
            # the "calibrating" indicator. Blue only shows when calibrating
            # AND currently normal (no lateral/up deviation right now).
            # Green = normal, orange = brief deviation, red = sustained deviation.
            if deviation_type is not None:
                if state and state.get_deviation_duration() >= self._persistence_secs:
                    colour = (0, 0, 255)       # Red — sustained deviation
                else:
                    colour = (0, 165, 255)     # Orange — brief deviation
            elif calibrating:
                colour = (255, 165, 0)         # Blue — calibrating, currently normal
            else:
                colour = (0, 255, 0)           # Green — normal

            # Draw face bounding box
            cv2.rectangle(frame_out, (x1, y1), (x2, y2), colour, 2)

            # Draw angle labels
            cal_tag = " [CALIBRATING]" if calibrating else ""
            label1 = f"{candidate_id} Y:{yaw:.0f} P:{pitch:.0f} R:{roll:.0f}{cal_tag}"
            cv2.putText(frame_out, label1, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)

            # Draw deviation duration if active
            if state and state.is_deviating:
                dur = state.get_deviation_duration()
                label2 = f"DEV: {dur:.1f}s / {self._persistence_secs}s"
                cv2.putText(frame_out, label2, (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        return frame_out
