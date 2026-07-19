"""
object_passing.py
==================
Object Passing Detection Module — Chapter 3 Section 3.8

DESIGN APPROACH
----------------
Two earlier approaches were considered and rejected before this design:

  1. Pure per-zone object COUNT deltas (zone loses one item, adjacent
     zone gains one) — cheap, but structurally CANNOT detect a
     simultaneous swap (e.g. two candidates trading answer sheets at
     the same time), because neither zone's total count changes.

  2. Full persistent object tracking (DeepSORT/ByteTrack) to follow an
     individual item's identity frame-to-frame — would catch swaps
     accurately, but is too computationally expensive for real-time
     CPU inference alongside the other three modules already running.

WHAT THIS MODULE ACTUALLY DOES (Option A — hand-crossing detection):
Instead of tracking objects, it tracks HAND MOVEMENT across the shared
boundary between two adjacent candidate zones, using MediaPipe Hands
for wrist landmarks. A crossing only counts as a probable pass if a
YOLO-detected small object (paper/pen/phone) is near the wrist at the
moment of crossing. This mirrors what a human invigilator actually
watches for — a hand moving into a neighbour's space while holding
something — rather than inferring it indirectly from before/after
counts. Critically, this DOES catch simultaneous swaps, since it's
triggered by the crossing motion itself, not a net count change.

ATTENDANCE / MASS-DISTRIBUTION HANDLING (pattern-based, Option 2):
Question papers and attendance sheets are typically passed down a row
in a short synchronised burst at a known point in the exam. This
produces many concurrent hand-crossings across DIFFERENT zone-pairs
within a few seconds — a distinct signature from an isolated, later,
one-off transfer between two specific candidates (which is far more
consistent with malpractice). If enough concurrent crossings are seen
within a short window, they are logged but excluded from scoring. A
simple time-based grace window at the start of the exam is layered on
top as a fallback for slower/staggered distribution.

DOCUMENTED LIMITATIONS (Option B — honesty over overclaiming)
----------------------------------------------------------------
- Object CLASSIFICATION is not attempted. The module flags "an
  object-like item crossed a zone boundary in a hand," not what the
  item specifically was. This is a deliberate scope decision — see
  Section 3.8 discussion in Chapter 3/4 write-up.
- The underlying object detector currently reuses the phone-detection
  YOLOv8n weights as a placeholder. Papers/pens/answer-sheet edges
  were not part of that training dataset, so detection of NON-phone
  items is expected to be materially weaker until a dedicated dataset
  is collected and the model retrained (Table 3.4 would need new
  classes: loose paper, pen/pencil, folded note).
- MediaPipe Hands requires a reasonably clear, unobstructed view of
  hands to produce wrist landmarks. At the current camera's 640x480
  ceiling and typical exam-hall camera distances, hand landmark
  quality is expected to be weaker than face/pose landmarks, which
  operate on a larger, more central subject.
- Burst detection is a heuristic pattern match, not a certainty. A
  genuinely isolated two-candidate collusion event that happens to
  coincide with a legitimate mass-distribution burst would be
  incorrectly suppressed. This is treated as an acceptable trade-off
  for reducing the far more common false-positive case (flagging
  every candidate during routine attendance/script handling).
- This module is expected to have the lowest reliability of the four
  detection modules by nature of the problem (multi-person, relational,
  motion-based). It is designed to contribute to the combination-bonus
  scoring alongside other modules (Section 3.10.4), not to stand alone
  as a high-confidence single-signal detector.
"""

import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO

from modules.phone_detection import DetectionEvent
from modules.posture_analysis import CandidateZoneTracker

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================

@dataclass
class HandObservation:
    """One hand's position + nearby-object status in a single frame."""
    wrist_x: float
    wrist_y: float
    zone_id: str
    holding_object: bool
    timestamp: float


@dataclass
class CrossingEvent:
    """A confirmed hand-crossing between two adjacent zones."""
    from_zone: str
    to_zone: str
    timestamp: float
    holding_object: bool


class BurstTracker:
    """
    Tracks recent crossings to distinguish an isolated suspicious
    transfer from a mass-distribution burst (attendance, question
    papers, script collection).
    """
    def __init__(self, window_seconds: float, min_concurrent: int):
        self.window_seconds = window_seconds
        self.min_concurrent = min_concurrent
        self._recent: List[CrossingEvent] = []

    def add(self, crossing: CrossingEvent):
        self._recent.append(crossing)
        self._prune(crossing.timestamp)

    def _prune(self, now: float):
        self._recent = [
            c for c in self._recent
            if (now - c.timestamp) <= self.window_seconds
        ]

    def is_burst_in_progress(self, now: float) -> bool:
        """
        True if enough DIFFERENT zone-pairs have crossed within the
        window to look like synchronised mass distribution rather than
        an isolated pass.
        """
        self._prune(now)
        distinct_pairs = {
            tuple(sorted((c.from_zone, c.to_zone))) for c in self._recent
        }
        return len(distinct_pairs) >= self.min_concurrent


# ============================================================
# Object Passing Detection Module
# ============================================================

class ObjectPassingModule:
    """
    Usage (matches phone_detection.py / gaze_detection.py /
    posture_analysis.py exactly):

        module = ObjectPassingModule(config, alert_queue, camera_id="cam_0")
        module.start()
        # Feed frames:
        module.put_frame(frame, frame_number)
        # Stop:
        module.stop()
    """
    def __init__(self, config: dict, alert_queue, camera_id: str = "cam_0"):
        self.config = config
        self.alert_queue = alert_queue
        self.camera_id = camera_id
        cfg = config['object_passing']

        self._object_conf = cfg['object_confidence_threshold']
        self._frame_skip = cfg['frame_skip']
        self._base_score = cfg['base_score']
        self._boundary_margin = cfg['boundary_margin_pixels']
        self._object_proximity = cfg['object_proximity_pixels']
        self._model_path = cfg['model_path']
        self._hand_detection_conf = cfg['hand_detection_confidence']
        self._hand_tracking_conf = cfg['hand_tracking_confidence']

        self._grace_window_seconds = cfg['grace_window_seconds']
        self._burst_tracker = BurstTracker(
            window_seconds=cfg['burst_detection_window_seconds'],
            min_concurrent=cfg['burst_min_concurrent_crossings'],
        )

        self._module_start_time = time.time()
        self._min_zone_separation = cfg['min_zone_separation']
        self._max_hand_movement_px = cfg['max_hand_movement_pixels']

        # Each module has its OWN frame queue (Section 3.13)
        max_q = config['queues']['max_size']
        self.frame_queue: "queue.Queue" = queue.Queue(maxsize=max_q)

        # Threading — matches phone_detection.py / gaze_detection.py /
        # posture_analysis.py exactly
        self._thread = threading.Thread(target=self._run, daemon=True,
                                         name="ObjectPassingThread")
        self._stop_event = threading.Event()

        # Model + zone tracker — loaded/created in thread to avoid
        # blocking the main thread, and lazily on first frame respectively
        # (same pattern as the other three modules)
        self._model = None
        self._hands = None
        self._zone_tracker: Optional[CandidateZoneTracker] = None
        self._zone_separation_reliable = True  # finalised once zone_tracker exists
        self._frame_width = 0
        self._frame_height = 0

        self._last_hand_positions: List[Tuple[float, float, str]] = []

        # Stats
        self.frames_processed = 0
        self.detections_total = 0
        self._start_time = None
        self._frames_actually_processed = 0
        self._total_crossings_detected = 0
        self._total_events_emitted = 0
        self._total_events_suppressed_burst = 0
        self._total_events_suppressed_grace = 0
        self._total_events_suppressed_zone_separation = 0
        self._frames_skipped = 0
        self._total_hand_observations = 0

        # Last frame's events — kept for draw_detections() during
        # isolation testing, since events are now pushed to alert_queue
        # rather than returned directly (matches the other three
        # modules' architecture)
        self._last_frame_events: List[DetectionEvent] = []

        logger.info(f"[ObjectPassing] Module initialised — camera: {camera_id}")

    def start(self):
        """Start the detection thread."""
        self._start_time = time.time()
        self._thread.start()
        logger.info("[ObjectPassing] Thread started")

    def stop(self):
        """Signal the thread to stop and wait for it."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info(f"[ObjectPassing] Stopped. "
                    f"Processed {self.frames_processed} frames, "
                    f"{self.detections_total} detections total.")

    def put_frame(self, frame: np.ndarray, frame_number: int):
        """
        Add a frame to this module's queue.
        If queue is full, drop oldest frame to stay real-time (Section 3.13).
        """
        try:
            self.frame_queue.put_nowait((frame, frame_number))
        except queue.Full:
            try:
                self.frame_queue.get_nowait()  # Drop oldest
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait((frame, frame_number))
            except queue.Full:
                pass

    # --------------------------------------------------------
    # Internal — runs in dedicated thread
    # --------------------------------------------------------
    def _run(self):
        """Main detection loop — runs in its own thread."""
        self._load_models()

        while not self._stop_event.is_set():
            try:
                frame, frame_number = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_frame(frame, frame_number)
            except Exception as e:
                logger.error(f"[ObjectPassing] Frame {frame_number} error: {e}")

            self.frames_processed += 1

    def _load_models(self):
        """Load YOLO + MediaPipe Hands. Called once when thread starts."""
        try:
            logger.info(f"[ObjectPassing] Loading model: {self._model_path}")
            self._model = YOLO(self._model_path)
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=8,
                min_detection_confidence=self._hand_detection_conf,
                min_tracking_confidence=self._hand_tracking_conf,
            )
            logger.info("[ObjectPassing] Models loaded successfully")
        except Exception as e:
            logger.error(f"[ObjectPassing] Failed to load models: {e}")
            self._stop_event.set()

    def _init_zone_tracker_if_needed(self, frame: np.ndarray):
        if self._zone_tracker is not None:
            return
        h, w = frame.shape[:2]
        self._frame_width, self._frame_height = w, h
        self._zone_tracker = CandidateZoneTracker(w, h)

        # --- Minimum zone separation guard (see module docstring) ---
        cell_fraction_of_width = self._zone_tracker.cell_w / w
        self._zone_separation_reliable = cell_fraction_of_width >= self._min_zone_separation
        if not self._zone_separation_reliable:
            logger.warning(
                f"[ObjectPassing] Grid cell width ({self._zone_tracker.cell_w:.0f}px, "
                f"{cell_fraction_of_width:.2%} of frame) is below "
                f"min_zone_separation ({self._min_zone_separation:.2%}). "
                f"Candidate zones are too close together for reliable "
                f"crossing detection at this grid resolution — scored "
                f"object_passing events are DISABLED until grid_cols/"
                f"grid_rows are reduced or the camera's field of view is "
                f"widened to space candidates further apart per cell."
            )

    # --------------------------------------------------------
    def _in_grace_window(self, now: float) -> bool:
        return (now - self._module_start_time) < self._grace_window_seconds

    def _get_adjacent_zone_pairs_boundary(self, zone_a: str, zone_b: str) -> bool:
        """True if two zone IDs (e.g. 'R2C3', 'R2C4') are grid neighbours."""
        try:
            row_a, col_a = self._parse_zone(zone_a)
            row_b, col_b = self._parse_zone(zone_b)
        except (ValueError, IndexError):
            return False
        return (abs(row_a - row_b) + abs(col_a - col_b)) == 1

    @staticmethod
    def _parse_zone(zone_id: str) -> Tuple[int, int]:
        # zone_id format: "R{row}C{col}"
        r_part, c_part = zone_id[1:].split('C')
        return int(r_part), int(c_part)

    def _find_matching_last_hand(self, x: float, y: float) -> Optional[str]:
        """
        Find the closest hand position from the PREVIOUS processed frame,
        within max_hand_movement_px. Returns that hand's zone_id if found,
        else None (no plausible match = treat as a newly-appeared hand,
        not a crossing).
        """
        best_zone = None
        best_dist = self._max_hand_movement_px
        for (px, py, pzone) in self._last_hand_positions:
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist <= best_dist:
                best_dist = dist
                best_zone = pzone
        return best_zone

    # --------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, frame_number: int):
        """
        1. Initialise zone tracker on first frame (matches phone/gaze/
           posture pattern — see main.py's candidate-ID-consistency note)
        2. Run YOLO to get object boxes (paper/pen/phone placeholder)
        3. Run MediaPipe Hands to get wrist positions
        4. For each hand, determine its current zone + whether an
           object is near the wrist ("holding")
        5. Compare to that hand's last-seen zone via nearest-neighbour
           matching — if changed AND the new/old zones are adjacent
           → record a CrossingEvent
        6. Feed every crossing into the burst tracker regardless of
           holding_object (needed so burst detection has visibility
           into the full crossing pattern, not just the ones that
           would otherwise score)
        7. Only crossings where holding_object=True are candidates for
           a scored DetectionEvent — an empty hand crossing zones is
           not evidence of anything
        8. Apply zone-separation, grace-window, and burst suppression,
           then push directly to the shared alert_queue (matches the
           other three modules — no return value)
        """
        self._init_zone_tracker_if_needed(frame)
        now = time.time()

        # Frame skip — this module runs BOTH YOLO and MediaPipe Hands per
        # processed frame, making it the heaviest of the four modules on
        # CPU. Skip early, before either model runs.
        if self.frames_processed % self._frame_skip != 0:
            self._frames_skipped += 1
            return

        self._frames_actually_processed += 1
        self._last_frame_events = []  # reset — isolation-test display only

        object_boxes = self._detect_objects(frame)
        hand_positions = self._detect_hands(frame)
        self._total_hand_observations += len(hand_positions)

        current_hand_positions: List[Tuple[float, float, str]] = []

        for wrist_x, wrist_y in hand_positions:
            zone_id = self._zone_tracker.get_candidate_id(wrist_x, wrist_y)

            holding_object = self._is_holding_object(
                wrist_x, wrist_y, object_boxes
            )

            last_zone = self._find_matching_last_hand(wrist_x, wrist_y)
            current_hand_positions.append((wrist_x, wrist_y, zone_id))

            # A crossing is proven by: zone changed + the two zones are
            # adjacent + the movement was within max_hand_movement_px
            # (already enforced by _find_matching_last_hand's distance
            # cap). See git history for why the earlier extra
            # "current position near boundary line" check was removed —
            # it was silently discarding valid crossings under frame_skip.
            if (last_zone is not None and last_zone != zone_id and
                    self._get_adjacent_zone_pairs_boundary(last_zone, zone_id)):

                self._total_crossings_detected += 1
                crossing = CrossingEvent(
                    from_zone=last_zone, to_zone=zone_id,
                    timestamp=now, holding_object=holding_object,
                )
                self._burst_tracker.add(crossing)

                if holding_object:
                    event = self._build_event(crossing, frame_number, now)
                    if event is not None:
                        try:
                            self.alert_queue.put_nowait(event)
                        except queue.Full:
                            logger.warning("[ObjectPassing] Alert queue full — event dropped")
                        self.detections_total += 1
                        self._last_frame_events.append(event)

        self._last_hand_positions = current_hand_positions

    # --------------------------------------------------------
    def _build_event(self, crossing: CrossingEvent, frame_number: int,
                      now: float) -> Optional[DetectionEvent]:
        """Apply zone-separation, grace-window, and burst suppression,
        then build the event."""
        if not self._zone_separation_reliable:
            self._total_events_suppressed_zone_separation += 1
            logger.debug(
                f"[ObjectPassing] Crossing {crossing.from_zone}->"
                f"{crossing.to_zone} suppressed (zone separation below "
                f"min_zone_separation — see startup warning)"
            )
            return None

        if self._in_grace_window(now):
            self._total_events_suppressed_grace += 1
            logger.debug(
                f"[ObjectPassing] Crossing {crossing.from_zone}->"
                f"{crossing.to_zone} suppressed (grace window)"
            )
            return None

        if self._burst_tracker.is_burst_in_progress(now):
            self._total_events_suppressed_burst += 1
            logger.debug(
                f"[ObjectPassing] Crossing {crossing.from_zone}->"
                f"{crossing.to_zone} suppressed (mass-distribution burst)"
            )
            return None

        self._total_events_emitted += 1
        # Passing target candidate is the RECEIVING zone — the concern
        # is material entering a candidate's zone from a neighbour.
        return DetectionEvent(
            module="object_passing",
            candidate_id=crossing.to_zone,
            behaviour_type="object_passing",
            confidence=1.0,   # binary detection, no continuous confidence
                               # score in this design — see limitations
            base_score=self._base_score,
            weighted_score=self._base_score,
            bbox=(0, 0, 0, 0),   # not a single fixed box; boundary-region
                                  # event rather than an object box
            frame_number=frame_number,
            timestamp=now,
            duration_seconds=0.0,
            requires_persistence=False,
        )

    # --------------------------------------------------------
    def _detect_objects(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        results = self._model(frame, conf=self._object_conf, imgsz=320, verbose=False)
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2, y2))
        return boxes

    def _detect_hands(self, frame: np.ndarray) -> List[Tuple[float, float]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        wrists = []
        if results.multi_hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
                wrists.append((wrist.x * w, wrist.y * h))
        return wrists

    def _is_holding_object(self, wrist_x: float, wrist_y: float,
                            object_boxes: List[Tuple[int, int, int, int]]) -> bool:
        for (x1, y1, x2, y2) in object_boxes:
            box_cx, box_cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist = ((wrist_x - box_cx) ** 2 + (wrist_y - box_cy) ** 2) ** 0.5
            if dist <= self._object_proximity:
                return True
        return False

    # NOTE: _near_boundary() was removed — it was an overly strict extra
    # gate on crossing detection that became counterproductive once
    # frame_skip was introduced (see process_frame comment above for the
    # full reasoning). Zone adjacency + a bounded max movement distance
    # between samples is sufficient evidence of a genuine crossing.

    # --------------------------------------------------------
    def get_stats(self) -> dict:
        """Matches phone_detection.py / gaze_detection.py / posture_analysis.py's
        get_stats() shape (frames_processed, fps, queue_size), plus this
        module's own additional diagnostic counters."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        return {
            'frames_processed': self.frames_processed,
            'detections_total': self.detections_total,
            'fps': round(fps, 2),
            'queue_size': self.frame_queue.qsize(),
            'frames_actually_processed': self._frames_actually_processed,
            'frames_skipped': self._frames_skipped,
            'total_hand_observations': self._total_hand_observations,
            'total_crossings_detected': self._total_crossings_detected,
            'events_emitted': self._total_events_emitted,
            'suppressed_grace_window': self._total_events_suppressed_grace,
            'suppressed_burst': self._total_events_suppressed_burst,
            'suppressed_zone_separation': self._total_events_suppressed_zone_separation,
            'zone_separation_reliable': self._zone_separation_reliable,
            'in_grace_window': self._in_grace_window(time.time()),
        }

    def get_last_frame_events(self) -> List[DetectionEvent]:
        """
        Isolation-test convenience only — events are pushed directly to
        alert_queue during normal operation (matching the other three
        modules), but the isolation test script also wants to draw
        boxes for whatever fired on the most recently processed frame
        without needing to separately drain the queue itself.
        """
        return self._last_frame_events

    def draw_detections(self, frame: np.ndarray,
                         events: List[DetectionEvent]) -> np.ndarray:
        """Isolation-test visualisation — draws grid + flagged zones."""
        out = frame.copy()
        if self._zone_tracker is None:
            return out  # not yet initialised (no frame processed yet)

        cell_w = int(self._zone_tracker.cell_w)
        cell_h = int(self._zone_tracker.cell_h)

        # Faint grid overlay
        for c in range(1, self._zone_tracker.grid_cols):
            cv2.line(out, (c * cell_w, 0), (c * cell_w, self._frame_height),
                      (80, 80, 80), 1)
        for r in range(1, self._zone_tracker.grid_rows):
            cv2.line(out, (0, r * cell_h), (self._frame_width, r * cell_h),
                      (80, 80, 80), 1)

        for event in events:
            row, col = self._parse_zone(event.candidate_id)
            x1, y1 = (col - 1) * cell_w, (row - 1) * cell_h
            x2, y2 = col * cell_w, row * cell_h
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(out, f"PASS -> {event.candidate_id}", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return out
