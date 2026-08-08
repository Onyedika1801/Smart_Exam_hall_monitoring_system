"""
phone_detection.py
==================
Smart Exam Hall Monitoring System — Phone Detection Module
Chapter 3 Reference: Section 3.7

Responsibilities:
- Runs as its own dedicated thread
- Reads frames from its own private frame queue
- Runs YOLOv8n inference at conf=0.55
- Applies confidence weighting per Table 3.7
- Outputs DetectionEvent objects to the shared alert queue
- NO scoring logic here — that belongs to alert_manager only

Architecture note (Section 3.13):
  Camera thread → [phone_frame_queue] → PhoneDetector thread → [alert_queue]
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import queue

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# Detection Event — output contract for all modules
# alert_manager reads these from the shared alert queue
# ============================================================
@dataclass
class DetectionEvent:
    """
    Standardised output from any detection module.
    alert_manager uses these to compute candidate scores.
    """
    module: str                    # "phone_detection"
    candidate_id: str              # Grid position e.g. "R2C3"
    behaviour_type: str            # "phone_detected"
    confidence: float              # YOLOv8 confidence score
    base_score: float              # From config — 35 for phone
    weighted_score: float          # base_score * confidence_weight
    bbox: tuple                    # (x1, y1, x2, y2) in pixels
    frame_number: int
    timestamp: float               # time.time()
    camera_id: str = "cam_0"
    duration_seconds: float = 0.0  # For phone: always 0 (no persistence)
    requires_persistence: bool = False


# ============================================================
# Candidate Zone Tracker
# Maps frame regions to candidate IDs (grid-based)
# ============================================================
class CandidateZoneTracker:
    """
    Divides the frame into a grid and assigns candidate IDs
    based on bounding box centre position.
    Candidates are tracked by spatial position, not identity
    (Section 3.15 ethical considerations).
    """
    def __init__(self, frame_width: int, frame_height: int,
                 grid_cols: int = 5, grid_rows: int = 4):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_cols = grid_cols
        self.grid_rows = grid_rows
        self.cell_w = frame_width / grid_cols
        self.cell_h = frame_height / grid_rows

    def get_candidate_id(self, x_centre: float, y_centre: float) -> str:
        """Returns grid-based candidate ID e.g. 'R2C3'"""
        col = min(int(x_centre / self.cell_w), self.grid_cols - 1)
        row = min(int(y_centre / self.cell_h), self.grid_rows - 1)
        return f"R{row+1}C{col+1}"


# ============================================================
# Confidence Weight Calculator (Chapter 3 Table 3.7)
# ============================================================
def calculate_confidence_weight(confidence: float, config: dict) -> float:
    """
    Returns the score weight multiplier based on YOLOv8 confidence.

    Table 3.7:
      >= 0.90  → 1.00 (100% of base score)
      >= 0.75  → 0.80 (80% of base score)
      >= 0.60  → 0.60 (60% of base score)
      < 0.55   → 0.00 (ignored — should not reach here)
    """
    weights_cfg = config['phone_detection']['confidence_weights']
    tier1 = weights_cfg['tier_1_threshold']  # 0.90
    tier2 = weights_cfg['tier_2_threshold']  # 0.75
    tier3 = weights_cfg['tier_3_threshold']  # 0.60

    if confidence >= tier1:
        return 1.00
    elif confidence >= tier2:
        return 0.80
    elif confidence >= tier3:
        return 0.60
    else:
        return 0.00


# ============================================================
# Phone Detection Module
# ============================================================
class PhoneDetectionModule:
    """
    YOLOv8n-based mobile phone detector.

    Usage:
        module = PhoneDetectionModule(config, alert_queue, camera_id="cam_0")
        module.start()
        # Feed frames:
        module.frame_queue.put((frame, frame_number))
        # Stop:
        module.stop()
    """

    def __init__(self, config: dict, alert_queue: queue.Queue,
                 camera_id: str = "cam_0"):
        self.config = config
        self.alert_queue = alert_queue
        self.camera_id = camera_id

        # Each module has its OWN frame queue (Section 3.13)
        max_q = config['queues']['max_size']
        self.frame_queue = queue.Queue(maxsize=max_q)

        # Threading
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="PhoneDetectionThread")
        self._stop_event = threading.Event()

        # Stats
        self.frames_processed = 0
        self._frames_actually_processed = 0
        self.detections_total = 0
        self._start_time = None

        # Model (loaded in thread to avoid blocking main thread)
        self._model = None

        # Candidate zone tracker
        ph_cfg = config['phone_detection']
        self._conf_threshold = ph_cfg['confidence_threshold']
        # NOTE: frame_skip did not exist in this module until live
        # integrated testing (main.py) showed phone_detection's queue
        # permanently backlogged (25-30/30 for almost an entire ~9 minute
        # run) while running under full four-module CPU load, with fps
        # stuck around 1.8 -- far below the camera's actual frame rate.
        # A frame_skip value HAD been tuned earlier, but only inside the
        # standalone isolation test script (test_phone_detection.py),
        # never in this actual module class, so main.py was running
        # YOLO on every single frame with no skip at all. Defaults to 1
        # (no skip) if not present in an older config.yaml, to avoid
        # breaking anything relying on the previous unconditional behaviour.
        self._frame_skip = ph_cfg.get('frame_skip', 1)

        # Per-candidate re-alert cooldown. Confirmed via live testing
        # (candidate score reaching 1616+ within seconds) that this
        # module previously emitted a brand-new DetectionEvent on EVERY
        # processed frame containing a qualifying phone, with no
        # throttling at all -- unlike gaze_detection (5s cooldown) and
        # posture_analysis (8s cooldown), which both already limit how
        # often an ongoing deviation re-fires. A continuously visible
        # phone at ~3-4 processed frames/second, each worth up to ~40+
        # points, explains runaway scores reaching the thousands within
        # seconds. This does NOT reintroduce a persistence delay before
        # the FIRST alert (Section 3.7.3: a single qualifying frame is
        # still sufficient to alert) -- it only throttles REPEAT events
        # for a candidate who remains flagged across many consecutive
        # processed frames.
        self._re_alert_cooldown = ph_cfg.get('re_alert_cooldown_seconds', 5.0)
        self._last_event_time: dict = {}
        self._base_score = ph_cfg['base_score']
        self._model_path = ph_cfg['model_path']
        self._imgsz = ph_cfg['inference_image_size']

        # --- Generic object detector (supplements the fine-tuned model —
        # see config.yaml comment for the bold-hold-pose rationale) ---
        self._use_generic_detector = ph_cfg.get('use_generic_object_detector', False)
        self._generic_model_path = ph_cfg.get('generic_model_path', 'yolov8n.pt')
        self._generic_conf_threshold = ph_cfg.get('generic_object_confidence_threshold', 0.45)
        self._generic_classes = set(ph_cfg.get('generic_object_classes', ['cell phone']))
        self._generic_model = None  # loaded in _load_model() if enabled

        # Zone tracker — initialised on first frame
        self._zone_tracker: Optional[CandidateZoneTracker] = None

        logger.info(f"[PhoneDetection] Module initialised — camera: {camera_id}")

    def start(self):
        """Start the detection thread."""
        self._start_time = time.time()
        self._thread.start()
        logger.info("[PhoneDetection] Thread started")

    def stop(self):
        """Signal the thread to stop and wait for it."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        logger.info(f"[PhoneDetection] Stopped. "
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

    def get_stats(self) -> dict:
        """Return processing statistics."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self.frames_processed / elapsed if elapsed > 0 else 0
        return {
            'frames_processed': self.frames_processed,
            'detections_total': self.detections_total,
            'fps': round(fps, 2),
            'queue_size': self.frame_queue.qsize()
        }

    # --------------------------------------------------------
    # Internal — runs in dedicated thread
    # --------------------------------------------------------
    def _run(self):
        """Main detection loop — runs in its own thread."""
        self._load_model()

        while not self._stop_event.is_set():
            try:
                frame, frame_number = self.frame_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_frame(frame, frame_number)
            except Exception as e:
                logger.error(f"[PhoneDetection] Frame {frame_number} error: {e}")

            self.frames_processed += 1

    def _load_model(self):
        """Load YOLOv8n model. Called once when thread starts.

        Explicitly detects and requests GPU (CUDA) if available, rather
        than relying on library-default device selection, so the chosen
        device is visible in the log and reproducible across machines.
        Falls back to CPU automatically if no GPU is present -- this
        module still runs correctly on CPU-only hardware (e.g. a
        marker's machine at defence), just slower, matching the original
        CPU-only design baseline documented in Chapter 3/4.
        """
        try:
            import torch
            from ultralytics import YOLO
            logger.info(f"[PhoneDetection] Loading model: {self._model_path}")
            self._model = YOLO(self._model_path)

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self._model.to('cuda')
                logger.info(f"[PhoneDetection] GPU detected ({device_name}) -- "
                            f"running inference on CUDA")
            else:
                logger.info("[PhoneDetection] No GPU detected -- running on CPU")

            if self._use_generic_detector:
                logger.info(f"[PhoneDetection] Loading generic COCO detector: "
                            f"{self._generic_model_path} (classes: "
                            f"{sorted(self._generic_classes)})")
                self._generic_model = YOLO(self._generic_model_path)
                if torch.cuda.is_available():
                    self._generic_model.to('cuda')
                logger.info("[PhoneDetection] Generic detector loaded")

            logger.info("[PhoneDetection] Model loaded successfully")
        except Exception as e:
            logger.error(f"[PhoneDetection] Failed to load model: {e}")
            self._stop_event.set()

    def _process_frame(self, frame: np.ndarray, frame_number: int):
        """
        Run YOLOv8n inference on a single frame.
        For each phone detection above confidence threshold,
        create a DetectionEvent and push to alert queue.
        """
        if self._model is None:
            return

        # Skip early, before running YOLO — see frame_skip comment in
        # __init__ for why this exists now.
        if self.frames_processed % self._frame_skip != 0:
            return
        self._frames_actually_processed += 1

        h, w = frame.shape[:2]

        # Initialise zone tracker on first frame
        if self._zone_tracker is None:
            self._zone_tracker = CandidateZoneTracker(w, h)

        # 1. Fine-tuned phone detector — primary source, trained
        # specifically on this project's dataset.
        detections = []  # list of (x1, y1, x2, y2, confidence, source)
        results = self._model(
            frame,
            conf=self._conf_threshold,
            imgsz=self._imgsz,
            verbose=False
        )
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self._conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append((x1, y1, x2, y2, confidence, "fine_tuned"))

        # 2. Generic COCO detector — supplementary source, catches some
        # bold-hold poses the fine-tuned model misses (see config.yaml
        # comment). Confidence comes from a different model/distribution
        # than the fine-tuned one; Table 3.7 weighting is still applied
        # the same way regardless of source (see config.yaml note).
        if self._use_generic_detector and self._generic_model is not None:
            generic_results = self._generic_model(
                frame, conf=self._generic_conf_threshold,
                imgsz=self._imgsz, verbose=False
            )
            for result in generic_results:
                if result.boxes is None:
                    continue
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = names.get(cls_id, "")
                    if cls_name not in self._generic_classes:
                        continue
                    confidence = float(box.conf[0])
                    if confidence < self._generic_conf_threshold:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append((x1, y1, x2, y2, confidence, "generic_coco"))

        for (x1, y1, x2, y2, confidence, source) in detections:
            x_centre = (x1 + x2) / 2
            y_centre = (y1 + y2) / 2

            # Assign candidate ID from grid position
            candidate_id = self._zone_tracker.get_candidate_id(x_centre, y_centre)

            # Per-candidate re-alert cooldown -- see __init__ note.
            # A single qualifying detection is still sufficient to
            # alert immediately (no persistence delay), but repeat
            # detections of the SAME candidate within the cooldown
            # window are skipped entirely rather than each creating
            # a fresh, fully-scored event. This also naturally
            # de-duplicates the (rare) case where BOTH the fine-tuned
            # and generic models detect the same physical phone in
            # the same processed frame -- the first one sets
            # _last_event_time, so the second is skipped here.
            now = time.time()
            last_time = self._last_event_time.get(candidate_id)
            if last_time is not None and (now - last_time) < self._re_alert_cooldown:
                continue

            # Apply confidence weight (Table 3.7)
            conf_weight = calculate_confidence_weight(confidence, self.config)
            if conf_weight == 0.0:
                continue  # Below minimum useful threshold

            weighted_score = self._base_score * conf_weight

            # Build detection event
            event = DetectionEvent(
                module="phone_detection",
                candidate_id=candidate_id,
                behaviour_type="phone_detected",
                confidence=confidence,
                base_score=self._base_score,
                weighted_score=weighted_score,
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                frame_number=frame_number,
                timestamp=now,
                camera_id=self.camera_id,
                duration_seconds=0.0,
                requires_persistence=False  # Phone = immediate alert (Section 3.7.3)
            )
            self._last_event_time[candidate_id] = now

            # Push to shared alert queue for alert_manager
            try:
                self.alert_queue.put_nowait(event)
            except queue.Full:
                logger.warning("[PhoneDetection] Alert queue full — event dropped")

            self.detections_total += 1
            logger.debug(
                f"[PhoneDetection] Phone detected ({source}) — "
                f"candidate: {candidate_id}, "
                f"conf: {confidence:.3f}, "
                f"weighted_score: {weighted_score:.1f}, "
                f"frame: {frame_number}"
            )

    def draw_detections(self, frame: np.ndarray,
                        events: list) -> np.ndarray:
        """
        Draw bounding boxes on frame for dashboard display.
        Call this in the camera/display thread, not detection thread.
        """
        frame_out = frame.copy()
        for event in events:
            if event.module != "phone_detection":
                continue
            x1, y1, x2, y2 = event.bbox
            conf = event.confidence

            # Colour by confidence tier
            if conf >= 0.90:
                colour = (0, 0, 255)    # Red — high confidence
            elif conf >= 0.75:
                colour = (0, 140, 255)  # Orange — medium-high
            else:
                colour = (0, 200, 255)  # Yellow — medium

            cv2.rectangle(frame_out, (x1, y1), (x2, y2), colour, 2)
            label = f"PHONE {conf:.2f} [{event.candidate_id}]"
            cv2.putText(frame_out, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        return frame_out
