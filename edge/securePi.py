#!/usr/bin/env python3
"""
SecurePi — Raspberry Pi AI Camera (IMX500) edge security monitor (FlowGuard).

Runs object detection *on the IMX500 sensor* and applies three independent
kinds of logic to the detections:

* **person** — tracked with stable ids, used to associate an owner with a
  nearby unattended object;
* **unattended objects** (backpack, handbag, suitcase) — tracked across
  frames; once the owner leaves, an unattended timer runs and an alert fires
  after a configurable threshold;
* **pests** (rat, mouse) — a *separate* detector with its own confidence
  threshold, confirmation (by duration or by consecutive frames) and alert
  cooldown. Pests never require an owner and never enter the object/owner
  association logic.

Detections are read from the neural-network output tensors carried in the
Picamera2 frame metadata via the IMX500 helper (`get_outputs` +
`convert_inference_coords`) — matching the official Raspberry Pi IMX500 object
detection demo. Runs with a live preview window by default, or fully headless
(`--headless`) for SSH / systemd use. Alert snapshots and a CSV event log are
written under a configurable runtime directory.

Settings come from preset files in the presets/ folder (required):
presets/common.args is applied automatically as the shared base, then the
named preset's overrides, then any command-line flags.

With ``--stream``, the same single process (and the same single Picamera2
instance) also serves the annotated frames as an MJPEG HTTP stream
(``/video_feed``) plus a JSON health endpoint (``/health``) — independent of,
and in addition to, the FlowGuard cloud alert integration below.

Examples
--------
    python edge/securePi.py @edge/presets/lobby.args
    python edge/securePi.py @edge/presets/kitchen.args --headless --unattended-time 60
    python edge/securePi.py @edge/presets/lobby.args --headless --stream --stream-port 8001
    python edge/securePi.py @edge/presets/lobby.args \
        --model models/imx500_custom_securepi.rpk \
        --labels models/labels.txt --headless
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import socketserver
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional, Any
import concurrent.futures

import cv2
import numpy as np

import base64
import urllib.error

try:
    from picamera2 import Picamera2
    from picamera2.devices import IMX500
    from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)
    _IMX500_AVAILABLE = True
except ImportError:  # let --help / imports work off-device
    Picamera2 = IMX500 = NetworkIntrinsics = postprocess_nanodet_detection = None  # type: ignore[assignment]
    _IMX500_AVAILABLE = False

# Optional FlowGuard cloud edge client (sibling module in edge/). Guarded so the
# monitor still imports and runs fully offline if the integration module is
# missing — the Pi NEVER calls WhatsApp; it only POSTs events to FlowGuard.
try:
    from flowguard_api import FlowGuardApiClient, build_event_id, mask_url
    _FLOWGUARD_AVAILABLE = True
except ImportError:  # pragma: no cover - integration is optional
    FlowGuardApiClient = None  # type: ignore[assignment]
    build_event_id = None  # type: ignore[assignment]
    mask_url = None  # type: ignore[assignment]
    _FLOWGUARD_AVAILABLE = False

try:
    from sensor_bridge import SensorBridge, start_sensor_reader_thread, _default_serial_port
    _SENSOR_BRIDGE_AVAILABLE = True
except ImportError:
    try:
        from edge.sensor_bridge import SensorBridge, start_sensor_reader_thread, _default_serial_port
        _SENSOR_BRIDGE_AVAILABLE = True
    except ImportError:
        SensorBridge = None  # type: ignore[assignment]
        start_sensor_reader_thread = None  # type: ignore[assignment]
        _default_serial_port = lambda: None  # type: ignore[assignment]
        _SENSOR_BRIDGE_AVAILABLE = False


LOGGER = logging.getLogger("securepi")


def _enqueue_flowguard(client, config, *, event_type, alert_type, object_class,
                       confidence, duration_seconds, track_id, snapshot_path, event_id,
                       person_name=None, identity_status=None, person_role=None,
                       severity=None, sensor_metadata=None):
    """Best-effort push of a detection event to the FlowGuard outbox (non-blocking)."""
    if client is None or not getattr(client, "enabled", False):
        return
    try:
        zone_name, camera_location = _flowguard_zone_camera(client, config)
        event = client.build_event(
            event_type=event_type,
            alert_type=alert_type,
            zone_name=zone_name,
            camera_location=camera_location,
            object_class=object_class,
            severity=severity,
            confidence=confidence,
            duration_seconds=duration_seconds,
            track_id=track_id,
            person_name=person_name,
            identity_status=identity_status,
            person_role=person_role,
            sensor_metadata=sensor_metadata,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            event_id=event_id,
        )
        client.enqueue_event(event)
    except Exception as exc:  # pragma: no cover - defensive; cloud must not break local
        LOGGER.warning("FlowGuard enqueue failed (non-fatal): %s", exc)

# Repo root = the folder that contains edge/. Used to resolve preset/model/label
# and runtime paths reliably regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent

# Default network shipped by `sudo apt install imx500-all` (COCO SSD MobileNetV2).
DEFAULT_MODEL = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"

# The three functional label categories. Labels come back as lower-case strings.
# NOTE: rat/mouse are *pests*, never unattended objects. With the default COCO
# model "mouse" means a *computer mouse* and "rat" is absent — pest detection is
# meaningful only with the custom FlowGuard model + models/labels.txt.
DEFAULT_PERSON_LABELS = {"person"}
DEFAULT_UNATTENDED_OBJECT_LABELS = {"backpack", "handbag", "suitcase"}
DEFAULT_PEST_LABELS = {"rat", "mouse"}
# Backwards-compatible alias (older code/imports referred to "bag" labels).
DEFAULT_BAG_LABELS = DEFAULT_UNATTENDED_OBJECT_LABELS

# Default root for all runtime output (snapshots, logs, alerts). Anchored to the
# repo so it is stable no matter where the process is launched from; override
# with --runtime-dir (e.g. an absolute path on the Pi for a systemd service).
DEFAULT_RUNTIME_DIR = REPO_ROOT / "runtime"

# Global executor for non-blocking snapshot saving / event logging.
SNAPSHOT_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# BGR colours (OpenCV order).
COLOR_PERSON = (0, 255, 0)
COLOR_ATTENDED = (255, 255, 0)
COLOR_WARNING = (0, 165, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_PEST = (255, 0, 255)   # magenta — visually distinct from the bag ALERT red
COLOR_HUD = (0, 255, 0)


@dataclass
class Config:
    """Tunable parameters for the monitor."""

    unattended_time_sec: float = 120.0   # alert after this long unattended
    stationary_radius: float = 120.0     # association radius: px a bag may move between
                                         # detections and still match the same track
    person_proximity_px: float = 150.0   # px within which the owner attends a bag
    owner_claim_sec: float = 3.0         # after a bag appears, a person near it within this
                                         # window is adopted as its OWNER; only the owner
                                         # attending resets the timer (others are ignored)
    person_timeout_sec: float = 2.0      # drop a person track unseen for this long
    person_match_radius: float = 150.0   # px to associate a person to the same track
    track_timeout_sec: float = 10.0      # "coast" window: keep a track alive (and its
                                         # unattended timer running) through detection
                                         # dropouts; only drop it after this long unseen
    draw_grace_sec: float = 1.5          # stop drawing a coasting track's box once it has
                                         # been unseen this long (avoids lingering "ghost"
                                         # boxes); the track itself lives until the timeout
    min_confidence: float = 0.5          # ignore person/object detections below this score
    # Per-label overrides layered on top of min_confidence (person/object labels)
    # or pest_confidence (pest labels): a label present here replaces its
    # category's shared threshold; any label absent still falls back to the
    # category default. See --class-confidence.
    #
    # No "suitcase" entry: IMX500Detector.detect() remaps every "suitcase"
    # prediction to "backpack" right after classification (the class
    # confidently misfires on backpacks/handbags - see docs/MODELS.md), so a
    # detection can never carry that label here. backpack is raised to the
    # same 0.55 the old "suitcase" entry used - without it, the remapped
    # noisy detections only have to clear the flat 0.5 min_confidence floor,
    # which is what let them flood in as spurious backpack alerts (confirmed
    # in runtime/logs/events.csv: backpack alerts landing right at 0.53-0.56,
    # the exact range 0.55 used to filter out). mouse is raised on direct
    # evidence from a real deployment log (runtime/logs/events.csv): it
    # repeatedly triggered right at the old flat 0.5 floor (scores as low as
    # 0.44-0.56), consistent with noisy, borderline detections being fed to
    # the tracker. rat is intentionally left at the pest_confidence default:
    # its poor real-world hit rate traces to training data (the "rat" class
    # was trained on Hamster images, never a real rat -- see docs/MODELS.md),
    # which no confidence threshold can fix.
    class_confidence: dict[str, float] = field(default_factory=lambda: {
        "backpack": 0.55,
        "mouse": 0.55,
    })
    box_smoothing: float = 0.6           # weight of the newest detection when smoothing a
                                         # track's drawn box (1.0 = no smoothing); damps
                                         # frame-to-frame detector jitter on static objects
    frame_size: tuple[int, int] = (640, 480)
    headless: bool = False               # run without a preview window
    alert_cooldown_sec: float = 30.0     # seconds between repeat alerts per bag
    max_snapshots: int = 500             # keep at most this many snapshots; oldest are
                                         # deleted so long runs can't fill the SD card

    # --- Runtime data locations (all git-ignored) -------------------------
    runtime_dir: Path = field(default_factory=lambda: DEFAULT_RUNTIME_DIR)
    zone: Optional[str] = None           # preset/location tag recorded in the event log
    snapshot_dir: Optional[Path] = None  # override; default: <runtime>/snapshots[/<zone>]
    log_dir: Optional[Path] = None       # override; default: <runtime>/logs
    restricted_hours_start: str = field(
        default_factory=lambda: os.environ.get("RESTRICTED_HOURS_START", "08:00")
    )
    restricted_hours_end: str = field(
        default_factory=lambda: os.environ.get("RESTRICTED_HOURS_END", "23:00")
    )

    # --- IMX500 detector settings ----------------------------------------
    model_path: str = DEFAULT_MODEL
    labels_path: Optional[str] = None    # override the model's built-in labels
    iou: float = 0.65                    # NMS IoU threshold (nanodet models)
    max_detections: int = 10             # max detections (nanodet models)

    # --- Functional label categories -------------------------------------
    unattended_object_labels: set[str] = field(
        default_factory=lambda: set(DEFAULT_UNATTENDED_OBJECT_LABELS))
    person_labels: set[str] = field(default_factory=lambda: set(DEFAULT_PERSON_LABELS))
    pest_labels: set[str] = field(default_factory=lambda: set(DEFAULT_PEST_LABELS))

    # --- Pest detection (rat/mouse) — independent of the bag logic --------
    pest_confidence: float = 0.50        # separate confidence threshold for pests
    pest_confirmation_time: float = 2.0  # confirm a pest after it is visible this long (s);
                                         # set 0 to disable the duration criterion
    pest_confirmation_frames: int = 3    # OR confirm after this many consecutive frames;
                                         # set 0 to disable the frame-count criterion
    pest_alert_cooldown_sec: float = 30.0  # seconds between repeat alerts for the same pest
    pest_timeout_sec: float = 2.0        # drop a pest track (and reset confirmation) after
                                         # this long unseen
    pest_match_radius: float = 220.0     # px gate for matching a pest detection to its track
                                         # between frames. Deliberately separate from
                                         # stationary_radius (tuned for a stationary bag): a
                                         # moving rat/mouse displaces much further frame-to-
                                         # frame, and sharing the bag's tighter radius let a
                                         # real pest's track keep breaking before it could
                                         # accumulate pest_confirmation_frames/time -- i.e. a
                                         # genuinely present pest could go unconfirmed forever.

    # --- MJPEG streaming (off unless --stream is passed) ------------------
    stream_enabled: bool = False
    stream_host: str = "0.0.0.0"
    stream_port: int = 8001
    stream_fps: float = 8.0                # annotated frames published per second
    stream_quality: int = 70               # JPEG quality 1..100

    def confidence_for(self, label: str, category_default: float) -> float:
        """Effective confidence threshold for one label: class_confidence override,
        else the caller's category default (min_confidence or pest_confidence)."""
        return self.class_confidence.get(label, category_default)

    def __post_init__(self) -> None:
        # Derive snapshot/log dirs from runtime_dir unless explicitly overridden.
        if self.snapshot_dir is None:
            base = self.runtime_dir / "snapshots"
            self.snapshot_dir = base / self.zone if self.zone else base
        if self.log_dir is None:
            self.log_dir = self.runtime_dir / "logs"


@dataclass
class Detection:
    """A single object detection for the current frame (box in image pixels)."""

    label: str
    score: float
    box: tuple[int, int, int, int]  # x, y, w, h in pixels

    @property
    def centroid(self) -> tuple[float, float]:
        return box_centroid(self.box)


@dataclass
class TrackedBag:
    """State for an unattended-object candidate followed across frames."""

    bag_id: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    last_seen: float
    created_at: float                       # when the track was first registered
    score: float = 0.0                      # confidence of the latest matched detection
    label: str = "bag"                      # specific class (backpack/handbag/suitcase) when known
    owner_id: Optional[int] = None          # person id adopted as the bag's owner
    unattended_start: Optional[float] = None
    alerted: bool = False
    last_alert_time: float = 0.0
    flowguard_event_id: Optional[str] = None  # stable cloud event id, set on the first alert


@dataclass
class PersonTrack:
    """A person followed across frames, giving them a stable id for owner matching."""

    person_id: int
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    last_seen: float


@dataclass
class TrackedPest:
    """A pest (rat/mouse) followed across frames — no owner, its own timers."""

    pest_id: int
    label: str
    centroid: tuple[float, float]
    box: tuple[int, int, int, int]
    first_seen: float                       # when this continuous sighting started
    last_seen: float
    score: float = 0.0
    frames: int = 1                         # consecutive frames matched to this track
    alerted: bool = False
    last_alert_time: float = 0.0
    flowguard_event_id: Optional[str] = None  # stable cloud event id, set on the first alert


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def box_centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def calculate_iou(boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    return interArea / float(boxAArea + boxBArea - interArea)


def smooth_box(old: tuple[int, int, int, int], new: tuple[int, int, int, int],
               alpha: float) -> tuple[int, int, int, int]:
    """Blend consecutive boxes to damp detector jitter on the drawn box.

    ``alpha`` is the weight of the NEW detection (1.0 disables smoothing). If
    the new box barely overlaps the old one the object genuinely moved, so snap
    to the detection instead of dragging a laggy box across the scene.
    """
    if alpha >= 1.0 or calculate_iou(old, new) < 0.2:
        return new
    x, y, w, h = (round(o + alpha * (n - o)) for o, n in zip(old, new))
    return (x, y, w, h)


# Detections/tracks overlapping this much are the same physical object.
DEDUP_IOU = 0.45


def dedup_detections(detections: list[Detection]) -> list[Detection]:
    """Cross-label NMS: collapse detections of the same physical object.

    The sensor's SSD post-processing suppresses duplicates per class only, so
    one bag can come back as both "backpack" and "handbag" at nearly the same
    box in the same frame — which would spawn one track (and one alert box)
    per label. All labels in a single category mean the same thing here, so
    keep only the highest-confidence detection of any overlapping cluster.
    """
    keep: list[Detection] = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        if all(calculate_iou(det.box, k.box) < DEDUP_IOU for k in keep):
            keep.append(det)
    return keep


def split_detections(detections: list[Detection], config: Config
                     ) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Route detections into the three functional categories.

    People and objects use ``min_confidence``; pests use their own
    ``pest_confidence`` so rodents can be caught at a different threshold.
    Either category default can be overridden per-label via
    ``config.class_confidence`` (see ``Config.confidence_for``). The
    categories are disjoint by label set, so a rat/mouse can never fall into
    the unattended-object bucket.
    """
    persons = [d for d in detections
               if d.label in config.person_labels
               and d.score >= config.confidence_for(d.label, config.min_confidence)]
    objects = [d for d in detections
               if d.label in config.unattended_object_labels
               and d.score >= config.confidence_for(d.label, config.min_confidence)]
    pests = [d for d in detections
             if d.label in config.pest_labels
             and d.score >= config.confidence_for(d.label, config.pest_confidence)]
    return persons, objects, pests


def match_detections(detections: list[Detection], tracks: list,
                     iou_gate: float, dist_gate: float) -> dict[int, Any]:
    """Best-first global matching of this frame's detections to tracks.

    Considers every detection/track pair at once: IoU overlaps (>= iou_gate)
    always outrank centroid-distance fallbacks (<= dist_gate), and within each
    tier the strongest pair is assigned first. Unlike per-detection greedy
    matching this is independent of detection order, so one detection can't
    steal the track that another detection overlaps better.

    Returns {detection_index: track}.
    """
    pairs: list[tuple[int, float, int, int]] = []
    for di, det in enumerate(detections):
        for ti, track in enumerate(tracks):
            iou = calculate_iou(det.box, track.box)
            if iou >= iou_gate:
                pairs.append((1, iou, di, ti))
            else:
                d = distance(det.centroid, track.centroid)
                if d <= dist_gate:
                    pairs.append((0, -d, di, ti))
    pairs.sort(key=lambda p: (p[0], p[1]), reverse=True)

    matches: dict[int, Any] = {}
    used_tracks: set[int] = set()
    for _, _, di, ti in pairs:
        if di in matches or ti in used_tracks:
            continue
        matches[di] = tracks[ti]
        used_tracks.add(ti)
    return matches


class IMX500Detector:
    """Loads a network onto the IMX500 and turns frame metadata into Detections.

    Mirrors the parsing contract of the official Raspberry Pi
    `imx500_object_detection_demo.py` (output tensors -> boxes/scores/classes ->
    `convert_inference_coords`), so it works with the standard COCO models.

    Supported output formats:

    1. NanoDet outputs decoded by Picamera2.
    2. Standard 3-tensor post-processed detection output
       (boxes, scores, classes).
    3. Ultralytics IMX500 NMS output with four tensors
       (boxes, scores, classes, valid_detection_count).

    The fourth tensor is important: the network reserves 100 detection slots,
    but only the first valid_detection_count entries are real. Reading all
    padded slots creates many false bag tracks.
    """

    def __init__(self, config: Config) -> None:
        if not _IMX500_AVAILABLE:
            raise RuntimeError(
                "IMX500 support is unavailable. Install the Pi camera stack: "
                "`sudo apt install -y python3-picamera2 imx500-all`."
            )
        # Use the lowest of every effective threshold (the two category defaults
        # plus any class_confidence overrides, which may sit below their
        # category default) so no candidate is dropped here before
        # split_detections() applies its precise per-label gate.
        self.threshold = min([config.min_confidence, config.pest_confidence,
                              *config.class_confidence.values()])
        self.iou = config.iou
        self.max_detections = config.max_detections

        self.imx500 = IMX500(config.model_path)
        intrinsics = self.imx500.network_intrinsics
        if not intrinsics:
            intrinsics = NetworkIntrinsics()
            intrinsics.task = "object detection"
        elif intrinsics.task != "object detection":
            raise RuntimeError(f"Model '{config.model_path}' is not an object-detection network.")

        if config.labels_path:
            with open(config.labels_path, encoding="utf-8") as fh:
                intrinsics.labels = fh.read().splitlines()
        intrinsics.update_with_defaults()

        self.intrinsics = intrinsics
        self._labels = self._build_labels()

    @property
    def camera_num(self) -> int:
        return self.imx500.camera_num

    @property
    def inference_rate(self) -> Optional[float]:
        return self.intrinsics.inference_rate

    def show_progress(self) -> None:
        """Display the firmware-upload progress bar (first run can take ~30s)."""
        self.imx500.show_network_fw_progress_bar()

    def _build_labels(self) -> list[str]:
        labels = self.intrinsics.labels or []
        if self.intrinsics.ignore_dash_labels:
            labels = [lbl for lbl in labels if lbl and lbl != "-"]
        return labels

    def _label_for(self, idx: int) -> str:
        if 0 <= idx < len(self._labels):
            return self._labels[idx].lower()
        return str(idx)

    def detect(self, metadata: Any, picam2: Any) -> list[Detection]:
        """Parse the IMX500 output tensors for this frame into detections."""
        np_outputs = self.imx500.get_outputs(metadata, add_batch=True)

        if np_outputs is None:
            return []  # firmware still uploading or no inference this frame

        if not getattr(self, "_outputs_printed", False):
            self._outputs_printed = True

            output_summary = []

            for index, output in enumerate(np_outputs):
                array = np.asarray(output)

                output_summary.append(
                    {
                        "index": index,
                        "shape": tuple(array.shape),
                        "dtype": str(array.dtype),
                        "min": float(array.min()) if array.size else None,
                        "max": float(array.max()) if array.size else None,
                        "first_values": array.reshape(-1)[:12].tolist(),
                    }
                )

            LOGGER.warning(
                "IMX RAW OUTPUTS: %s",
                output_summary,
            )

        intr = self.intrinsics
        input_w, input_h = self.imx500.get_input_size()

        if intr.postprocess == "nanodet":
            from picamera2.devices.imx500.postprocess import scale_boxes

            boxes, scores, classes = postprocess_nanodet_detection(
                outputs=np_outputs[0],
                conf=self.threshold,
                iou_thres=self.iou,
                max_out_dets=self.max_detections,
            )[0]

            boxes = scale_boxes(
                boxes,
                1,
                1,
                input_h,
                input_w,
                False,
                False,
            )

        elif len(np_outputs) >= 4:
            # Ultralytics IMX500 export with in-network NMS:
            #   output 0: boxes   [1, max_det, 4] in XYXY input pixels
            #   output 1: scores  [1, max_det]
            #   output 2: classes [1, max_det]
            #   output 3: number of valid detections [1, 1]
            #
            # Slots after valid_count are padding and MUST NOT be tracked.
            boxes = np.asarray(np_outputs[0])[0]
            scores = np.asarray(np_outputs[1])[0].reshape(-1)
            classes = np.asarray(np_outputs[2])[0].reshape(-1)

            count_values = np.asarray(np_outputs[3]).reshape(-1)
            valid_count = (
                int(round(float(count_values[0])))
                if count_values.size
                else 0
            )

            available = min(
                len(boxes),
                len(scores),
                len(classes),
            )

            valid_count = max(
                0,
                min(
                    valid_count,
                    available,
                    self.max_detections,
                ),
            )

            boxes = boxes[:valid_count].astype(
                np.float32,
                copy=False,
            )
            scores = scores[:valid_count]
            classes = classes[:valid_count]

            if valid_count:
                # Ultralytics NMS boxes are XYXY in the 320×320 model-input
                # coordinate space. convert_inference_coords expects normalised
                # YXYX coordinates.
                boxes[:, [0, 2]] /= float(input_w)
                boxes[:, [1, 3]] /= float(input_h)
                boxes = boxes[:, [1, 0, 3, 2]]

        elif len(np_outputs) >= 3:
            # Standard SSD-style post-processed output.
            boxes = np.asarray(np_outputs[0])[0]
            scores = np.asarray(np_outputs[1])[0].reshape(-1)
            classes = np.asarray(np_outputs[2])[0].reshape(-1)

            if intr.bbox_normalization:
                boxes = boxes / input_h

            if intr.bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

        else:
            LOGGER.error(
                "Unsupported IMX500 output tensor count: %d",
                len(np_outputs),
            )
            return []

        detections: list[Detection] = []

        for box, score, category in zip(
            boxes,
            scores,
            classes,
        ):
            score_value = float(score)

            if score_value <= self.threshold:
                continue

            category_value = int(round(float(category)))

            if not 0 <= category_value < len(self._labels):
                LOGGER.debug(
                    "Ignoring invalid class index %s",
                    category_value,
                )
                continue

            x, y, w, h = self.imx500.convert_inference_coords(
                box,
                metadata,
                picam2,
            )

            label = self._label_for(category_value)
            if label == "suitcase":
                # The custom model's "suitcase" class is unreliable - it
                # confidently (scores up to 0.80 in real deployment logs, not
                # just borderline noise) fires on backpacks/handbags instead.
                # Until it's retrained with a properly balanced bag dataset,
                # remap it to "backpack" here, before tracking, alerting, or
                # logging ever sees the label.
                label = "backpack"

            detections.append(
                Detection(
                    label,
                    score_value,
                    (
                        round(x),
                        round(y),
                        round(w),
                        round(h),
                    ),
                )
            )

        return detections


class PersonTracker:
    """Lightweight IoU/centroid tracker that assigns persons stable ids.

    The ids let BagTracker tell a bag's owner apart from anyone else who happens
    to walk near it.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.tracks: dict[int, PersonTrack] = {}
        self._next_id = 0

    def update(self, detections: list[Detection], now: float) -> None:
        detections = dedup_detections(detections)
        matches = match_detections(detections, list(self.tracks.values()),
                                   iou_gate=0.2,
                                   dist_gate=self.config.person_match_radius)
        for di, det in enumerate(detections):
            track = matches.get(di)
            if track is None:
                track = PersonTrack(self._next_id, det.centroid, det.box, now)
                self.tracks[track.person_id] = track
                self._next_id += 1
            else:
                track.box = smooth_box(track.box, det.box, self.config.box_smoothing)
                track.centroid = box_centroid(track.box)
                track.last_seen = now

    def prune(self, now: float) -> None:
        expired = [pid for pid, t in self.tracks.items()
                   if now - t.last_seen > self.config.person_timeout_sec]
        for pid in expired:
            del self.tracks[pid]


class BagTracker:
    """Greedy nearest-centroid tracker with an unattended timer per bag."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.bags: dict[int, TrackedBag] = {}
        self._next_id = 0

    def update(self, bag_detections: list[Detection],
               persons: list[PersonTrack], now: float) -> None:
        """Match this frame's bag detections to tracks and refresh attention state."""
        bag_detections = dedup_detections(bag_detections)
        matches = match_detections(bag_detections, list(self.bags.values()),
                                   iou_gate=0.3,
                                   dist_gate=self.config.stationary_radius)
        for di, det in enumerate(bag_detections):
            bag = matches.get(di)
            if bag is None:
                bag = self._register(det, now)
            else:
                bag.box = smooth_box(bag.box, det.box, self.config.box_smoothing)
                bag.centroid = box_centroid(bag.box)
                bag.last_seen = now
                bag.score = det.score
                bag.label = det.label  # keep the specific class current (backpack/handbag/suitcase)
            self._update_attention(bag, persons, now)
        self._merge_duplicate_tracks()

    def _merge_duplicate_tracks(self) -> None:
        """Collapse tracks stacked on the same physical bag.

        Detection flicker can strand a coasting track that a fresh track then
        piles onto — both would alert (and both would stamp a red box on the
        snapshot). The oldest track wins: it keeps its id, owner, and
        unattended timer, and adopts the duplicate's fresher sighting.
        """
        ids = sorted(self.bags)  # ascending id = oldest first
        dropped: set[int] = set()
        for i, keep_id in enumerate(ids):
            if keep_id in dropped:
                continue
            keeper = self.bags[keep_id]
            for dup_id in ids[i + 1:]:
                if dup_id in dropped:
                    continue
                dup = self.bags[dup_id]
                if calculate_iou(keeper.box, dup.box) < DEDUP_IOU:
                    continue
                if dup.last_seen > keeper.last_seen:
                    keeper.box = dup.box
                    keeper.centroid = dup.centroid
                    keeper.last_seen = dup.last_seen
                if keeper.owner_id is None:
                    keeper.owner_id = dup.owner_id
                dropped.add(dup_id)
                LOGGER.debug("Merged duplicate bag #%d into bag #%d", dup_id, keep_id)
        for bid in dropped:
            del self.bags[bid]

    def prune(self, now: float) -> list[int]:
        """Drop tracks not seen within the timeout. Returns removed ids."""
        expired = [
            bid for bid, b in self.bags.items()
            if now - b.last_seen > self.config.track_timeout_sec
        ]
        for bid in expired:
            LOGGER.debug("Dropping bag #%d (unseen %.1fs > timeout %.1fs)",
                         bid, now - self.bags[bid].last_seen, self.config.track_timeout_sec)
            del self.bags[bid]
        return expired

    def _register(self, det: Detection, now: float) -> TrackedBag:
        bag = TrackedBag(self._next_id, det.centroid, det.box, now, now,
                         score=det.score, label=det.label)
        self.bags[bag.bag_id] = bag
        self._next_id += 1
        LOGGER.debug("Registered new bag #%d (%s) at %s", bag.bag_id, det.label, det.box)
        return bag

    def _update_attention(self, bag: TrackedBag,
                          persons: list[PersonTrack], now: float) -> None:
        """Owner-locked attention: only the bag's owner being near resets the timer.

        The owner is the person adopted while the bag is new (within owner_claim_sec).
        Once a bag has no owner — e.g. it entered the scene already abandoned — no
        bystander can claim it, so a passer-by or someone standing nearby never
        resets the unattended timer.
        """
        prox = self.config.person_proximity_px

        # Is the current owner (if any) near the bag right now?
        owner_present = False
        if bag.owner_id is not None:
            owner = next((p for p in persons if p.person_id == bag.owner_id), None)
            owner_present = owner is not None and distance(bag.centroid, owner.centroid) <= prox

        # While the bag is still new, adopt the nearest in-range person as its owner.
        if bag.owner_id is None and (now - bag.created_at) <= self.config.owner_claim_sec:
            in_range = [(distance(bag.centroid, p.centroid), p) for p in persons]
            nearest = min((pair for pair in in_range if pair[0] <= prox),
                          key=lambda pair: pair[0], default=(None, None))[1]
            if nearest is not None:
                bag.owner_id = nearest.person_id
                owner_present = True
                LOGGER.debug("Bag #%d adopted owner = person #%d", bag.bag_id, nearest.person_id)

        if owner_present:
            bag.unattended_start = None
            bag.alerted = False
        elif bag.unattended_start is None:
            bag.unattended_start = now


class PestTracker:
    """Tracks pests (rat/mouse) independently of any owner/attention logic.

    A pest is *confirmed* once it has been continuously visible for
    ``pest_confirmation_time`` seconds OR seen in ``pest_confirmation_frames``
    consecutive frames (either criterion; set one to 0 to disable it). A
    confirmed pest alerts once, then at most once per ``pest_alert_cooldown``
    seconds while it stays visible — so a rat sitting in view doesn't spam
    alerts. A pest unseen for ``pest_timeout_sec`` is dropped and its
    confirmation resets.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.pests: dict[int, TrackedPest] = {}
        self._next_id = 0

    def update(self, detections: list[Detection], now: float) -> None:
        detections = dedup_detections(detections)
        matches = match_detections(detections, list(self.pests.values()),
                                   iou_gate=0.3,
                                   dist_gate=self.config.pest_match_radius)
        for di, det in enumerate(detections):
            pest = matches.get(di)
            if pest is None:
                pest = TrackedPest(self._next_id, det.label, det.centroid, det.box,
                                   now, now, score=det.score)
                self.pests[pest.pest_id] = pest
                self._next_id += 1
                LOGGER.debug("Registered new pest #%d (%s) at %s",
                             pest.pest_id, det.label, det.box)
            else:
                pest.box = smooth_box(pest.box, det.box, self.config.box_smoothing)
                pest.centroid = box_centroid(pest.box)
                pest.last_seen = now
                pest.label = det.label
                pest.score = det.score
                pest.frames += 1

    def prune(self, now: float) -> list[int]:
        expired = [pid for pid, p in self.pests.items()
                   if now - p.last_seen > self.config.pest_timeout_sec]
        for pid in expired:
            del self.pests[pid]
        return expired


def pest_confirmed(pest: TrackedPest, config: Config, now: float) -> bool:
    """True once a pest satisfies either confirmation criterion."""
    by_time = (config.pest_confirmation_time > 0
               and (now - pest.first_seen) >= config.pest_confirmation_time)
    by_frames = (config.pest_confirmation_frames > 0
                 and pest.frames >= config.pest_confirmation_frames)
    return by_time or by_frames


def _pest_alert_due(pest: TrackedPest, config: Config, now: float) -> bool:
    """True if this pest should log + snapshot this frame (first alert or cooldown up)."""
    if not pest_confirmed(pest, config, now):
        return False
    return not pest.alerted or now - pest.last_alert_time >= config.pest_alert_cooldown_sec


def save_snapshot_worker(frame_copy, path: Path, directory: Path, keep: int) -> None:
    if cv2.imwrite(str(path), frame_copy):
        LOGGER.info("Saved alert snapshot: %s", path)
    else:
        LOGGER.error("Failed to save alert snapshot: %s", path)
    try:
        # Cap the directory so an unattended deployment can't fill the SD card
        # (a full card takes the whole Pi down, not just the snapshots).
        snaps = sorted(directory.glob("*.jpg"), key=lambda f: f.stat().st_mtime)
        for old in snaps[:max(0, len(snaps) - keep)]:
            old.unlink()
            LOGGER.debug("Pruned old snapshot: %s", old)
    except OSError as exc:
        LOGGER.warning("Snapshot pruning failed: %s", exc)


def save_snapshot(frame, config: Config, filename: str) -> Path:
    """Write an annotated snapshot into the snapshot dir (non-blocking) and return its path."""
    config.snapshot_dir.mkdir(parents=True, exist_ok=True)
    path = config.snapshot_dir / filename
    SNAPSHOT_EXECUTOR.submit(save_snapshot_worker, frame.copy(), path,
                             config.snapshot_dir, config.max_snapshots)
    return path


# Every alert is appended here (inside the log dir) as a CSV row, giving a
# machine-readable record of incidents for later analysis or reporting.
EVENT_LOG_NAME = "events.csv"
EVENT_HEADER = ["time", "event_type", "label", "confidence",
                "track_id", "zone", "duration_sec", "snapshot"]


def append_event_worker(csv_path: Path, row: list, header: list = EVENT_HEADER) -> None:
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        header_needed = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            if header_needed:
                writer.writerow(header)
            writer.writerow(row)
    except OSError as exc:
        LOGGER.warning("Could not append to event log %s: %s", csv_path, exc)


def log_event(config: Config, *, event_type: str, label: str,
              confidence: Optional[float], track_id: Optional[int],
              duration_sec: Optional[float], snapshot: Optional[str]) -> None:
    """Append one structured incident row to the event log (non-blocking)."""
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        event_type,
        label,
        f"{confidence:.2f}" if confidence is not None else "",
        track_id if track_id is not None else "",
        config.zone or "",
        int(duration_sec) if duration_sec is not None else "",
        snapshot or "",
    ]
    SNAPSHOT_EXECUTOR.submit(append_event_worker, config.log_dir / EVENT_LOG_NAME, row)


def _alert_due(bag: TrackedBag, config: Config, now: float) -> bool:
    """True if this bag should log + snapshot this frame (first alert or cooldown up)."""
    if bag.unattended_start is None:
        return False
    if now - bag.unattended_start < config.unattended_time_sec:
        return False
    return not bag.alerted or now - bag.last_alert_time >= config.alert_cooldown_sec


def _flowguard_zone_camera(client: "FlowGuardApiClient", config: Config) -> tuple[str, str]:
    """Human zone/camera labels for a cloud event. Falls back sensibly so the
    backend's required zone_name/camera_location are always present."""
    zone_name = (config.zone or "").strip().title() or (client.camera_location or "SecurePi Zone")
    camera_location = client.camera_location or f"{zone_name} Camera"
    return zone_name, camera_location


def _fire_sensor_person_alert(frame, person, identity_info: dict, sensor_metadata: dict,
                              config: Config, now: float, renderer: "Renderer", *, client=None) -> None:
    """Fire a Restricted-Zone Motion alert when a person is detected during sensor inspection."""
    person_id = person.person_id
    identity_status = identity_info.get("identity_status", "UNAVAILABLE")
    person_name = identity_info.get("person_name")
    person_role = identity_info.get("person_role")
    conf = identity_info.get("confidence") or getattr(person, "score", None)

    severity = "Critical" if identity_status in ("SUSPICIOUS", "SUSPENDED") else "High"

    LOGGER.warning("RESTRICTED-ZONE MOTION - person #%d (%s - %s)",
                   person_id, person_name or "Unknown", identity_status)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_path = save_snapshot(frame, config, f"sensor_person_{person_id}_{stamp}.jpg")

    log_event(config, event_type="restricted_motion", label="person",
              confidence=conf, track_id=person_id,
              duration_sec=None, snapshot=snapshot_path.name)

    if client is not None and getattr(client, "enabled", False) and build_event_id is not None:
        event_id = build_event_id(client.device_id, "restricted_motion", person_id, time.time())
        _enqueue_flowguard(
            client, config,
            event_type="restricted_motion",
            alert_type="Restricted-Zone Motion",
            object_class="person",
            confidence=conf,
            duration_seconds=None,
            track_id=person_id,
            snapshot_path=snapshot_path,
            event_id=event_id,
            person_name=person_name,
            identity_status=identity_status,
            person_role=person_role,
            severity=severity,
            sensor_metadata=sensor_metadata,
        )


def _fire_alert(frame, bag: TrackedBag, config: Config, now: float,
                renderer: "Renderer", *, client=None) -> None:
    bag.alerted = True
    bag.last_alert_time = now
    duration = int(now - bag.unattended_start)
    LOGGER.warning("UNATTENDED OBJECT ALERT - bag #%d unattended for %ds",
                   bag.bag_id, duration)
    # Stamp the alert box here rather than relying on the render pass: the
    # renderer skips tracks unseen past draw_grace_sec, so a bag the detector
    # lost at alert time would otherwise save a snapshot with nothing marking
    # it. Redrawing over an already-drawn box is a no-op.
    renderer.draw_box(frame, bag.box, COLOR_ALERT,
                      f"ALERT! Bag #{bag.bag_id} unattended {duration}s",
                      thickness=3)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_path = save_snapshot(frame, config, f"alert_bag{bag.bag_id}_{stamp}.jpg")
    # Preserve the specific detected class (backpack/handbag/suitcase) rather than a
    # generic "bag" — falls back to "bag" only when the track carries no class.
    object_class = bag.label or "bag"
    log_event(config, event_type="unattended_object", label=object_class,
              confidence=bag.score or None, track_id=bag.bag_id,
              duration_sec=duration, snapshot=snapshot_path.name)
    # One stable event_id per occurrence: computed on the FIRST alert and reused on
    # every cooldown re-alert and Wi-Fi retry so the backend de-duplicates.
    if client is not None and getattr(client, "enabled", False) and build_event_id is not None:
        if bag.flowguard_event_id is None:
            bag.flowguard_event_id = build_event_id(client.device_id, "unattended_object",
                                                    bag.bag_id, time.time())
        _enqueue_flowguard(client, config, event_type="unattended_object",
                           alert_type="Unattended Object", object_class=object_class,
                           confidence=bag.score or None, duration_seconds=duration,
                           track_id=bag.bag_id, snapshot_path=snapshot_path,
                           event_id=bag.flowguard_event_id)


def _fire_pest_alert(frame, pest: TrackedPest, config: Config, now: float,
                     renderer: "Renderer", *, client=None) -> None:
    pest.alerted = True
    pest.last_alert_time = now
    duration = now - pest.first_seen
    LOGGER.warning("PEST ALERT - %s (pest #%d) visible for %.1fs (conf %.2f)",
                   pest.label, pest.pest_id, duration, pest.score)
    renderer.draw_box(frame, pest.box, COLOR_PEST,
                      f"PEST! {pest.label} #{pest.pest_id}", thickness=3)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot_path = save_snapshot(frame, config,
                                  f"pest_{pest.label}_{pest.pest_id}_{stamp}.jpg")
    log_event(config, event_type="pest", label=pest.label,
              confidence=pest.score or None, track_id=pest.pest_id,
              duration_sec=duration, snapshot=snapshot_path.name)
    # The exact detected pest label (rat/mouse) rides through to FlowGuard — never
    # reduced to a generic term, never sent as an unattended-object alert_type.
    if client is not None and getattr(client, "enabled", False) and build_event_id is not None:
        if pest.flowguard_event_id is None:
            pest.flowguard_event_id = build_event_id(client.device_id, "pest_detection",
                                                     pest.pest_id, time.time())
        _enqueue_flowguard(client, config, event_type="pest_detection",
                           alert_type="Pest Detection", object_class=pest.label,
                           confidence=pest.score or None, duration_seconds=None,
                           track_id=pest.pest_id, snapshot_path=snapshot_path,
                           event_id=pest.flowguard_event_id)


class Renderer:
    """Handles UI rendering to decouple it from tracking logic."""

    def __init__(self, config: Config):
        self.config = config

    def draw_box(self, frame, box: tuple[int, int, int, int], color, label: Optional[str] = None, thickness: int = 2) -> None:
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        if label:
            cv2.putText(frame, label, (x, max(y - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def draw_countdown(self, frame, box: tuple[int, int, int, int],
                       fraction: float, color) -> None:
        """Progress bar under the box filling toward the alert threshold."""
        x, y, w, h = box
        bar_h = 6
        y0 = y + h + 4
        if y0 + bar_h >= frame.shape[0]:   # would fall off-frame: draw above instead
            y0 = max(0, y - bar_h - 4)
        x0 = max(0, x)
        x1 = min(frame.shape[1] - 1, x + w)
        if x1 <= x0:
            return
        cv2.rectangle(frame, (x0, y0), (x1, y0 + bar_h), color, 1)
        fill = x0 + int((x1 - x0) * min(1.0, max(0.0, fraction)))
        if fill > x0:
            cv2.rectangle(frame, (x0, y0), (fill, y0 + bar_h), color, -1)

    def draw_hud(self, frame, person_count: int, bag_count: int,
                 pest_count: int, fps: float) -> None:
        cv2.rectangle(frame, (10, 10), (360, 72), (0, 0, 0), -1)
        cv2.putText(frame, f"People: {person_count}  Bags: {bag_count}  Pests: {pest_count}",
                    (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_HUD, 2)

    def handle_bag(self, frame, bag: TrackedBag, now: float) -> None:
        """Draw a bag in the right state."""
        unseen = now - bag.last_seen
        # Don't draw a long-coasting track: avoids a "ghost" box lingering at the
        # old location after the bag has moved (or genuinely left). The track stays
        # alive until track_timeout_sec so its unattended timer keeps running.
        if unseen > self.config.draw_grace_sec:
            return
        # Mark tracks we're coasting on (no detection matched this frame).
        suffix = " (searching...)" if unseen > 0.5 else ""

        if bag.unattended_start is None:
            self.draw_box(frame, bag.box, COLOR_ATTENDED, f"Bag #{bag.bag_id} (Attended){suffix}")
            return

        duration = now - bag.unattended_start
        if duration >= self.config.unattended_time_sec:
            self.draw_box(frame, bag.box, COLOR_ALERT,
                     f"ALERT! Bag #{bag.bag_id} unattended {int(duration)}s", thickness=3)
            self.draw_countdown(frame, bag.box, 1.0, COLOR_ALERT)
        else:
            self.draw_box(frame, bag.box, COLOR_WARNING,
                     f"Bag #{bag.bag_id} unattended {int(duration)}s{suffix}")
            # Countdown bar fills toward the alert threshold so the state is
            # readable from across the room.
            self.draw_countdown(frame, bag.box,
                                duration / self.config.unattended_time_sec,
                                COLOR_WARNING)

    def handle_pest(self, frame, pest: TrackedPest, now: float) -> None:
        """Draw a pest track (magenta once confirmed, dimmer while pending)."""
        unseen = now - pest.last_seen
        if unseen > self.config.draw_grace_sec:
            return
        confirmed = pest_confirmed(pest, self.config, now)
        state = "PEST" if confirmed else "pest?"
        self.draw_box(frame, pest.box, COLOR_PEST,
                      f"{state} {pest.label} #{pest.pest_id}",
                      thickness=3 if confirmed else 2)


class FrameBuffer:
    """Thread-safe holder for the single latest annotated JPEG frame.

    Only the newest frame is kept — there is no queue, so a slow or stalled
    client can never build a backlog, and every client always receives the
    most recent frame the detection loop published.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: Optional[bytes] = None
        self._seq = 0                       # bumps on every publish
        self._published_at: Optional[float] = None  # monotonic

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._published_at = time.monotonic()
            self._cond.notify_all()

    def wait_for_frame(self, last_seq: int,
                       timeout: float = 1.0) -> tuple[Optional[bytes], int]:
        """Block until a frame newer than ``last_seq`` arrives.

        Returns ``(jpeg, seq)``; ``jpeg`` is None if the timeout expired first.
        A client that joins late (``last_seq=0``) gets the current frame at once.
        """
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout)
            if self._seq == last_seq or self._jpeg is None:
                return None, last_seq
            return self._jpeg, self._seq

    def age_seconds(self) -> Optional[float]:
        """Seconds since the last publish, or None if nothing published yet."""
        with self._cond:
            if self._published_at is None:
                return None
            return max(0.0, time.monotonic() - self._published_at)


class _StreamHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True    # in-flight client threads never block process exit
    allow_reuse_address = True


class StreamServer:
    """Embedded HTTP server exposing /health, /sensor_status, /video_feed, /people-count, and /snapshot.

    Lives inside the SecurePi process in a background thread and serves only
    frames the detection loop has already annotated and published to a
    FrameBuffer. It never touches Picamera2 or the IMX500, so the
    one-camera-owner rule holds no matter how many clients connect, and a
    disconnecting client can only ever kill its own handler thread.
    """

    def __init__(self, config: Config, buffer: FrameBuffer, state_provider: Optional[Callable] = None, sensor_bridge: Optional[Any] = None) -> None:
        self._stop_event = threading.Event()
        handler = self._make_handler(buffer, self._stop_event, state_provider, sensor_bridge)
        self._httpd = _StreamHTTPServer((config.stream_host, config.stream_port),
                                        handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        name="securepi-http", daemon=True)

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()      # ends any in-flight /video_feed loops
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)

    @staticmethod
    def _make_handler(buffer: FrameBuffer, stop_event: threading.Event, state_provider: Optional[Callable] = None, sensor_bridge: Optional[Any] = None):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                LOGGER.debug("HTTP %s %s", self.address_string(), fmt % args)

            def _common_headers(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")

            def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
                try:
                    path = self.path.split("?", 1)[0]
                    if path == "/health":
                        self._serve_health()
                    elif path == "/sensor_status":
                        self._serve_sensor_status()
                    elif path == "/video_feed":
                        self._serve_video_feed()
                    elif path == "/people-count":
                        self._serve_people_count()
                    elif path == "/snapshot":
                        self._serve_snapshot()
                    else:
                        self.send_error(404, "Not Found")
                except OSError:
                    # Broken pipe / reset: the client went away mid-response.
                    # Swallow it — a dead browser tab must never disturb the
                    # detection loop or the other clients.
                    LOGGER.debug("HTTP client %s disconnected", self.address_string())

            def _serve_sensor_status(self) -> None:
                if sensor_bridge and hasattr(sensor_bridge, "get_sensor_status"):
                    st = sensor_bridge.get_sensor_status()
                else:
                    st = {
                        "connected": False,
                        "pir_ready": False,
                        "pir": False,
                        "motion": False,
                        "distance_cm": None,
                        "baseline_distance_cm": None,
                        "distance_change_cm": None,
                        "object_close": False,
                        "trigger": None,
                        "inspection_active": False,
                        "inspection_remaining_seconds": 0.0,
                        "after_hours": False,
                        "inspection_id": None,
                    }
                body = json.dumps(st).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._common_headers()
                self.end_headers()
                self.wfile.write(body)

            def _serve_health(self) -> None:
                age = buffer.age_seconds()
                if sensor_bridge and hasattr(sensor_bridge, "get_sensor_status"):
                    sensor_st = sensor_bridge.get_sensor_status()
                else:
                    sensor_st = {
                        "connected": False,
                        "pir_ready": False,
                        "pir": False,
                        "motion": False,
                        "distance_cm": None,
                        "baseline_distance_cm": None,
                        "distance_change_cm": None,
                        "object_close": False,
                        "trigger": None,
                        "inspection_active": False,
                        "inspection_remaining_seconds": 0.0,
                        "after_hours": False,
                        "inspection_id": None,
                    }
                body = json.dumps({
                    "status": "online",
                    "camera": "IMX500",
                    "streaming": True,
                    "latest_frame_age_seconds": (round(age, 3)
                                                 if age is not None else None),
                    "sensor": sensor_st,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._common_headers()
                self.end_headers()
                self.wfile.write(body)

            def _serve_people_count(self) -> None:
                st = state_provider() if state_provider else {"count": 0, "detection_active": False}
                body = json.dumps({
                    "count": int(st.get("count", st.get("people_count", 0))),
                    "detection_active": bool(st.get("detection_active", False)),
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self._common_headers()
                self.end_headers()
                self.wfile.write(body)

            def _serve_snapshot(self) -> None:
                jpeg, _ = buffer.wait_for_frame(0, timeout=0.1)
                if jpeg is None:
                    self.send_error(503, "Service Unavailable")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self._common_headers()
                self.end_headers()
                self.wfile.write(jpeg)

            def _serve_video_feed(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self._common_headers()
                self.end_headers()
                seq = 0
                while not stop_event.is_set():
                    jpeg, seq = buffer.wait_for_frame(seq, timeout=1.0)
                    if jpeg is None:
                        continue    # no new frame yet — keep the socket open
                    self.wfile.write(b"--frame\r\n"
                                     b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n"
                                     .encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")

        return Handler


def _sigterm_exit(signum, frame) -> None:
    """Route SIGTERM (systemctl stop) through the normal cleanup path."""
    raise SystemExit(0)


def _check_flowguard_connectivity(client: "FlowGuardApiClient", timeout: float = 3.0) -> None:
    """One-shot reachability check logged at startup, so it's obvious from the
    CLI alone whether the configured FlowGuard backend is actually up — not
    just whether the integration is *enabled*. Best-effort only: never raises,
    and a failure here doesn't stop the monitor (the outbox will keep retrying)."""
    label = mask_url(client.api_url) if mask_url else client.api_url
    try:
        with urllib.request.urlopen(client.api_url, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
        LOGGER.info("[FlowGuard] CONNECTED -> %s (HTTP %s)", label, status)
    except Exception as exc:  # pragma: no cover - defensive; never blocks startup
        LOGGER.warning("[FlowGuard] NOT REACHABLE -> %s (%s)", label, exc)


def run(config: Config, client=None, sensor_bridge=None) -> None:
    detector = IMX500Detector(config)

    # Make sure the runtime tree exists up-front so operators can find it even
    # before the first alert. All three are git-ignored.
    for sub in (config.snapshot_dir, config.log_dir, config.runtime_dir / "alerts"):
        Path(sub).mkdir(parents=True, exist_ok=True)

    # Flush any events left queued by a previous run (crash / power-loss recovery).
    if client is not None:
        client.start()
        _check_flowguard_connectivity(client)
    else:
        LOGGER.info("[FlowGuard] Disabled (FLOWGUARD_EDGE_ENABLED not set) — running offline.")

    # Initialize SensorBridge if not provided
    if sensor_bridge is None and _SENSOR_BRIDGE_AVAILABLE and SensorBridge is not None:
        zone = config.zone or "Restricted Zone"
        sensor_bridge = SensorBridge(
            client=client,
            zone_name=zone,
            camera_location=getattr(config, "camera_location", "Camera 01") or f"{zone} Camera",
            hours_start=getattr(config, "restricted_hours_start", os.environ.get("RESTRICTED_HOURS_START", "08:00")),
            hours_end=getattr(config, "restricted_hours_end", os.environ.get("RESTRICTED_HOURS_END", "23:00")),
        )

    if sensor_bridge is not None and start_sensor_reader_thread:
        port = _default_serial_port() if callable(_default_serial_port) else None
        if port:
            start_sensor_reader_thread(sensor_bridge, port)

    picam2 = Picamera2(detector.camera_num)
    controls = {}
    if detector.inference_rate:
        controls["FrameRate"] = detector.inference_rate
    cam_config = picam2.create_preview_configuration(
        main={"size": config.frame_size, "format": "RGB888"},
        controls=controls,
        buffer_count=6,
    )
    picam2.configure(cam_config)
    detector.show_progress()  # firmware upload progress bar on first run
    picam2.start()

    signal.signal(signal.SIGTERM, _sigterm_exit)  # systemd stop → clean shutdown

    tracker = BagTracker(config)
    person_tracker = PersonTracker(config)
    pest_tracker = PestTracker(config)
    renderer = Renderer(config)

    current_visible_count = 0
    state_provider = lambda: {
        "count": current_visible_count,
        "detection_active": sensor_bridge.is_inspection_active() if sensor_bridge else False,
    }

    frame_buffer: Optional[FrameBuffer] = None
    stream_server: Optional[StreamServer] = None
    if config.stream_enabled:
        frame_buffer = FrameBuffer()
        stream_server = StreamServer(config, frame_buffer, state_provider=state_provider, sensor_bridge=sensor_bridge)
        stream_server.start()
        LOGGER.info("MJPEG stream on http://%s:%d/video_feed (health: /health, telemetry: /people-count, snapshot: /snapshot, sensor: /sensor_status)",
                    config.stream_host, stream_server.port)
    stream_interval = 1.0 / config.stream_fps
    next_publish = 0.0

    LOGGER.info("Security monitor started (%s mode). Press 'q' in the window to quit.",
                "headless" if config.headless else "preview")

    fps = 0.0
    prev = time.monotonic()
    last_flowguard_flush = time.monotonic()
    last_detection_log = 0.0
    fired_sensor_person_alerts = set()

    try:
        while True:
            request = picam2.capture_request()
            try:
                metadata = request.get_metadata()
                now = time.monotonic()
                detections = detector.detect(metadata, picam2)
                if detections and now - last_detection_log >= 1.0:
                    last_detection_log = now
                    summary = ", ".join(f"{d.label}({d.score:.2f})" for d in detections)
                    LOGGER.info("Detected: %s", summary)
                person_dets, bag_dets, pest_dets = split_detections(detections, config)

                person_tracker.update(person_dets, now)
                person_tracker.prune(now)
                persons = list(person_tracker.tracks.values())

                tracker.update(bag_dets, persons, now)
                tracker.prune(now)

                pest_tracker.update(pest_dets, now)
                pest_tracker.prune(now)

                bag_alerts_due = [b for b in tracker.bags.values()
                                  if _alert_due(b, config, now)]
                pest_alerts_due = [p for p in pest_tracker.pests.values()
                                   if _pest_alert_due(p, config, now)]
                stream_due = stream_server is not None and now >= next_publish
                frame = (request.make_array("main")
                         if not config.headless or bag_alerts_due or pest_alerts_due or stream_due
                         else None)
            finally:
                request.release()

            dt = now - prev
            prev = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

            if frame is not None:
                owner_ids = {b.owner_id for b in tracker.bags.values()
                             if b.owner_id is not None}
                visible_persons = 0
                inspection_active = sensor_bridge.is_inspection_active() if sensor_bridge else False
                sensor_meta = sensor_bridge.latest_sensor_metadata if (sensor_bridge and inspection_active) else {}

                for person in persons:
                    if now - person.last_seen > config.draw_grace_sec:
                        continue
                    visible_persons += 1
                    tag = " (owner)" if person.person_id in owner_ids else ""
                    face_info = _recognize_person_with_flowguard(frame, person.box, person.person_id)

                    if isinstance(face_info, dict):
                        display_label = face_info.get("display_label", f"#{person.person_id} Person")
                    else:
                        display_label = face_info or f"#{person.person_id} Person"

                    renderer.draw_box(frame, person.box, COLOR_PERSON, f"{display_label}{tag}")

                    if inspection_active and isinstance(face_info, dict):
                        expiry = getattr(sensor_bridge, "_inspection_expiry_dt", None)
                        alert_key = (person.person_id, expiry)
                        if alert_key not in fired_sensor_person_alerts:
                            fired_sensor_person_alerts.add(alert_key)
                            _fire_sensor_person_alert(frame, person, face_info, sensor_meta,
                                                       config, now, renderer, client=client)

                current_visible_count = visible_persons
                for bag in tracker.bags.values():
                    renderer.handle_bag(frame, bag, now)
                for pest in pest_tracker.pests.values():
                    renderer.handle_pest(frame, pest, now)
                renderer.draw_hud(frame, visible_persons, len(tracker.bags),
                                  len(pest_tracker.pests), fps)

            for bag in bag_alerts_due:
                _fire_alert(frame, bag, config, now, renderer, client=client)
            for pest in pest_alerts_due:
                _fire_pest_alert(frame, pest, config, now, renderer, client=client)

            # Publish only after every annotation (including any alert box just
            # stamped above) is on the frame, so the stream shows exactly what
            # a snapshot would.
            if stream_due and frame is not None:
                ok, jpeg = cv2.imencode(
                    ".jpg", frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), config.stream_quality])
                if ok:
                    frame_buffer.publish(jpeg.tobytes())
                else:
                    LOGGER.warning("JPEG encode failed; stream frame skipped.")
                next_publish = now + stream_interval

            # Periodically retry the outbox so events queued while Wi-Fi was down get
            # re-sent even when no new alert fires. Non-blocking (runs on the worker).
            if client is not None and (now - last_flowguard_flush) >= client.retry_interval:
                last_flowguard_flush = now
                client.flush_async()

            if not config.headless:
                cv2.imshow("SecurePi - AI Security Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        if stream_server is not None:
            stream_server.stop()
        picam2.stop()
        cv2.destroyAllWindows()
        # Stop the FlowGuard background worker cleanly; unsent events remain safely
        # on disk in the outbox and will flush on the next startup.
        if client is not None:
            client.stop(wait=True)
        LOGGER.info("Security monitor stopped.")


class _ArgFileParser(argparse.ArgumentParser):
    """ArgumentParser that also reads flags from a file: `securePi.py @site.args`.

    Inside the file, blank lines are skipped, everything after a `#` is a
    comment, and a flag and its value may share a line (`--unattended-time 60`).
    """

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        line = arg_line.split("#", 1)[0].strip()
        return line.split()


# Location presets live here; common.args in the same folder is the shared
# base applied automatically before whichever preset the user names.
PRESETS_DIR = Path(__file__).resolve().parent / "presets"


def _env_flag(name: str) -> bool:
    """Interpret an environment variable as a boolean flag (default off)."""
    return str(os.environ.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def _resolve_project_path(value: Optional[str]) -> Optional[str]:
    """Resolve a relative model/labels path against the repo root as a fallback.

    A path that exists as given (relative to the CWD) or is absolute is kept;
    otherwise, if `<repo_root>/<value>` exists it is used. This lets
    `--model models/x.rpk` work from any working directory. If neither exists
    the original value is returned so the downstream error is clear.
    """
    if not value:
        return value
    p = Path(value)
    if p.is_absolute() or p.exists():
        return str(p)
    candidate = REPO_ROOT / value
    if candidate.exists():
        return str(candidate)
    return value


def _require_args_file(parser: argparse.ArgumentParser, raw_args: list[str]) -> None:
    """Refuse to run without an @<preset>.args file, listing the ones available."""
    if any(str(a).startswith("@") for a in raw_args):
        return
    presets = sorted(f.name for f in PRESETS_DIR.glob("*.args")
                     if f.name != "common.args")
    if presets:
        hint = "available presets: " + ", ".join(f"@{name}" for name in presets)
    else:
        hint = f"create one in {PRESETS_DIR} (see README)"
    parser.error(f"a settings file is required, e.g. `python edge/securePi.py @lobby.args` — {hint}")


def _resolve_preset_refs(raw_args: list[str]) -> list[str]:
    """Let `@lobby.args` find presets/lobby.args regardless of working directory.

    An @path that exists as given (relative to the CWD, or absolute) is kept;
    otherwise it is looked up by name in PRESETS_DIR.
    """
    resolved = []
    for arg in raw_args:
        if isinstance(arg, str) and arg.startswith("@") and not Path(arg[1:]).exists():
            candidate = PRESETS_DIR / Path(arg[1:]).name
            if candidate.exists():
                arg = f"@{candidate}"
        resolved.append(arg)
    return resolved


def parse_args(argv=None) -> argparse.Namespace:
    p = _ArgFileParser(
        description="SecurePi — IMX500 unattended-object + pest security monitor.",
        fromfile_prefix_chars="@",
        epilog="A @<preset>.args settings file is required, e.g. `python edge/securePi.py "
               "@lobby.args`. Presets live in the presets/ folder next to this script; "
               "presets/common.args is applied automatically as the base. Flags given "
               "after the preset override it.",
    )
    d = Config()  # single source of truth for defaults
    p.add_argument("--model", default=d.model_path,
                   help="Path to the IMX500 .rpk network (default: COCO SSD MobileNetV2).")
    p.add_argument("--labels", default=d.labels_path,
                   help="Optional path to a labels file overriding the model's built-in labels.")
    p.add_argument("--unattended-time", type=float, default=d.unattended_time_sec,
                   help="Seconds before an unattended object triggers an alert "
                        "(default: %(default)s).")
    p.add_argument("--proximity", type=float, default=d.person_proximity_px,
                   help="Max pixel distance for the owner to attend a bag "
                        "(default: %(default)s).")
    p.add_argument("--owner-claim-time", type=float, default=d.owner_claim_sec,
                   help="Seconds after a bag first appears during which a nearby person is "
                        "adopted as its owner (default: %(default)s). Only the owner "
                        "attending resets the unattended timer; passers-by and bystanders "
                        "are ignored.")
    p.add_argument("--stationary-radius", type=float, default=d.stationary_radius,
                   help="Association radius in px: how far a bag may move between "
                        "detections and still match the same track (default: %(default)s). "
                        "Raise it if moving a bag spawns a second box; lower it if "
                        "two nearby bags get merged.")
    p.add_argument("--timeout", type=float, default=d.track_timeout_sec,
                   help="Coast window: seconds a bag track survives detection dropouts "
                        "before being dropped (default: %(default)s). Raise it if static "
                        "bags vanish; the unattended timer keeps running while coasting.")
    p.add_argument("--min-confidence", type=float, default=d.min_confidence,
                   help="Minimum person/object detection confidence 0..1 (default: %(default)s).")
    p.add_argument("--class-confidence", nargs="+", default=[], metavar="LABEL=VALUE",
                   help="Per-label confidence overrides layered on top of "
                        "--min-confidence (person/object labels) or "
                        "--pest-confidence (pest labels), e.g. "
                        "--class-confidence backpack=0.55 mouse=0.45. Repeat to set "
                        f"multiple labels (built-in overrides: {d.class_confidence}).")
    p.add_argument("--box-smoothing", type=float, default=d.box_smoothing,
                   help="Weight of the newest detection when smoothing drawn boxes, "
                        "0..1 (default: %(default)s). Lower = steadier boxes on static "
                        "objects; 1.0 disables smoothing.")
    p.add_argument("--iou", type=float, default=d.iou,
                   help="NMS IoU threshold for nanodet models (default: %(default)s).")
    p.add_argument("--max-detections", type=int, default=d.max_detections,
                   help="Max detections for nanodet models (default: %(default)s).")
    # --bag-labels kept as a backward-compatible alias of --unattended-object-labels.
    p.add_argument("--unattended-object-labels", "--bag-labels", nargs="+",
                   dest="unattended_object_labels",
                   default=sorted(d.unattended_object_labels),
                   help="Labels treated as unattended-object candidates "
                        "(default: %(default)s). NEVER include rat/mouse here — those are "
                        "pests (--pest-labels).")
    p.add_argument("--person-labels", nargs="+", default=sorted(d.person_labels),
                   help="Labels treated as people (default: %(default)s).")
    # --- Pest detection (rat/mouse) --------------------------------------
    p.add_argument("--pest-labels", nargs="+", default=sorted(d.pest_labels),
                   help="Labels treated as pests / rodents (default: %(default)s). Uses "
                        "separate pest logic — no owner association. Meaningful only with "
                        "the custom model (default COCO 'mouse' is a computer mouse).")
    p.add_argument("--pest-confidence", type=float, default=d.pest_confidence,
                   help="Confidence threshold for pest detections 0..1 (default: %(default)s).")
    p.add_argument("--pest-confirmation-time", type=float, default=d.pest_confirmation_time,
                   help="Seconds a pest must stay visible to be confirmed (default: "
                        "%(default)s). Set 0 to confirm by frame count only.")
    p.add_argument("--pest-confirmation-frames", type=int, default=d.pest_confirmation_frames,
                   help="Consecutive frames that alternatively confirm a pest (default: "
                        "%(default)s). Set 0 to confirm by duration only.")
    p.add_argument("--pest-alert-cooldown", type=float, default=d.pest_alert_cooldown_sec,
                   help="Seconds between repeat alerts for the same pest (default: %(default)s).")
    p.add_argument("--pest-match-radius", type=float, default=d.pest_match_radius,
                   help="Px gate for matching a pest detection to its track between frames "
                        "(default: %(default)s). Separate from --stationary-radius (bag-"
                        "tuned) because a moving rat/mouse displaces further between frames "
                        "than a stationary bag; raise it if a real pest's track keeps "
                        "breaking before it can be confirmed.")
    # --- Mode / runtime output -------------------------------------------
    p.add_argument("--headless", action="store_true",
                   help="Run without a preview window; only save alert snapshots.")
    p.add_argument("--zone", default=d.zone,
                   help="Location/preset tag recorded in the event log and used to group "
                        "snapshots under <runtime>/snapshots/<zone> (default: none).")
    p.add_argument("--runtime-dir", type=Path, default=d.runtime_dir,
                   help="Root directory for runtime output — snapshots/, logs/, alerts/ "
                        "(default: %(default)s). Override with an absolute path for systemd.")
    p.add_argument("--snapshot-dir", type=Path, default=None,
                   help="Override the snapshot directory (default: <runtime-dir>/snapshots"
                        "[/<zone>]).")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Override the event-log directory (default: <runtime-dir>/logs).")
    p.add_argument("--max-snapshots", type=int, default=d.max_snapshots,
                   help="Keep at most this many snapshots; oldest are deleted so long "
                        "runs can't fill the SD card (default: %(default)s).")
    p.add_argument("--alert-cooldown", type=float, default=d.alert_cooldown_sec,
                   help="Seconds between repeat alerts for the same bag "
                        "(default: %(default)s).")
    # --- MJPEG streaming (independent of the FlowGuard cloud integration) ---
    p.add_argument("--stream", action="store_true",
                   help="Serve the annotated frames as an MJPEG HTTP stream from "
                        "inside this process (endpoints: /video_feed, /health). "
                        "Off by default.")
    p.add_argument("--stream-host", default=d.stream_host,
                   help="Interface the stream server binds to (default: %(default)s).")
    p.add_argument("--stream-port", type=int, default=d.stream_port,
                   help="Port for the stream server (default: %(default)s).")
    p.add_argument("--stream-fps", type=float, default=d.stream_fps,
                   help="Annotated frames published to the stream per second "
                        "(default: %(default)s). Detection always runs at full "
                        "camera rate regardless.")
    p.add_argument("--stream-quality", type=int, default=d.stream_quality,
                   help="JPEG quality for streamed frames, 1-100 (default: %(default)s).")
    # --- FlowGuard cloud integration (edge -> backend edge-ingest endpoint) ---
    # The Pi NEVER calls WhatsApp directly — it only POSTs events to FlowGuard.
    # The ingest token is read from the EDGE_INGEST_TOKEN env var ONLY (never a CLI
    # flag, so it can't leak into the process list). Defaults come from env so a
    # systemd EnvironmentFile can drive everything without editing presets.
    p.add_argument("--flowguard-enabled", dest="flowguard_enabled", action="store_true",
                   default=_env_flag("FLOWGUARD_EDGE_ENABLED"),
                   help="Send detection events to the FlowGuard backend. Default from "
                        "FLOWGUARD_EDGE_ENABLED (off = fully offline: local snapshots + CSV only).")
    p.add_argument("--flowguard-url", dest="flowguard_url",
                   default=os.environ.get("FLOWGUARD_API_URL"),
                   help="FlowGuard backend base URL (default from FLOWGUARD_API_URL).")
    p.add_argument("--device-id", dest="device_id",
                   default=os.environ.get("SECUREPI_DEVICE_ID"),
                   help="Stable device id used in the deterministic event_id "
                        "(default from SECUREPI_DEVICE_ID, else securepi[-<zone>]).")
    p.add_argument("--camera-location", dest="camera_location",
                   default=os.environ.get("SECUREPI_CAMERA_LOCATION"),
                   help="Human camera label sent to FlowGuard "
                        "(default from SECUREPI_CAMERA_LOCATION).")
    p.add_argument("--outbox-dir", dest="outbox_dir",
                   default=os.environ.get("SECUREPI_OUTBOX_DIR"),
                   help="Disk-backed outbox directory (default <runtime-dir>/alerts/outbox "
                        "or SECUREPI_OUTBOX_DIR).")
    p.add_argument("--hours-start", "--restricted-hours-start", dest="restricted_hours_start",
                   default=os.environ.get("RESTRICTED_HOURS_START", d.restricted_hours_start),
                   help="Restricted hours start HH:MM, Singapore time (default from RESTRICTED_HOURS_START or 08:00).")
    p.add_argument("--hours-end", "--restricted-hours-end", dest="restricted_hours_end",
                   default=os.environ.get("RESTRICTED_HOURS_END", d.restricted_hours_end),
                   help="Restricted hours end HH:MM, Singapore time (default from RESTRICTED_HOURS_END or 23:00).")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    raw = sys.argv[1:] if argv is None else list(argv)
    expanded = _resolve_preset_refs(raw)
    base = PRESETS_DIR / "common.args"
    if base.exists():
        # Shared base first, so the user's preset and CLI flags override it.
        expanded = [f"@{base}"] + expanded
    args = p.parse_args(expanded)   # parse first so -h/--help still works
    _require_args_file(p, raw)
    if args.stream_fps <= 0:
        p.error("--stream-fps must be greater than 0")
    if not 1 <= args.stream_quality <= 100:
        p.error("--stream-quality must be between 1 and 100")
    if not 0 <= args.stream_port <= 65535:
        p.error("--stream-port must be between 0 and 65535")
    return args


def _merge_class_confidence(pairs: list[str], base: dict[str, float]) -> dict[str, float]:
    """Merge --class-confidence LABEL=VALUE pairs onto the built-in defaults."""
    merged = dict(base)
    for pair in pairs:
        label, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"--class-confidence expects LABEL=VALUE, got {pair!r}")
        merged[label.strip().lower()] = float(value)
    return merged


_FACE_INFO_CACHE = {}
_FACE_LAST_CHECK = {}
_FACE_CHECK_INTERVAL_SEC = 3.0


def reset_face_cache() -> None:
    """Clear cached facial recognition results (for tests)."""
    _FACE_INFO_CACHE.clear()
    _FACE_LAST_CHECK.clear()


def _recognize_person_with_flowguard(frame, box, person_id):
    """Call FlowGuard facial recognition and cache structured identity info per SecurePi person track."""
    api_url = os.environ.get("FLOWGUARD_API_URL", "").rstrip("/")
    edge_token = os.environ.get("EDGE_SERVICE_TOKEN", "")

    now = time.monotonic()
    cached = _FACE_INFO_CACHE.get(person_id)
    last_check = _FACE_LAST_CHECK.get(person_id, 0)

    if cached and (now - last_check < _FACE_CHECK_INTERVAL_SEC):
        return cached

    if not api_url or not edge_token:
        info = {
            "identity_status": "UNAVAILABLE",
            "person_name": None,
            "person_role": None,
            "confidence": None,
            "display_label": f"#{person_id} Identity unavailable",
        }
        _FACE_INFO_CACHE[person_id] = info
        return info

    _FACE_LAST_CHECK[person_id] = now

    x, y, w, h = box
    fh, fw = frame.shape[:2]

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(fw, int(x + w))
    y2 = min(fh, int(y + h))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return cached or {
            "identity_status": "UNAVAILABLE",
            "person_name": None,
            "person_role": None,
            "confidence": None,
            "display_label": f"#{person_id} Identity unavailable",
        }

    ok, buffer = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    if not ok:
        return cached or {
            "identity_status": "UNAVAILABLE",
            "person_name": None,
            "person_role": None,
            "confidence": None,
            "display_label": f"#{person_id} Identity unavailable",
        }

    image_b64 = base64.b64encode(buffer).decode("ascii")
    payload = json.dumps({
        "image": f"data:image/jpeg;base64,{image_b64}",
        "cameraLocation": os.environ.get("SECUREPI_CAMERA_LOCATION", "SecurePi Camera"),
    }).encode("utf-8")

    request = urllib.request.Request(
        f"{api_url}/api/facial-recognition/recognize",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-edge-token": edge_token,
        },
    )

    try:
        LOGGER.debug("[Face] track #%s recognition requested", person_id)
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        LOGGER.warning("[Face] track #%s recognition service unavailable: %s", person_id, exc)
        info = {
            "identity_status": "UNAVAILABLE",
            "person_name": None,
            "person_role": None,
            "confidence": None,
            "display_label": f"#{person_id} Identity unavailable",
        }
        _FACE_INFO_CACHE[person_id] = info
        return info

    user = data.get("user") or {}
    name = user.get("name")
    status = user.get("status")
    role = user.get("role") or user.get("person_role")
    conf = user.get("confidence")
    try:
        conf_float = float(conf) if conf is not None else None
    except (ValueError, TypeError):
        conf_float = None

    if status == "AUTHORIZED" and name and name != "Unknown Person":
        info = {
            "identity_status": "VERIFIED",
            "person_name": name,
            "person_role": role or "Staff",
            "confidence": conf_float,
            "display_label": f"#{person_id} {name} - VERIFIED",
        }
        LOGGER.info("[Face] track #%s VERIFIED name=%s confidence=%s", person_id, name, conf_float)
    elif status == "SUSPENDED" and name:
        info = {
            "identity_status": "SUSPENDED",
            "person_name": name,
            "person_role": role or "Staff",
            "confidence": conf_float,
            "display_label": f"#{person_id} {name} - SUSPENDED",
        }
        LOGGER.info("[Face] track #%s SUSPENDED name=%s", person_id, name)
    elif status == "DENIED" or name == "Unknown Person":
        info = {
            "identity_status": "SUSPICIOUS",
            "person_name": "Unknown Person",
            "person_role": None,
            "confidence": conf_float,
            "display_label": f"#{person_id} Unknown Person - SUSPICIOUS",
        }
        LOGGER.info("[Face] track #%s SUSPICIOUS / Unknown Person", person_id)
    else:
        info = {
            "identity_status": "SUSPICIOUS",
            "person_name": "Unknown Person",
            "person_role": None,
            "confidence": conf_float,
            "display_label": f"#{person_id} Unknown Person - SUSPICIOUS",
        }
        LOGGER.info("[Face] track #%s mapped status=%s name=%s -> SUSPICIOUS", person_id, status, name)

    _FACE_INFO_CACHE[person_id] = info
    return info


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    edge_service_tok = os.environ.get("EDGE_SERVICE_TOKEN")
    edge_ingest_tok = os.environ.get("EDGE_INGEST_TOKEN")
    LOGGER.info("[FlowGuard] Facial recognition token: %s", "configured" if edge_service_tok else "missing")
    LOGGER.info("[FlowGuard] Edge alert ingest token: %s", "configured" if edge_ingest_tok else "missing")
    config = Config(
        unattended_time_sec=args.unattended_time,
        stationary_radius=args.stationary_radius,
        person_proximity_px=args.proximity,
        owner_claim_sec=args.owner_claim_time,
        track_timeout_sec=args.timeout,
        min_confidence=args.min_confidence,
        class_confidence=_merge_class_confidence(args.class_confidence, Config().class_confidence),
        box_smoothing=args.box_smoothing,
        headless=args.headless,
        alert_cooldown_sec=args.alert_cooldown,
        max_snapshots=args.max_snapshots,
        runtime_dir=args.runtime_dir,
        zone=args.zone,
        snapshot_dir=args.snapshot_dir,
        log_dir=args.log_dir,
        model_path=_resolve_project_path(args.model),
        labels_path=_resolve_project_path(args.labels),
        iou=args.iou,
        max_detections=args.max_detections,
        unattended_object_labels=set(args.unattended_object_labels),
        person_labels=set(args.person_labels),
        pest_labels=set(args.pest_labels),
        pest_confidence=args.pest_confidence,
        pest_confirmation_time=args.pest_confirmation_time,
        pest_confirmation_frames=args.pest_confirmation_frames,
        pest_alert_cooldown_sec=args.pest_alert_cooldown,
        pest_match_radius=args.pest_match_radius,
        stream_enabled=args.stream,
        stream_host=args.stream_host,
        stream_port=args.stream_port,
        stream_fps=args.stream_fps,
        stream_quality=args.stream_quality,
        restricted_hours_start=args.restricted_hours_start,
        restricted_hours_end=args.restricted_hours_end,
    )
    client = _build_flowguard_client(args, config)
    run(config, client=client)


def _build_flowguard_client(args, config: Config):
    """Construct the FlowGuard edge client when integration is enabled, else None.

    Disabled (the default) means the monitor runs fully offline: no client, no
    network calls, and no outbox files are created."""
    if not (_FLOWGUARD_AVAILABLE and getattr(args, "flowguard_enabled", False)):
        return None
    token = os.environ.get("EDGE_INGEST_TOKEN")
    outbox = args.outbox_dir or str(config.runtime_dir / "alerts" / "outbox")
    device_id = args.device_id or (f"securepi-{config.zone}" if config.zone else "securepi")
    client = FlowGuardApiClient(
        api_url=args.flowguard_url,
        token=token,
        enabled=True,
        device_id=device_id,
        camera_location=args.camera_location,
        timeout=float(os.environ.get("SECUREPI_HTTP_TIMEOUT_SEC", "5") or 5),
        retry_interval=float(os.environ.get("SECUREPI_RETRY_INTERVAL_SEC", "30") or 30),
        outbox_dir=outbox,
    )
    if not args.flowguard_url or not token:
        LOGGER.warning("[FlowGuard] Integration enabled but URL/token incomplete — events "
                       "will queue in the outbox until FLOWGUARD_API_URL / EDGE_INGEST_TOKEN are set.")
    return client


if __name__ == "__main__":
    main()
