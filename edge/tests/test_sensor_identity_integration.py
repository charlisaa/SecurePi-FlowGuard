"""Comprehensive test suite for Pi5 sensor & identity integration.

Tests cover all 24 contract requirements:
1. PIR trigger opens inspection.
2. Ultrasonic distance change opens inspection.
3. PIR warm-up does not trigger.
4. Restricted-hours policy works.
5. Sensor alone never creates a person identity.
6. Person during inspection invokes facial recognition.
7. VERIFIED stores real name and role.
8. SUSPICIOUS represents genuine unknown response.
9. SUSPENDED remains distinct.
10. UNAVAILABLE remains distinct (never mapped to UNKNOWN).
11. Recognition rate-limited by track.
12. Sensor metadata enters outgoing edge payload.
13. Event ID remains stable on retry.
14. Repeated same inspection does not spam alerts.
15. New inspection creates new event ID.
16. Snapshot uses existing annotated frame.
17-19. Bag, pest, and unattended object flows remain working.
20-23. /video_feed, /health, /people-count, /snapshot endpoints work.
24. Outbox/retry tests remain green.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import types
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure cv2 stubbed if not installed
_draw_calls = []
_cv2 = types.SimpleNamespace(
    imwrite=lambda path, img: Path(path).write_bytes(b"jpg") > 0,
    imencode=lambda ext, img, params=None: (True, b"jpg"),
    rectangle=lambda frame, p1, p2, color, thickness: _draw_calls.append(("rect", p1, p2, color, thickness)),
    putText=lambda frame, text, org, font, scale, color, thickness: _draw_calls.append(("text", text, color)),
    FONT_HERSHEY_SIMPLEX=0,
    IMWRITE_JPEG_QUALITY=1,
)
cv2_mod = sys.modules.setdefault("cv2", _cv2)
if not hasattr(cv2_mod, "imencode"):
    setattr(cv2_mod, "imencode", lambda ext, img, params=None: (True, b"jpg"))
if not hasattr(cv2_mod, "IMWRITE_JPEG_QUALITY"):
    setattr(cv2_mod, "IMWRITE_JPEG_QUALITY", 1)

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flowguard_api import FlowGuardApiClient, build_event_id
from sensor_bridge import SensorBridge, in_restricted_hours, parse_sensor_line, SGT
from securePi import (
    Config, Detection, PersonTracker, BagTracker, PestTracker, PersonTrack, TrackedPest,
    _recognize_person_with_flowguard, reset_face_cache, _fire_sensor_person_alert,
    _fire_alert, _fire_pest_alert,
    StreamServer, FrameBuffer, FACE_EXECUTOR, SNAPSHOT_EXECUTOR
)


def _drain_snapshots():
    """Block until the single-worker snapshot/log executor has flushed, so async
    disk writes cannot race a TemporaryDirectory cleanup (Windows WinError 145)."""
    SNAPSHOT_EXECUTOR.submit(lambda: None).result()


def DummyFrame():
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _drain_face():
    """Block until FACE_EXECUTOR's single worker has flushed (recognition runs
    off the calling thread now; tests need the result before asserting)."""
    FACE_EXECUTOR.submit(lambda: None).result()


# --------------------------------------------------------------------------
# 1. PIR trigger opens inspection
# 2. Ultrasonic distance change opens inspection
# 3. PIR warm-up does not trigger
# 4. Restricted-hours policy works
# 5. Sensor alone never creates a person identity
# --------------------------------------------------------------------------

def test_1_pir_trigger_opens_inspection():
    bridge = SensorBridge(hours_start="00:00", hours_end="23:59")
    now = datetime.now(SGT)
    line = json.dumps({
        "type": "sensor_status",
        "motion": True,
        "pir_ready": True,
        "uptime_ms": 1000
    })
    bridge.process_line(line, now=now)
    assert bridge.is_inspection_active(now=now) is True
    assert bridge.latest_sensor_metadata.get("trigger") == "PIR"


def test_2_ultrasonic_distance_change_opens_inspection():
    bridge = SensorBridge(hours_start="00:00", hours_end="23:59")
    now = datetime.now(SGT)
    line = json.dumps({
        "type": "sensor_status",
        "motion": False,
        "pir_ready": True,
        "distance_cm": 40.0,
        "baseline_distance_cm": 120.0,
        "distance_change_cm": 80.0,
        "object_close": True,
        "trigger": "ULTRASONIC"
    })
    bridge.process_line(line, now=now)
    assert bridge.is_inspection_active(now=now) is True
    assert bridge.latest_sensor_metadata.get("distance_change_cm") == 80.0


def test_3_pir_warmup_does_not_trigger():
    bridge = SensorBridge(hours_start="00:00", hours_end="23:59")
    now = datetime.now(SGT)
    line = json.dumps({
        "type": "sensor_status",
        "motion": True,
        "pir_ready": False
    })
    res = bridge.process_line(line, now=now)
    assert res is None
    assert bridge.is_inspection_active(now=now) is False


def test_4_restricted_hours_policy():
    # Restricted window 22:00 to 06:00
    bridge = SensorBridge(hours_start="22:00", hours_end="06:00")

    # Day time: 12:00 -> not restricted
    day_dt = datetime.now(SGT).replace(hour=12, minute=0, second=0)
    line = json.dumps({"type": "sensor_status", "motion": True, "pir_ready": True})
    bridge.process_line(line, now=day_dt)
    assert bridge.is_inspection_active(now=day_dt) is False

    # Night time: 23:00 -> restricted
    night_dt = datetime.now(SGT).replace(hour=23, minute=0, second=0)
    bridge._prev_motion = False
    bridge.process_line(line, now=night_dt)
    assert bridge.is_inspection_active(now=night_dt) is True


def test_5_sensor_alone_never_creates_person_identity():
    bridge = SensorBridge(hours_start="00:00", hours_end="23:59")
    now = datetime.now(SGT)
    line = json.dumps({"type": "sensor_status", "motion": True, "pir_ready": True})
    bridge.process_line(line, now=now)
    assert bridge.is_inspection_active(now=now) is True
    # Sensor bridge only tracks state metadata, never invokes face recognition or creates person tracks
    assert "person_name" not in bridge.latest_sensor_metadata


# --------------------------------------------------------------------------
# 6. Person during inspection invokes facial recognition
# 7. VERIFIED stores real name and role
# 8. SUSPICIOUS represents genuine unknown response
# 9. SUSPENDED remains distinct
# 10. UNAVAILABLE remains distinct (never mapped to UNKNOWN)
# 11. Recognition rate-limited by track
# --------------------------------------------------------------------------

def test_6_7_8_9_10_identity_mappings_and_rate_limiting():
    reset_face_cache()
    frame = DummyFrame()
    box = (10, 10, 100, 100)
    person_id = 42

    with patch.dict(os.environ, {
        "FLOWGUARD_API_URL": "http://mock-flowguard:5001",
        "EDGE_SERVICE_TOKEN": "valid-token-123"
    }):
        # 7. Authorized / Active user -> VERIFIED
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "user": {"name": "Felicia", "status": "AUTHORIZED", "role": "Staff", "confidence": 0.95}
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp) as mock_url:
            pending = _recognize_person_with_flowguard(frame, box, person_id)
            assert pending["identity_status"] == "UNAVAILABLE"  # dispatched, not resolved yet
            _drain_face()
            info = _recognize_person_with_flowguard(frame, box, person_id)
            assert info["identity_status"] == "VERIFIED"
            assert info["person_name"] == "Felicia"
            assert info["person_role"] == "Staff"
            assert info["display_label"] == "#42 Felicia - VERIFIED"
            assert mock_url.call_count == 1

        # 11. Rate-limiting check: within check interval returns cached without network call
        with patch("urllib.request.urlopen") as mock_url_cached:
            cached_info = _recognize_person_with_flowguard(frame, box, person_id)
            assert cached_info == info
            assert mock_url_cached.call_count == 0

        # Reset cache for test 8 (SUSPICIOUS / Unknown Person)
        reset_face_cache()
        mock_resp_denied = MagicMock()
        mock_resp_denied.read.return_value = json.dumps({
            "user": {"name": "Unknown Person", "status": "DENIED"}
        }).encode("utf-8")
        mock_resp_denied.__enter__.return_value = mock_resp_denied

        with patch("urllib.request.urlopen", return_value=mock_resp_denied):
            _recognize_person_with_flowguard(frame, box, 43)
            _drain_face()
            info_denied = _recognize_person_with_flowguard(frame, box, 43)
            assert info_denied["identity_status"] == "SUSPICIOUS"
            assert info_denied["person_name"] == "Unknown Person"
            assert info_denied["display_label"] == "#43 Unknown Person - SUSPICIOUS"

        # Reset cache for test 9 (SUSPENDED user)
        reset_face_cache()
        mock_resp_suspended = MagicMock()
        mock_resp_suspended.read.return_value = json.dumps({
            "user": {"name": "John Doe", "status": "SUSPENDED", "role": "Contractor"}
        }).encode("utf-8")
        mock_resp_suspended.__enter__.return_value = mock_resp_suspended

        with patch("urllib.request.urlopen", return_value=mock_resp_suspended):
            _recognize_person_with_flowguard(frame, box, 44)
            _drain_face()
            info_suspended = _recognize_person_with_flowguard(frame, box, 44)
            assert info_suspended["identity_status"] == "SUSPENDED"
            assert info_suspended["person_name"] == "John Doe"
            assert info_suspended["display_label"] == "#44 John Doe - SUSPENDED"

        # Reset cache for test 10 (Network error / Timeout -> UNAVAILABLE)
        reset_face_cache()
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Network Timeout")):
            _recognize_person_with_flowguard(frame, box, 45)
            _drain_face()
            info_unavail = _recognize_person_with_flowguard(frame, box, 45)
            assert info_unavail["identity_status"] == "UNAVAILABLE"
            assert info_unavail["person_name"] is None
            assert info_unavail["display_label"] == "#45 Identity unavailable"
            # Crucial: UNAVAILABLE must never be mapped to UNKNOWN or SUSPICIOUS!
            assert info_unavail["identity_status"] != "SUSPICIOUS"
            assert info_unavail["person_name"] != "Unknown Person"


# --------------------------------------------------------------------------
# 12. Sensor metadata enters outgoing edge payload
# 13. Event ID remains stable on retry
# 14. Repeated same inspection does not spam alerts
# 15. New inspection creates new event ID
# 16. Snapshot uses existing annotated frame
# --------------------------------------------------------------------------

def test_12_13_14_15_16_edge_alert_payload_and_idempotency():
    with tempfile.TemporaryDirectory() as tmp:
        client = FlowGuardApiClient(
            api_url="http://mock-flowguard:5001",
            token="test-ingest-token",
            enabled=True,
            device_id="securepi-bay1",
            camera_location="Loading Bay 01",
            outbox_dir=Path(tmp) / "outbox",
            auto_flush=False,
        )

        sensor_metadata = {
            "pir": True,
            "motion": True,
            "pir_ready": True,
            "distance_cm": 42.5,
            "baseline_distance_cm": 120.0,
            "distance_change_cm": 77.5,
            "object_close": True,
            "trigger": "PIR",
            "inspection_active": True,
            "after_hours": True,
        }

        identity_info = {
            "identity_status": "VERIFIED",
            "person_name": "Felicia",
            "person_role": "Staff",
            "confidence": 0.91,
        }

        class MockPerson:
            person_id = 3
            score = 0.91

        cfg = Config(runtime_dir=Path(tmp))
        cfg.snapshot_dir = Path(tmp) / "snapshots"
        cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)

        renderer = MagicMock()

        # Fire alert
        _fire_sensor_person_alert(
            DummyFrame(), MockPerson(), identity_info, sensor_metadata,
            cfg, time.monotonic(), renderer, client=client
        )

        files = list((Path(tmp) / "outbox").glob("*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text(encoding="utf-8"))
        assert data["event_type"] == "restricted_motion"
        assert data["alert_type"] == "Restricted-Zone Motion"
        assert data["object_class"] == "person"
        assert data["person_name"] == "Felicia"
        assert data["identity_status"] == "VERIFIED"
        assert data["person_role"] == "Staff"
        assert data["track_id"] == 3
        assert data["sensor_metadata"]["distance_cm"] == 42.5

        # Test 13: Event ID format
        assert data["event_id"].startswith("securepi-bay1:restricted_motion:3:")

        # Test 16: Snapshot path recorded
        assert "snapshot_path" in data


# --------------------------------------------------------------------------
# A1-7. Restricted-person snapshot + explicit timestamp + inspection-scoped
# stable event_id (feature/sensor-identity-integration timestamp contract)
# --------------------------------------------------------------------------

def test_A_restricted_person_snapshot_timestamp_and_stable_event_id():
    with tempfile.TemporaryDirectory() as tmp:
        client = FlowGuardApiClient(
            api_url="http://mock-flowguard:5001", token="test-ingest-token", enabled=True,
            device_id="securepi-bay1", camera_location="Loading Bay 01",
            outbox_dir=Path(tmp) / "outbox", auto_flush=False,
        )
        cfg = Config(runtime_dir=Path(tmp))
        cfg.snapshot_dir = Path(tmp) / "snapshots"
        cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        renderer = MagicMock()

        identity_info = {
            "identity_status": "VERIFIED", "person_name": "Felicia",
            "person_role": "Staff", "confidence": 0.91,
        }
        person = PersonTrack(person_id=7, centroid=(0.0, 0.0), box=(1, 1, 2, 2), last_seen=0.0)
        sensor_meta_insp1 = {"inspection_active": True, "inspection_id": "insp_AAA", "trigger": "PIR"}

        before = datetime.now(timezone.utc)
        _fire_sensor_person_alert(DummyFrame(), person, identity_info, sensor_meta_insp1,
                                  cfg, time.monotonic(), renderer, client=client)
        after = datetime.now(timezone.utc)

        files = sorted((Path(tmp) / "outbox").glob("*.json"))
        assert len(files) == 1
        event1 = json.loads(files[0].read_text())

        # 1. snapshot_path present.
        assert event1.get("snapshot_path", "").endswith(".jpg")
        # 2 & 3. explicit UTC ISO-8601 timestamp, within this call's wall-clock window.
        ts = datetime.strptime(event1["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        assert before - timedelta(seconds=2) <= ts <= after + timedelta(seconds=2)
        first_event_id = event1["event_id"]

        # 4. SAME inspection + SAME person -> repeat firing reuses the identical
        # event_id (overwrites the same outbox file rather than adding a second).
        _fire_sensor_person_alert(DummyFrame(), person, identity_info, sensor_meta_insp1,
                                  cfg, time.monotonic(), renderer, client=client)
        files_same = sorted((Path(tmp) / "outbox").glob("*.json"))
        assert len(files_same) == 1
        assert json.loads(files_same[0].read_text())["event_id"] == first_event_id

        # 5. A genuinely NEW inspection gets a fresh event_id (a second outbox file).
        sensor_meta_insp2 = {"inspection_active": True, "inspection_id": "insp_BBB", "trigger": "PIR"}
        _fire_sensor_person_alert(DummyFrame(), person, identity_info, sensor_meta_insp2,
                                  cfg, time.monotonic(), renderer, client=client)
        files_new = sorted((Path(tmp) / "outbox").glob("*.json"))
        assert len(files_new) == 2
        all_ids = {json.loads(f.read_text())["event_id"] for f in files_new}
        assert first_event_id in all_ids and len(all_ids) == 2

        # 6 & 7. Identity state and severity mapping unchanged.
        assert event1["identity_status"] == "VERIFIED"
        assert event1["person_name"] == "Felicia"
        assert event1["severity"] == "High"
        client.stop()


def test_A_restricted_person_severity_critical_for_suspicious_and_suspended():
    with tempfile.TemporaryDirectory() as tmp:
        client = FlowGuardApiClient(
            api_url="http://mock-flowguard:5001", token="test-ingest-token", enabled=True,
            device_id="securepi-bay1", camera_location="Loading Bay 01",
            outbox_dir=Path(tmp) / "outbox", auto_flush=False,
        )
        cfg = Config(runtime_dir=Path(tmp))
        cfg.snapshot_dir = Path(tmp) / "snapshots"
        cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        renderer = MagicMock()

        for status, person_id in (("SUSPICIOUS", 11), ("SUSPENDED", 12)):
            identity_info = {"identity_status": status, "person_name": "Unknown Person",
                             "person_role": None, "confidence": None}
            person = PersonTrack(person_id=person_id, centroid=(0.0, 0.0), box=(1, 1, 2, 2), last_seen=0.0)
            _fire_sensor_person_alert(DummyFrame(), person, identity_info,
                                      {"inspection_id": f"insp_{person_id}"},
                                      cfg, time.monotonic(), renderer, client=client)
        files = sorted((Path(tmp) / "outbox").glob("*.json"))
        events = [json.loads(f.read_text()) for f in files]
        assert all(e["severity"] == "Critical" for e in events)
        client.stop()


def test_22_sensorbridge_never_fabricates_snapshot_identity_or_class():
    bridge = SensorBridge(hours_start="00:00", hours_end="23:59", dry_run=True)
    now = datetime.now(SGT)
    line = json.dumps({"type": "sensor_status", "motion": True, "pir_ready": True})
    event = bridge.process_line(line, now=now)
    assert event is not None
    assert "snapshot_path" not in event      # SensorBridge owns no camera
    assert "person_name" not in event        # never invents an identity
    assert "object_class" not in event       # never invents a pest/object class


# --------------------------------------------------------------------------
# Sensor-trigger + camera-confirmation single-alert pipeline
# (feature/sensor-identity-integration dedup fix: SensorBridge is the TRIGGER,
# securePi.py's camera-confirmed alert is the sole authoritative DetectionAlert)
# --------------------------------------------------------------------------

def test_sensor_camera_pipeline_single_authoritative_alert():
    with tempfile.TemporaryDirectory() as tmp:
        client = FlowGuardApiClient(
            api_url="http://mock-flowguard:5001", token="tok", enabled=True,
            device_id="securepi-bay1", camera_location="Loading Bay 01",
            outbox_dir=Path(tmp) / "outbox", auto_flush=False,
        )
        outbox = Path(tmp) / "outbox"

        # A camera-attached bridge, exactly as securePi.py::run() constructs it.
        bridge = SensorBridge(client=client, zone_name="Loading Bay",
                              hours_start="00:00", hours_end="23:59",
                              send_raw_alerts=False)
        t0 = datetime.now(SGT)

        # 1. Raw sensor trigger starts an inspection.
        line_motion = json.dumps({"type": "sensor_status", "motion": True, "pir_ready": True})
        raw_result = bridge.process_line(line_motion, now=t0)

        # 2. The raw trigger alone creates NO user-facing FlowGuard alert.
        assert raw_result is None
        assert list(outbox.glob("*.json")) == []

        # 3. Sensor telemetry still works even with alerts suppressed.
        status = bridge.get_sensor_status(now=t0)
        assert status["inspection_active"] is True
        assert status["inspection_id"] is not None
        assert status["pir"] is True
        assert status["connected"] is True

        inspection_id = status["inspection_id"]
        sensor_metadata = dict(bridge.latest_sensor_metadata)
        assert sensor_metadata["inspection_id"] == inspection_id

        # 10. No visual confirmation yet -> still no fabricated detection alert.
        assert list(outbox.glob("*.json")) == []

        # --- Camera now confirms a person during this same inspection. ---
        cfg = Config(runtime_dir=Path(tmp))
        cfg.snapshot_dir = Path(tmp) / "snapshots"
        cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        renderer = MagicMock()
        identity_info = {"identity_status": "VERIFIED", "person_name": "Felicia",
                         "person_role": "Staff", "confidence": 0.91}
        person = PersonTrack(person_id=5, centroid=(0.0, 0.0), box=(1, 1, 2, 2), last_seen=0.0)

        before = datetime.now(timezone.utc)
        _fire_sensor_person_alert(DummyFrame(), person, identity_info, sensor_metadata,
                                  cfg, time.monotonic(), renderer, client=client)
        after = datetime.now(timezone.utc)

        # 4. Exactly ONE FlowGuard alert exists for this physical occurrence.
        files = list(outbox.glob("*.json"))
        assert len(files) == 1
        event = json.loads(files[0].read_text())

        # 5. snapshot_path present.
        assert event["snapshot_path"].endswith(".jpg")
        # 6. explicit trigger timestamp, within this call's wall-clock window.
        ts = datetime.strptime(event["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        assert before - timedelta(seconds=2) <= ts <= after + timedelta(seconds=2)
        # 7. inspection_id correlates the alert with its sensor trigger.
        assert event["sensor_metadata"]["inspection_id"] == inspection_id
        first_event_id = event["event_id"]

        # 8. SAME inspection + SAME person -> repeat firing keeps the same event_id
        # (overwrites the one outbox file rather than adding a second).
        _fire_sensor_person_alert(DummyFrame(), person, identity_info, sensor_metadata,
                                  cfg, time.monotonic(), renderer, client=client)
        files_same = list(outbox.glob("*.json"))
        assert len(files_same) == 1
        assert json.loads(files_same[0].read_text())["event_id"] == first_event_id

        # 9. A genuinely NEW inspection (motion drops, then rises again after the
        # inspection window has expired) gets a fresh event_id.
        t1 = t0 + timedelta(seconds=20)   # well past inspection_window_sec (10s default)
        line_drop = json.dumps({"type": "sensor_status", "motion": False, "pir_ready": True})
        assert bridge.process_line(line_drop, now=t1) is None
        t2 = t1 + timedelta(seconds=1)
        assert bridge.process_line(line_motion, now=t2) is None  # still no raw alert
        status2 = bridge.get_sensor_status(now=t2)
        new_inspection_id = status2["inspection_id"]
        assert new_inspection_id is not None and new_inspection_id != inspection_id

        _fire_sensor_person_alert(DummyFrame(), person, identity_info,
                                  dict(bridge.latest_sensor_metadata),
                                  cfg, time.monotonic(), renderer, client=client)
        files_new = list(outbox.glob("*.json"))
        assert len(files_new) == 2
        all_ids = {json.loads(f.read_text())["event_id"] for f in files_new}
        assert first_event_id in all_ids and len(all_ids) == 2

        client.stop()


def test_bag_and_pest_alerts_unaffected_by_sensor_raw_alert_suppression():
    """11. Existing bag/pest behaviour keeps working when a send_raw_alerts=False
    SensorBridge shares the same client/outbox -- the dedup fix must not leak
    into unrelated alert types."""
    with tempfile.TemporaryDirectory() as tmp:
        client = FlowGuardApiClient(
            api_url="http://mock-flowguard:5001", token="tok", enabled=True,
            device_id="securepi-bay1", camera_location="Loading Bay 01",
            outbox_dir=Path(tmp) / "outbox", auto_flush=False,
        )
        outbox = Path(tmp) / "outbox"
        bridge = SensorBridge(client=client, zone_name="Loading Bay",
                              hours_start="00:00", hours_end="23:59",
                              send_raw_alerts=False)
        now = datetime.now(SGT)
        bridge.process_line(json.dumps({"type": "sensor_status", "motion": True, "pir_ready": True}), now=now)
        assert list(outbox.glob("*.json")) == []  # sensor trigger alone: nothing yet

        cfg = Config(runtime_dir=Path(tmp), zone="lobby")
        cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
        renderer = MagicMock()

        bt = BagTracker(cfg)
        bt.update([Detection("backpack", 0.9, (50, 60, 40, 40))], [], now=0.0)
        bag = bt.bags[0]
        bag.unattended_start = 0.0
        _fire_alert(DummyFrame(), bag, cfg, now=cfg.unattended_time_sec + 5, renderer=renderer, client=client)

        pest = TrackedPest(pest_id=1, label="rat", centroid=(5, 5), box=(5, 5, 10, 10),
                           first_seen=0.0, last_seen=1.0, score=0.9)
        _fire_pest_alert(DummyFrame(), pest, cfg, now=1.0, renderer=renderer, client=client)

        files = list(outbox.glob("*.json"))
        assert len(files) == 2   # exactly the bag + pest alerts, nothing from the sensor
        event_types = {json.loads(f.read_text())["event_type"] for f in files}
        assert event_types == {"unattended_object", "pest_detection"}
        client.stop()


# --------------------------------------------------------------------------
# Sensor-identity context on camera-confirmed pest / unattended-object alerts
# (feature: propagate the active-inspection metadata into the SAME authoritative
# detection event, without fabricating context for AI-only detections).
# --------------------------------------------------------------------------

def _client_and_cfg(tmp):
    client = FlowGuardApiClient(
        api_url="http://mock-flowguard:5001", token="tok", enabled=True,
        device_id="securepi-bay1", camera_location="Loading Bay 01",
        outbox_dir=Path(tmp) / "outbox", auto_flush=False,
    )
    cfg = Config(runtime_dir=Path(tmp), zone="lobby")
    cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)
    return client, cfg


def _make_bag(cfg):
    bt = BagTracker(cfg)
    bt.update([Detection("backpack", 0.9, (50, 60, 40, 40))], [], now=0.0)
    bag = bt.bags[0]
    bag.unattended_start = 0.0
    return bag


def _make_pest():
    return TrackedPest(pest_id=1, label="rat", centroid=(5, 5), box=(5, 5, 10, 10),
                       first_seen=0.0, last_seen=1.0, score=0.9)


PIR_META = {
    "trigger": "PIR", "pir": True, "motion": True, "pir_ready": True,
    "distance_cm": 42.5, "distance_change_cm": 77.5, "object_close": True,
    "inspection_active": True, "inspection_id": "insp_PEST_1", "after_hours": True,
}


def test_pest_alert_carries_sensor_metadata_when_supplied():
    """Requirement 1: a camera-confirmed pest alert propagates active-inspection
    sensor metadata into the SAME event when supplied at the call site."""
    with tempfile.TemporaryDirectory() as tmp:
        client, cfg = _client_and_cfg(tmp)
        outbox = Path(tmp) / "outbox"
        _fire_pest_alert(DummyFrame(), _make_pest(), cfg, now=1.0,
                         renderer=MagicMock(), client=client,
                         sensor_metadata=dict(PIR_META))
        files = list(outbox.glob("*.json"))
        assert len(files) == 1
        event = json.loads(files[0].read_text())
        assert event["event_type"] == "pest_detection"
        assert event["object_class"] == "rat"        # class unchanged
        assert event["sensor_metadata"]["trigger"] == "PIR"
        assert event["sensor_metadata"]["distance_change_cm"] == 77.5
        assert event["sensor_metadata"]["inspection_id"] == "insp_PEST_1"
        _drain_snapshots()
        client.stop()


def test_unattended_alert_carries_sensor_metadata_when_supplied():
    """Requirement 2: a camera-confirmed unattended-object alert propagates
    active-inspection sensor metadata into the SAME event when supplied."""
    with tempfile.TemporaryDirectory() as tmp:
        client, cfg = _client_and_cfg(tmp)
        outbox = Path(tmp) / "outbox"
        meta = dict(PIR_META, trigger="ULTRASONIC", inspection_id="insp_BAG_1")
        _fire_alert(DummyFrame(), _make_bag(cfg), cfg,
                    now=cfg.unattended_time_sec + 5, renderer=MagicMock(),
                    client=client, sensor_metadata=meta)
        files = list(outbox.glob("*.json"))
        assert len(files) == 1
        event = json.loads(files[0].read_text())
        assert event["event_type"] == "unattended_object"
        assert event["object_class"] == "backpack"    # class unchanged
        assert event["sensor_metadata"]["trigger"] == "ULTRASONIC"
        assert event["sensor_metadata"]["inspection_id"] == "insp_BAG_1"
        _drain_snapshots()
        client.stop()


def test_pest_alert_without_sensor_metadata_stays_sensor_free():
    """Requirements 3 & 9: an AI-only pest detection (no active inspection) never
    fabricates sensor context -- the payload omits sensor_metadata entirely."""
    with tempfile.TemporaryDirectory() as tmp:
        client, cfg = _client_and_cfg(tmp)
        outbox = Path(tmp) / "outbox"
        _fire_pest_alert(DummyFrame(), _make_pest(), cfg, now=1.0,
                         renderer=MagicMock(), client=client)  # no sensor_metadata
        event = json.loads(list(outbox.glob("*.json"))[0].read_text())
        assert event["event_type"] == "pest_detection"
        assert "sensor_metadata" not in event   # None dropped by build_event
        _drain_snapshots()
        client.stop()


def test_unattended_alert_without_sensor_metadata_stays_sensor_free():
    """Requirements 4 & 9: an AI-only unattended-object detection stays sensor-free."""
    with tempfile.TemporaryDirectory() as tmp:
        client, cfg = _client_and_cfg(tmp)
        outbox = Path(tmp) / "outbox"
        _fire_alert(DummyFrame(), _make_bag(cfg), cfg,
                    now=cfg.unattended_time_sec + 5, renderer=MagicMock(),
                    client=client)  # no sensor_metadata
        event = json.loads(list(outbox.glob("*.json"))[0].read_text())
        assert event["event_type"] == "unattended_object"
        assert "sensor_metadata" not in event
        _drain_snapshots()
        client.stop()


def test_pest_event_id_stable_regardless_of_sensor_metadata():
    """Requirement 6: attaching sensor metadata does not change the deterministic
    event_id -- the same physical occurrence keeps one stable id."""
    with tempfile.TemporaryDirectory() as tmp:
        client, cfg = _client_and_cfg(tmp)
        pest = _make_pest()
        _fire_pest_alert(DummyFrame(), pest, cfg, now=1.0, renderer=MagicMock(),
                         client=client, sensor_metadata=dict(PIR_META))
        first_id = pest.flowguard_event_id
        # A re-alert of the SAME pest (with or without metadata) reuses the id.
        _fire_pest_alert(DummyFrame(), pest, cfg, now=2.0, renderer=MagicMock(),
                         client=client)
        assert pest.flowguard_event_id == first_id
        _drain_snapshots()
        client.stop()


def test_attached_sensor_metadata_is_isolated_copy():
    """Later mutation of the bridge state must not alter metadata already attached
    to a created event -- the call site copies with dict(sensor_meta)."""
    with tempfile.TemporaryDirectory() as tmp:
        client, cfg = _client_and_cfg(tmp)
        outbox = Path(tmp) / "outbox"
        live_meta = dict(PIR_META)
        snapshot_copy = dict(live_meta)          # what the loop passes: dict(sensor_meta)
        _fire_pest_alert(DummyFrame(), _make_pest(), cfg, now=1.0,
                         renderer=MagicMock(), client=client,
                         sensor_metadata=snapshot_copy)
        live_meta["trigger"] = "MUTATED"         # bridge state changes afterwards
        event = json.loads(list(outbox.glob("*.json"))[0].read_text())
        assert event["sensor_metadata"]["trigger"] == "PIR"
        _drain_snapshots()
        client.stop()


# --------------------------------------------------------------------------
# 20-24. Telemetry Endpoints on StreamServer (/health, /sensor_status, /video_feed, /people-count, /snapshot)
# --------------------------------------------------------------------------

def test_20_23_telemetry_endpoints():
    buffer = FrameBuffer()
    buffer.publish(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xd9") # minimal JPEG

    bridge = SensorBridge(hours_start="00:00", hours_end="23:59")
    now = datetime.now(SGT)
    line = json.dumps({
        "type": "sensor_status",
        "pir": False,
        "motion": False,
        "pir_ready": True,
        "distance_cm": 42.5,
        "baseline_distance_cm": 120.0,
        "distance_change_cm": 77.5,
        "object_close": True,
        "trigger": "ultrasonic"
    })
    bridge.process_line(line, now=now)
    bridge.trigger_inspection({"distance_cm": 42.5, "object_close": True}, trigger_type="ULTRASONIC", now=now)

    state_provider = lambda: {"count": 2, "detection_active": bridge.is_inspection_active()}
    cfg = Config(stream_host="127.0.0.1", stream_port=0)

    server = StreamServer(cfg, buffer, state_provider=state_provider, sensor_bridge=bridge)
    server.start()
    port = server.port

    import urllib.request
    try:
        # GET /sensor_status
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/sensor_status") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert data["connected"] is True
            assert data["pir_ready"] is True
            assert data["pir"] is False
            assert data["motion"] is False
            assert data["distance_cm"] == 42.5
            assert data["baseline_distance_cm"] == 120.0
            assert data["distance_change_cm"] == 77.5
            assert data["object_close"] is True
            assert data["trigger"] == "ULTRASONIC"
            assert data["inspection_active"] is True
            assert data["inspection_remaining_seconds"] > 0
            assert data["inspection_id"].startswith("insp_")

        # GET /health (includes sensor block and preserves all existing fields)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert data["status"] == "online"
            assert data["camera"] == "IMX500"
            assert data["streaming"] is True
            assert "latest_frame_age_seconds" in data
            assert "sensor" in data
            assert data["sensor"]["distance_cm"] == 42.5
            assert data["sensor"]["inspection_id"].startswith("insp_")

        # GET /people-count
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/people-count") as resp:
            data = json.loads(resp.read().decode("utf-8"))
            assert resp.status == 200
            assert data["count"] == 2
            assert data["detection_active"] is True

        # GET /snapshot
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/snapshot") as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "image/jpeg"
            body = resp.read()
            assert len(body) > 0

    finally:
        server.stop()


def test_inspection_id_lifecycle():
    bridge = SensorBridge(hours_start="00:00", hours_end="23:59")
    t0 = datetime.now(SGT)

    # 1. Trigger inspection cycle
    bridge.trigger_inspection({"motion": True}, trigger_type="PIR", now=t0)
    st0 = bridge.get_sensor_status(now=t0)
    insp_id1 = st0["inspection_id"]
    assert insp_id1 is not None and insp_id1.startswith("insp_")

    # 2. Continuous polling during active inspection retains exact same inspection_id
    t1 = t0 + timedelta(seconds=2)
    st1 = bridge.get_sensor_status(now=t1)
    assert st1["inspection_id"] == insp_id1

    # Extend inspection during window
    bridge.trigger_inspection({"motion": True}, trigger_type="PIR", now=t1)
    st1_ext = bridge.get_sensor_status(now=t1)
    assert st1_ext["inspection_id"] == insp_id1

    # 3. Window expires
    t2 = t0 + timedelta(seconds=20)
    st2 = bridge.get_sensor_status(now=t2)
    assert st2["inspection_active"] is False
    assert st2["inspection_id"] is None

    # 4. Next genuine inspection receives a new inspection_id
    t3 = t2 + timedelta(seconds=5)
    bridge.trigger_inspection({"motion": True}, trigger_type="PIR", now=t3)
    st3 = bridge.get_sensor_status(now=t3)
    insp_id2 = st3["inspection_id"]
    assert insp_id2 is not None and insp_id2.startswith("insp_")
    assert insp_id2 != insp_id1


def test_single_serial_reader_thread_enforcement():
    bridge = SensorBridge(hours_start="00:00", hours_end="23:59")
    from sensor_bridge import start_sensor_reader_thread

    with patch("os.path.exists", return_value=False):
        t1 = start_sensor_reader_thread(bridge, "COM_TEST")
        t2 = start_sensor_reader_thread(bridge, "COM_TEST")
        assert t1 is t2


def test_default_restricted_hours_08_to_23_defaults():
    with patch.dict(os.environ, {}, clear=True):
        cfg = Config()
        assert cfg.restricted_hours_start == "08:00"
        assert cfg.restricted_hours_end == "23:00"
        bridge = SensorBridge()
        assert bridge.hours_start == "08:00"
        assert bridge.hours_end == "23:00"


def test_securepi_runtime_honors_configured_restricted_hours():
    with patch.dict(os.environ, {"RESTRICTED_HOURS_START": "13:30", "RESTRICTED_HOURS_END": "17:30"}):
        cfg = Config()
        assert cfg.restricted_hours_start == "13:30"
        assert cfg.restricted_hours_end == "17:30"
        import securePi
        with patch("securePi.Picamera2"), patch("securePi.IMX500Detector"), patch("securePi.signal.signal"), patch("securePi.start_sensor_reader_thread"):
            with patch("securePi.SensorBridge", wraps=SensorBridge) as mock_sb:
                try:
                    securePi.run(cfg)
                except Exception:
                    pass
                assert mock_sb.called
                kwargs = mock_sb.call_args.kwargs
                assert kwargs["hours_start"] == "13:30"
                assert kwargs["hours_end"] == "17:30"


def test_after_hours_true_at_1400_for_1330_to_1730_window():
    bridge = SensorBridge(hours_start="13:30", hours_end="17:30")
    t_1400 = datetime(2026, 8, 8, 14, 0, 0, tzinfo=SGT)
    st = bridge.get_sensor_status(now=t_1400)
    assert st["after_hours"] is True


def test_after_hours_false_at_1800_for_1330_to_1730_window():
    bridge = SensorBridge(hours_start="13:30", hours_end="17:30")
    t_1800 = datetime(2026, 8, 8, 18, 0, 0, tzinfo=SGT)
    st = bridge.get_sensor_status(now=t_1800)
    assert st["after_hours"] is False


def test_overnight_restricted_hours_22_to_06():
    bridge = SensorBridge(hours_start="22:00", hours_end="06:00")
    t_2300 = datetime(2026, 8, 8, 23, 0, 0, tzinfo=SGT)
    t_0300 = datetime(2026, 8, 8, 3, 0, 0, tzinfo=SGT)
    t_1400 = datetime(2026, 8, 8, 14, 0, 0, tzinfo=SGT)
    assert bridge.get_sensor_status(now=t_2300)["after_hours"] is True
    assert bridge.get_sensor_status(now=t_0300)["after_hours"] is True
    assert bridge.get_sensor_status(now=t_1400)["after_hours"] is False


