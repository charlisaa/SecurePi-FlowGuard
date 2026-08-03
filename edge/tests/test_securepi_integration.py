"""Off-device tests for the SecurePi -> FlowGuard alert callbacks.

Verifies that firing an unattended/pest alert:
  * still writes the local snapshot + events.csv row (offline behaviour preserved),
  * enqueues the correct FlowGuard event payload when a client is attached,
  * keeps rat/mouse as pest_detection (never unattended_object),
  * reuses ONE stable event_id across a cooldown re-alert,
  * and does nothing network-y (no outbox) when integration is disabled.

``cv2`` and ``picamera2`` are stubbed so the module imports off-device; the real
FlowGuardApiClient writes to a temp outbox with auto_flush disabled (no network).
Run: python -m pytest edge/tests/test_securepi_integration.py -q
"""

import json
import sys
import tempfile
import types
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EDGE_DIR))

# Stub cv2 BEFORE importing securePi (mirrors test_securepi.py). imwrite writes a
# real file so snapshot globbing works; drawing calls are no-ops here.
_cv2 = types.SimpleNamespace(
    imwrite=lambda path, img: bool(Path(path).write_bytes(b"jpg")) or True,
    rectangle=lambda *a, **k: None,
    putText=lambda *a, **k: None,
    FONT_HERSHEY_SIMPLEX=0,
)
sys.modules.setdefault("cv2", _cv2)

import flowguard_api as fg       # noqa: E402
import securePi                  # noqa: E402
from securePi import (           # noqa: E402
    Config, Detection, BagTracker, TrackedPest, Renderer,
    _fire_alert, _fire_pest_alert, SNAPSHOT_EXECUTOR,
)


class FakeFrame:
    def copy(self):
        return self


def _drain():
    """Block until the async snapshot/CSV writes have flushed."""
    SNAPSHOT_EXECUTOR.submit(lambda: None).result()


def _client(tmp):
    return fg.FlowGuardApiClient(
        "https://flowguard.example", "tok", enabled=True,
        device_id="securepi-lobby-01", camera_location="Lobby Camera 01",
        outbox_dir=str(Path(tmp) / "outbox"), auto_flush=False,
    )


def _only_event(client):
    files = list(Path(client.outbox_dir).glob("*.json"))
    assert len(files) == 1, f"expected 1 queued event, found {len(files)}"
    return json.loads(files[0].read_text())


def _bag(cfg, label="backpack"):
    bt = BagTracker(cfg)
    bt.update([Detection(label, 0.91, (50, 60, 40, 40))], [], now=0.0)
    bag = bt.bags[0]
    bag.unattended_start = 0.0
    return bag


def test_offline_preserves_snapshot_and_csv_and_makes_no_outbox():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(runtime_dir=Path(tmp), zone="lobby")
        bag = _bag(cfg)
        securePi.LOGGER.disabled = True
        try:
            _fire_alert(FakeFrame(), bag, cfg, now=cfg.unattended_time_sec + 5, renderer=Renderer(cfg), client=None)
        finally:
            securePi.LOGGER.disabled = False
        _drain()
        assert len(list((Path(tmp) / "snapshots" / "lobby").glob("alert_*.jpg"))) == 1
        events_csv = Path(tmp) / "logs" / "events.csv"
        assert events_csv.exists() and "unattended_object" in events_csv.read_text()
        # No FlowGuard outbox is created when there is no client.
        assert not (Path(tmp) / "outbox").exists()


def test_unattended_callback_builds_correct_payload_with_specific_label():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(runtime_dir=Path(tmp), zone="lobby")
        client = _client(tmp)
        bag = _bag(cfg, label="suitcase")  # specific COCO subtype must survive
        securePi.LOGGER.disabled = True
        try:
            _fire_alert(FakeFrame(), bag, cfg, now=cfg.unattended_time_sec + 5, renderer=Renderer(cfg), client=client)
        finally:
            securePi.LOGGER.disabled = False
        _drain()
        event = _only_event(client)
        assert event["alert_type"] == "Unattended Object"
        assert event["object_class"] == "suitcase"    # NOT a generic "bag"
        assert event["zone_name"] == "Lobby"
        assert event["camera_location"] == "Lobby Camera 01"
        assert event["event_id"].startswith("securepi-lobby-01:unattended_object:")
        assert event["snapshot_path"].endswith(".jpg")
        client.stop()


def test_pest_callback_uses_exact_label_never_unattended():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(runtime_dir=Path(tmp), zone="kitchen")
        client = _client(tmp)
        pest = TrackedPest(pest_id=4, label="rat", centroid=(10, 10), box=(10, 10, 20, 20),
                           first_seen=0.0, last_seen=1.0, score=0.92)
        securePi.LOGGER.disabled = True
        try:
            _fire_pest_alert(FakeFrame(), pest, cfg, now=1.0, renderer=Renderer(cfg), client=client)
        finally:
            securePi.LOGGER.disabled = False
        _drain()
        event = _only_event(client)
        assert event["alert_type"] == "Pest Detection"
        assert event["alert_type"] != "Unattended Object"
        assert event["object_class"] == "rat"
        assert ":pest_detection:" in event["event_id"]
        client.stop()


def test_stable_event_id_across_cooldown_realert():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(runtime_dir=Path(tmp), zone="lobby", alert_cooldown_sec=0.0)
        client = _client(tmp)
        bag = _bag(cfg)
        securePi.LOGGER.disabled = True
        try:
            _fire_alert(FakeFrame(), bag, cfg, now=cfg.unattended_time_sec + 5, renderer=Renderer(cfg), client=client)
            first_id = bag.flowguard_event_id
            # Cooldown re-alert of the SAME occurrence: same id, same outbox file.
            _fire_alert(FakeFrame(), bag, cfg, now=cfg.unattended_time_sec + 40, renderer=Renderer(cfg), client=client)
        finally:
            securePi.LOGGER.disabled = False
        _drain()
        assert bag.flowguard_event_id == first_id
        assert _only_event(client)["event_id"] == first_id  # still exactly one queued event
        client.stop()


if __name__ == "__main__":
    failures = 0
    for name in sorted(n for n in dir() if n.startswith("test_")):
        try:
            globals()[name]()
            print(f"PASS {name}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
