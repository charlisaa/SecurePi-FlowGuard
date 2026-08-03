"""Off-device tests for the PIR/ultrasonic -> FlowGuard restricted-motion bridge.

No serial hardware and no network: lines are fed to ``process_line`` directly and
time is injected. Run with:
    python -m pytest edge/tests/test_sensor_bridge.py -q
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # edge/ on path

import sensor_bridge as sb  # noqa: E402

SGT = sb.SGT


def _line(motion, pir=True, distance="18.4"):
    return ('{"type":"sensor_status","pir_ready":%s,"motion":%s,"distance_cm":%s,'
            '"object_close":true,"uptime_ms":123}' % (str(pir).lower(), str(motion).lower(), distance))


def _bridge(**over):
    kw = dict(client=None, zone_name="Chemical Storage", camera_location="Chemical Storage Camera 01",
              hours_start="22:00", hours_end="06:00", cooldown_sec=60, device_id="securepi-chem-01", dry_run=True)
    kw.update(over)
    return sb.SensorBridge(**kw)


# ---- parsing --------------------------------------------------------------
def test_parse_valid_sensor_line():
    data = sb.parse_sensor_line(_line(True))
    assert data["motion"] is True and data["pir_ready"] is True and data["distance_cm"] == 18.4


def test_parse_ignores_banner_and_malformed_lines():
    assert sb.parse_sensor_line("System started") is None
    assert sb.parse_sensor_line("PIR warming up for 30 seconds...") is None
    assert sb.parse_sensor_line('{"type":"sensor_status", partial') is None
    assert sb.parse_sensor_line('{"type":"other","x":1}') is None
    assert sb.parse_sensor_line("") is None
    assert sb.parse_sensor_line(None) is None


def test_no_echo_distance_is_null():
    data = sb.parse_sensor_line(_line(True, distance="null"))
    assert data["distance_cm"] is None


# ---- restricted hours -----------------------------------------------------
def test_overnight_restricted_hours():
    f = lambda h, m=0: sb.in_restricted_hours(datetime(2026, 7, 29, h, m, tzinfo=SGT), "22:00", "06:00")
    assert f(22, 0) and f(23) and f(2) and f(5, 59)
    assert not f(6, 0) and not f(12) and not f(21, 59)


def test_daytime_restricted_hours():
    g = lambda h: sb.in_restricted_hours(datetime(2026, 7, 29, h, tzinfo=SGT), "09:00", "17:00")
    assert g(9) and g(16) and not g(8) and not g(17)


def test_equal_start_end_is_always_active():
    assert sb.in_restricted_hours(datetime(2026, 7, 29, 3, tzinfo=SGT), "00:00", "00:00")


# ---- transition / cooldown / warm-up / hours ------------------------------
def test_motion_transition_emits_one_event():
    b = _bridge()
    t = datetime(2026, 7, 29, 23, 0, 0, tzinfo=SGT)
    e = b.process_line(_line(True), now=t)
    assert e is not None
    assert e["alert_type"] == "Restricted-Zone Motion"
    assert e["event_id"].startswith("securepi-chem-01:restricted_motion:pir:")
    assert e["sensor_metadata"]["distance_cm"] == 18.4


def test_continuous_motion_does_not_repeat():
    b = _bridge()
    t = datetime(2026, 7, 29, 23, 0, 0, tzinfo=SGT)
    assert b.process_line(_line(True), now=t) is not None
    assert b.process_line(_line(True), now=t.replace(second=1)) is None  # still HIGH -> no new edge
    assert b.process_line(_line(True), now=t.replace(second=2)) is None


def test_cooldown_blocks_then_allows():
    b = _bridge(cooldown_sec=60)
    t = datetime(2026, 7, 29, 23, 0, 0, tzinfo=SGT)
    assert b.process_line(_line(True), now=t) is not None
    b.process_line(_line(False), now=t.replace(second=5))       # drop
    assert b.process_line(_line(True), now=t.replace(second=10)) is None  # rise within cooldown
    # After the cooldown elapses, a fresh rising edge alerts again.
    later = t + timedelta(seconds=61)
    b.process_line(_line(False), now=later.replace(second=0))
    assert b.process_line(_line(True), now=later) is not None


def test_no_alert_during_pir_warmup():
    b = _bridge()
    t = datetime(2026, 7, 29, 23, 0, 0, tzinfo=SGT)
    assert b.process_line(_line(True, pir=False), now=t) is None


def test_no_alert_outside_restricted_hours():
    b = _bridge()
    noon = datetime(2026, 7, 29, 12, 0, 0, tzinfo=SGT)
    assert b.process_line(_line(True), now=noon) is None


def test_malformed_line_does_not_crash_or_alert():
    b = _bridge()
    t = datetime(2026, 7, 29, 23, 0, 0, tzinfo=SGT)
    assert b.process_line("garbage not json", now=t) is None
    assert b.process_line("", now=t) is None
    # A valid transition still works afterwards (state not corrupted).
    assert b.process_line(_line(True), now=t) is not None


def test_no_echo_distance_carried_as_null_in_event():
    b = _bridge()
    t = datetime(2026, 7, 29, 23, 0, 0, tzinfo=SGT)
    e = b.process_line(_line(True, distance="null"), now=t)
    assert e is not None and e["sensor_metadata"]["distance_cm"] is None


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
