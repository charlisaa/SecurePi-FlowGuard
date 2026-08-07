"""PIR + ultrasonic serial -> FlowGuard restricted-motion bridge.

Reads the Arduino's JSON status lines from a serial port and, during configured
restricted hours, raises a RESTRICTED_MOTION event on a motion TRANSITION (the
rising edge — not every line), through the shared FlowGuard edge client. Like
securePi.py, the Pi NEVER calls WhatsApp directly.

This process owns ONLY the serial port — it does not open or control the IMX500
camera, so it is safe to run alongside securePi.py (which owns the camera).

Design notes / deliberate limitations:
  * Singapore time is fixed UTC+8 (no DST), so no IANA tz database is required.
  * Restricted hours support an overnight range such as 22:00-06:00.
  * The ultrasonic distance is reported only as ``sensor_metadata`` — it can NOT
    identify an object class, so item pick-up/set-down alerts are intentionally
    NOT derived from distance (that would be an unverified claim).
  * Malformed / partial serial lines are ignored, never crash the loop.
  * ``--dry-run`` prints the event it would send, and sends nothing.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

# edge/ is on sys.path when run as a script or under the test harness.
from flowguard_api import FlowGuardApiClient, build_event_id

LOGGER = logging.getLogger("securepi.sensor_bridge")

# Singapore has no daylight saving — a fixed +8 offset is correct and avoids a
# dependency on the system IANA tz database (which Windows lacks by default).
SGT = timezone(timedelta(hours=8))

DEFAULT_SERIAL_CANDIDATES = ("/dev/ttyACM0", "/dev/ttyUSB0")


def parse_sensor_line(line: str) -> Optional[dict]:
    """Parse one serial line into a sensor_status dict, or None if it isn't one.

    Defensive: a truncated/garbled line, a boot banner, or anything that isn't a
    JSON object with type == 'sensor_status' returns None instead of raising."""
    if not line:
        return None
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("type") != "sensor_status":
        return None
    return data


def _parse_hm(value: str) -> int:
    """'HH:MM' -> minutes since midnight."""
    hours, minutes = str(value).split(":")
    return int(hours) * 60 + int(minutes)


def in_restricted_hours(now: datetime, start: str, end: str) -> bool:
    """True if ``now`` (Singapore time) falls in the [start, end) window. Handles
    an overnight range (start > end, e.g. 22:00-06:00). start == end means the
    window is always active (24h)."""
    current = now.hour * 60 + now.minute
    start_min = _parse_hm(start)
    end_min = _parse_hm(end)
    if start_min == end_min:
        return True
    if start_min < end_min:
        return start_min <= current < end_min
    # Overnight window: active from start through midnight to end.
    return current >= start_min or current < end_min


class SensorBridge:
    """Turns Arduino sensor lines into restricted-motion events (rising-edge only,
    warm-up-safe, restricted-hours-gated, with a cooldown) and maintains the active
    sensor inspection state for securePi camera identity matching."""

    def __init__(
        self,
        *,
        client: Optional[FlowGuardApiClient] = None,
        zone_name: str = "Restricted Zone",
        camera_location: str = "Camera 01",
        hours_start: str = "22:00",
        hours_end: str = "06:00",
        cooldown_sec: float = 60.0,
        inspection_window_sec: float = 10.0,
        device_id: Optional[str] = None,
        dry_run: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        self.client = client
        self.zone_name = zone_name
        self.camera_location = camera_location
        self.hours_start = hours_start
        self.hours_end = hours_end
        self.cooldown_sec = float(cooldown_sec)
        self.inspection_window_sec = float(inspection_window_sec)
        self.device_id = device_id or (getattr(client, "device_id", None) if client else None) or "securepi-sensor"
        self.dry_run = bool(dry_run)
        self.log = logger or LOGGER

        self._prev_motion = False
        self._prev_object_close = False
        self._last_alert_dt: Optional[datetime] = None
        self._inspection_expiry_dt: Optional[datetime] = None
        self.latest_sensor_metadata: dict = {}
        self._lock = threading.Lock()

    def is_inspection_active(self, now: Optional[datetime] = None) -> bool:
        """Returns True if a sensor-triggered inspection window is currently active."""
        if now is None:
            now = datetime.now(SGT)
        with self._lock:
            if self._inspection_expiry_dt is not None:
                if now < self._inspection_expiry_dt:
                    return True
                else:
                    self._inspection_expiry_dt = None
            return False

    def trigger_inspection(self, metadata: dict, now: Optional[datetime] = None) -> None:
        """Explicitly open or extend an inspection window with the given sensor metadata."""
        if now is None:
            now = datetime.now(SGT)
        with self._lock:
            self._inspection_expiry_dt = now + timedelta(seconds=self.inspection_window_sec)
            meta = dict(metadata)
            meta["inspection_active"] = True
            self.latest_sensor_metadata = meta

    def process_line(self, line: str, now: Optional[datetime] = None) -> Optional[dict]:
        """Process one serial line. Returns the event dict when an alert fires
        (also enqueued / printed as a side effect), else None. ``now`` is injectable
        for tests; production uses Singapore wall-clock."""
        if now is None:
            now = datetime.now(SGT)

        data = parse_sensor_line(line)
        if data is None:
            return None

        motion = bool(data.get("motion", data.get("pir")))
        pir_ready = bool(data.get("pir_ready", True))
        distance_change = float(data.get("distance_change_cm") or 0.0)
        object_close = bool(data.get("object_close"))

        # Rising edge or ultrasonic distance change
        pir_rising = motion and not self._prev_motion
        ultrasonic_rising = (distance_change > 15.0 or (object_close and not self._prev_object_close))
        rising = pir_rising or ultrasonic_rising

        self._prev_motion = motion
        self._prev_object_close = object_close

        if not rising:
            return None

        # No alerts during PIR warm-up.
        if not pir_ready:
            return None

        # Only inside the configured restricted hours (Singapore time).
        restricted = in_restricted_hours(now, self.hours_start, self.hours_end)
        after_hours = data.get("after_hours")
        if after_hours is None:
            after_hours = restricted
        if not restricted:
            return None

        trigger_type = "PIR" if pir_rising else "ULTRASONIC"
        if data.get("trigger"):
            trigger_type = str(data.get("trigger"))

        sensor_metadata = {
            "pir": motion,
            "motion": motion,
            "pir_ready": pir_ready,
            "distance_cm": data.get("distance_cm"),
            "baseline_distance_cm": data.get("baseline_distance_cm"),
            "distance_change_cm": data.get("distance_change_cm"),
            "object_close": object_close,
            "trigger": trigger_type,
            "inspection_active": True,
            "after_hours": bool(after_hours),
            "uptime_ms": data.get("uptime_ms"),
        }

        # Activate bounded inspection window
        self.trigger_inspection(sensor_metadata, now=now)

        # Cooldown between motion alerts.
        if self._last_alert_dt is not None and (now - self._last_alert_dt).total_seconds() < self.cooldown_sec:
            return None

        self._last_alert_dt = now
        event = self._build_event(data, now, sensor_metadata=sensor_metadata)
        if self.dry_run:
            print(json.dumps(event))
            self.log.info("[SensorBridge] DRY-RUN restricted-motion event (not sent).")
        elif self.client is not None:
            self.client.enqueue_event(event)
            self.log.warning("[SensorBridge] RESTRICTED-ZONE MOTION queued for %s.", self.zone_name)
        return event

    def _build_event(self, data: dict, now: datetime, sensor_metadata: Optional[dict] = None) -> dict:
        if sensor_metadata is None:
            sensor_metadata = {
                "distance_cm": data.get("distance_cm"),
                "object_close": data.get("object_close"),
                "pir_ready": data.get("pir_ready"),
                "uptime_ms": data.get("uptime_ms"),
            }
        event_id = build_event_id(self.device_id, "restricted_motion", "pir", now)
        if self.client is not None:
            return self.client.build_event(
                event_type="restricted_motion",
                alert_type="Restricted-Zone Motion",
                zone_name=self.zone_name,
                camera_location=self.camera_location,
                severity="High",
                device_id=self.device_id,
                sensor_metadata=sensor_metadata,
                timestamp=now,
                event_id=event_id,
            )
        # No client (pure dry-run without config): assemble a minimal event.
        return {
            "event_id": event_id,
            "zone_name": self.zone_name,
            "camera_location": self.camera_location,
            "alert_type": "Restricted-Zone Motion",
            "severity": "High",
            "device_id": self.device_id,
            "sensor_metadata": sensor_metadata,
            "timestamp": now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def run(self, serial_port: str, baud: int) -> None:
        """Open the serial port and process lines until interrupted. pyserial is
        imported lazily so the module stays importable (and testable) without it."""
        try:
            import serial  # type: ignore
        except ImportError:
            self.log.error("pyserial is not installed. Install it: pip install pyserial")
            raise

        self.log.info("[SensorBridge] Opening serial %s @ %d baud (zone=%s, hours=%s-%s, dry_run=%s).",
                      serial_port, baud, self.zone_name, self.hours_start, self.hours_end, self.dry_run)
        connection = serial.Serial(serial_port, baud, timeout=1)
        try:
            while True:
                raw = connection.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    self.process_line(line)
        except KeyboardInterrupt:
            self.log.info("[SensorBridge] Interrupted by user.")
        finally:
            try:
                connection.close()
            except Exception:  # pragma: no cover - defensive
                pass
            if self.client is not None:
                self.client.stop(wait=True)
            self.log.info("[SensorBridge] Stopped.")


def start_sensor_reader_thread(
    bridge: SensorBridge,
    serial_port: str,
    baud: int = 9600,
    stop_event: Optional[threading.Event] = None,
) -> Optional[threading.Thread]:
    """Launch a background daemon thread reading Arduino serial lines into ``bridge``.
    Defensive against serial reconnects and port absence."""
    def _reader_loop():
        try:
            import serial
        except ImportError:
            LOGGER.warning("[SensorBridgeThread] pyserial not available — serial bridge disabled.")
            return

        LOGGER.info("[SensorBridgeThread] Background serial reader starting on %s @ %d baud.", serial_port, baud)
        while stop_event is None or not stop_event.is_set():
            try:
                if not os.path.exists(serial_port) and not serial_port.startswith("COM"):
                    time.sleep(2.0)
                    continue
                with serial.Serial(serial_port, baud, timeout=1) as ser:
                    while stop_event is None or not stop_event.is_set():
                        raw = ser.readline()
                        if not raw:
                            continue
                        line = raw.decode("utf-8", errors="replace").strip()
                        if line:
                            bridge.process_line(line)
            except Exception as exc:
                LOGGER.debug("[SensorBridgeThread] Serial read retry after error: %s", exc)
                time.sleep(2.0)

    thread = threading.Thread(target=_reader_loop, name="securepi-sensor-reader", daemon=True)
    thread.start()
    return thread


def _default_serial_port(env=None) -> Optional[str]:
    """Configured port, else the first likely Pi device that actually exists."""
    env = env if env is not None else os.environ
    configured = env.get("SENSOR_SERIAL_PORT")
    if configured:
        return configured
    for candidate in DEFAULT_SERIAL_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    return None


def _env_flag(name: str, env=None) -> bool:
    env = env if env is not None else os.environ
    return str(env.get(name, "")).strip().lower() in ("1", "true", "yes", "on")


def parse_args(argv=None) -> argparse.Namespace:
    env = os.environ
    p = argparse.ArgumentParser(
        description="SecurePi PIR/ultrasonic -> FlowGuard restricted-motion bridge. "
                    "Owns only the serial port (never the camera); never calls WhatsApp.",
    )
    p.add_argument("--serial-port", default=_default_serial_port(),
                   help="Serial device (default: SENSOR_SERIAL_PORT, else the first of "
                        f"{', '.join(DEFAULT_SERIAL_CANDIDATES)} that exists).")
    p.add_argument("--baud", type=int, default=int(env.get("SENSOR_BAUD_RATE", "9600") or 9600),
                   help="Serial baud rate (default from SENSOR_BAUD_RATE, else 9600).")
    p.add_argument("--zone", default=env.get("RESTRICTED_ZONE_NAME", "Chemical Storage"),
                   help="Restricted zone name (default from RESTRICTED_ZONE_NAME).")
    p.add_argument("--camera-location",
                   default=env.get("RESTRICTED_CAMERA_LOCATION", "Chemical Storage Camera 01"),
                   help="Camera label for the event (default from RESTRICTED_CAMERA_LOCATION).")
    p.add_argument("--hours-start", default=env.get("RESTRICTED_HOURS_START", "22:00"),
                   help="Restricted hours start HH:MM, Singapore time (default from RESTRICTED_HOURS_START).")
    p.add_argument("--hours-end", default=env.get("RESTRICTED_HOURS_END", "06:00"),
                   help="Restricted hours end HH:MM, Singapore time (default from RESTRICTED_HOURS_END).")
    p.add_argument("--cooldown", type=float,
                   default=float(env.get("MOTION_ALERT_COOLDOWN_SEC", "60") or 60),
                   help="Seconds between restricted-motion alerts (default from MOTION_ALERT_COOLDOWN_SEC).")
    p.add_argument("--enabled", action="store_true", default=_env_flag("SENSOR_BRIDGE_ENABLED"),
                   help="Run the bridge (default from SENSOR_BRIDGE_ENABLED). Ignored with --dry-run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the event that WOULD be sent, and send nothing.")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging.")
    return p.parse_args(sys.argv[1:] if argv is None else list(argv))


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.enabled and not args.dry_run:
        LOGGER.info("[SensorBridge] Disabled (set SENSOR_BRIDGE_ENABLED=true or pass --enabled / --dry-run).")
        return 0
    if not args.serial_port:
        LOGGER.error("[SensorBridge] No serial port configured or detected. Pass --serial-port.")
        return 2

    # In dry-run we never send, so the FlowGuard client may stay disabled.
    client = FlowGuardApiClient.from_env(enabled=False) if args.dry_run else FlowGuardApiClient.from_env()
    bridge = SensorBridge(
        client=client,
        zone_name=args.zone,
        camera_location=args.camera_location,
        hours_start=args.hours_start,
        hours_end=args.hours_end,
        cooldown_sec=args.cooldown,
        dry_run=args.dry_run,
    )
    bridge.run(args.serial_port, args.baud)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
