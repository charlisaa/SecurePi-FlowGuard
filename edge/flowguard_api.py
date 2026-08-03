"""FlowGuard cloud edge client for SecurePi.

SecurePi sends detection events to the FlowGuard backend edge-ingest endpoint.
The Raspberry Pi NEVER calls WhatsApp directly — it only POSTs authenticated
events to FlowGuard, which owns database persistence, WhatsApp credentials,
message formatting, notification status and retries.

Design goals (see README "Cloud integration (FlowGuard)"):
  * Never block the IMX500 camera loop — all network I/O runs on a background
    worker thread; the caller only enqueues (a fast local file write).
  * Crash-safe disk-backed outbox — one JSON file per event, written atomically
    (temp file + os.replace) so a power loss can never leave a half-written or
    corrupt event. Flushed on startup and periodically while running.
  * Deterministic, retry-stable ``event_id`` — computed ONCE at alert time and
    persisted in the outbox file, so a Wi-Fi retry re-sends the identical id and
    the backend de-duplicates it. We never generate a new id per retry.
  * Failure classification: HTTP 200/201 success; network errors and HTTP 5xx
    retryable; HTTP 401/403 configuration errors (stop hammering); HTTP 400/422
    permanent payload errors (dead-letter, never retried forever).
  * Standard-library only (``urllib``) — no extra dependency on the Pi.
  * Never logs the ingest token or any secret; endpoints are logged host-masked.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

LOGGER = logging.getLogger("securepi.flowguard")

# --- Failure classification results -----------------------------------------
SUCCESS = "success"          # HTTP 200/201 — remove from outbox
RETRYABLE = "retryable"      # network error / HTTP 5xx / 408 / 429 — keep for retry
CONFIG_ERROR = "config_error"  # HTTP 401/403 — halt sending; a token/URL fix is needed
PERMANENT = "permanent"      # HTTP 400/422 (and other 4xx) — dead-letter, do not retry

DEFAULT_ENDPOINT_PATH = "/api/edge/detection-alerts"


def classify_http_failure(status: Optional[int], exc: Optional[BaseException] = None) -> str:
    """Classify an HTTP outcome into one of the SUCCESS/RETRYABLE/CONFIG_ERROR/PERMANENT
    buckets. A transport exception (no status) is always retryable."""
    if exc is not None and status is None:
        return RETRYABLE
    if status is None:
        return RETRYABLE
    if status in (200, 201) or 200 <= status < 300:
        return SUCCESS
    if status in (401, 403):
        return CONFIG_ERROR
    if status in (408, 429) or 500 <= status < 600:
        return RETRYABLE
    # 400, 422 and any other 4xx: the payload itself is wrong — retrying won't help.
    return PERMANENT


def mask_token(token: Optional[str]) -> str:
    """Mask a bearer token for logs — never reveal it. Shows only a short prefix."""
    if not token:
        return "(none)"
    if len(token) <= 6:
        return "****"
    return f"{token[:3]}...({len(token)} chars)"


def mask_url(url: Optional[str]) -> str:
    """Host-only view of an endpoint for logs (drops path and query)."""
    if not url:
        return "(none)"
    match = re.match(r"^(https?://[^/]+)", str(url))
    return match.group(1) if match else "(masked)"


def _to_datetime(value) -> datetime:
    """Coerce None/epoch/datetime/ISO-string into a timezone-aware UTC datetime."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        # Already an ISO string — parse best-effort so we can also derive the id stamp.
        try:
            text = value.replace("Z", "+00:00")
            parsed = datetime.fromisoformat(text)
            return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return datetime.now(timezone.utc)
    return datetime.now(timezone.utc)


def iso_utc(value=None) -> str:
    """ISO-8601 UTC timestamp (e.g. 2026-07-29T08:46:00Z). The backend re-formats
    this into Asia/Singapore wall-clock for the WhatsApp message."""
    dt = _to_datetime(value)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def compact_utc(value=None) -> str:
    """Compact UTC stamp for the event_id (e.g. 20260729T084600Z)."""
    return _to_datetime(value).strftime("%Y%m%dT%H%M%SZ")


def build_event_id(device_id: str, event_type: str, track_id, first_alert_timestamp) -> str:
    """Deterministic id for one alert OCCURRENCE:

        <device-id>:<event-type>:<track-id>:<first-alert-timestamp>

    Computed once when the alert first fires and then persisted; retries reuse the
    identical id so the backend recognises the duplicate. Do NOT call this again on
    each retry with a fresh timestamp."""
    track = "na" if track_id is None else str(track_id)
    return f"{device_id or 'securepi'}:{event_type}:{track}:{compact_utc(first_alert_timestamp)}"


def _safe_filename(event_id: str) -> str:
    """Filesystem-safe outbox filename derived from the event_id (so the same event
    maps to the same file — a second write of the same occurrence overwrites, giving
    on-disk de-duplication too). ':' is illegal on Windows, so sanitise it."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", event_id or "event")
    return f"{safe[:200]}.json"


class FlowGuardApiClient:
    """Non-blocking, crash-safe client for POSTing SecurePi detection events to
    FlowGuard. Thread-safe; a single background worker performs all network I/O."""

    def __init__(
        self,
        api_url: Optional[str],
        token: Optional[str],
        *,
        enabled: bool = True,
        device_id: Optional[str] = None,
        camera_location: Optional[str] = None,
        timeout: float = 5.0,
        retry_interval: float = 30.0,
        outbox_dir: Optional[os.PathLike] = None,
        endpoint_path: str = DEFAULT_ENDPOINT_PATH,
        auto_flush: bool = True,
        logger: Optional[logging.Logger] = None,
        urlopen: Optional[Callable] = None,
    ):
        self.enabled = bool(enabled)
        self.api_url = (api_url or "").rstrip("/")
        self.token = token or ""
        self.device_id = device_id
        self.camera_location = camera_location
        self.timeout = float(timeout)
        self.retry_interval = float(retry_interval)
        self.endpoint_path = endpoint_path
        self.auto_flush = bool(auto_flush)
        self.log = logger or LOGGER
        self._urlopen = urlopen or urllib.request.urlopen

        self.outbox_dir = Path(outbox_dir) if outbox_dir else Path("runtime/alerts/outbox")
        self.dead_dir = self.outbox_dir / "dead"

        self._executor: Optional[ThreadPoolExecutor] = None
        self._flush_lock = threading.Lock()
        self._halted = False          # set on a CONFIG_ERROR so we stop hammering
        self._stopped = False

        if self.enabled:
            self._ensure_dirs()
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flowguard")

    # -- configuration helpers ------------------------------------------------
    @classmethod
    def from_env(cls, env=None, **overrides) -> "FlowGuardApiClient":
        """Construct from FLOWGUARD_*/EDGE_INGEST_TOKEN/SECUREPI_* environment
        variables (used by the sensor bridge and standalone tools)."""
        env = env if env is not None else os.environ

        def truthy(name, default="false"):
            return str(env.get(name, default)).strip().lower() in ("1", "true", "yes", "on")

        params = dict(
            api_url=env.get("FLOWGUARD_API_URL"),
            token=env.get("EDGE_INGEST_TOKEN"),
            enabled=truthy("FLOWGUARD_EDGE_ENABLED"),
            device_id=env.get("SECUREPI_DEVICE_ID"),
            camera_location=env.get("SECUREPI_CAMERA_LOCATION"),
            timeout=float(env.get("SECUREPI_HTTP_TIMEOUT_SEC", "5") or 5),
            retry_interval=float(env.get("SECUREPI_RETRY_INTERVAL_SEC", "30") or 30),
            outbox_dir=env.get("SECUREPI_OUTBOX_DIR") or None,
        )
        params.update(overrides)
        return cls(**params)

    def _ensure_dirs(self) -> None:
        self.outbox_dir.mkdir(parents=True, exist_ok=True)
        self.dead_dir.mkdir(parents=True, exist_ok=True)

    def masked_config(self) -> dict:
        """Config summary safe to log — token and full endpoint path are masked."""
        return {
            "enabled": self.enabled,
            "endpoint": mask_url(self.api_url),
            "token": mask_token(self.token),
            "device_id": self.device_id,
            "outbox_dir": str(self.outbox_dir),
        }

    # -- event construction ---------------------------------------------------
    def build_event(
        self,
        *,
        event_type: str,
        alert_type: str,
        zone_name: str,
        camera_location: Optional[str] = None,
        object_class: Optional[str] = None,
        severity: Optional[str] = None,
        confidence: Optional[float] = None,
        duration_seconds=None,
        snapshot_path: Optional[str] = None,
        snapshot_url: Optional[str] = None,
        device_id: Optional[str] = None,
        track_id=None,
        person_name: Optional[str] = None,
        timestamp=None,
        first_alert_timestamp=None,
        sensor_metadata: Optional[dict] = None,
        event_id: Optional[str] = None,
    ) -> dict:
        """Build the JSON payload for POST /api/edge/detection-alerts. ``event_type``
        is the machine key (unattended_object/pest_detection/restricted_motion) used
        to build the deterministic event_id; ``alert_type`` is the human label the
        backend maps to a heading. UTC ISO-8601 timestamps only."""
        dev = device_id or self.device_id
        cam = camera_location or self.camera_location
        first_ts = first_alert_timestamp if first_alert_timestamp is not None else timestamp
        eid = event_id or build_event_id(dev, event_type, track_id, first_ts)

        payload = {
            "event_id": eid,
            "zone_name": zone_name,
            "camera_location": cam,
            "alert_type": alert_type,
            "object_class": object_class,
            "severity": severity,
            "confidence": confidence,
            "duration_seconds": duration_seconds,
            "snapshot_path": snapshot_path,
            "snapshot_url": snapshot_url,
            "device_id": dev,
            "track_id": track_id,
            "person_name": person_name,
            "sensor_metadata": sensor_metadata,
            "timestamp": iso_utc(timestamp),
        }
        # Drop keys that are None so the payload stays compact (event_id/timestamp
        # are always present).
        return {k: v for k, v in payload.items() if v is not None}

    # -- outbox (crash-safe) --------------------------------------------------
    def _atomic_write(self, path: Path, data: dict) -> None:
        """Write JSON atomically: temp file in the same dir, then os.replace()."""
        self._ensure_dirs()
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)  # atomic on the same filesystem
        except BaseException:
            # Never leave a stray temp file behind on failure.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def enqueue_event(self, event: dict) -> Optional[Path]:
        """Persist an event to the outbox (fast, local) and — when auto_flush — hand
        a flush to the background worker. Returns the outbox path, or None when the
        integration is disabled (so a disabled Pi never creates outbox files)."""
        if not self.enabled or self._stopped:
            return None
        path = self.outbox_dir / _safe_filename(event.get("event_id", "event"))
        self._atomic_write(path, event)
        if self.auto_flush:
            self.flush_async()
        return path

    def flush_async(self) -> None:
        """Hand a flush to the background worker (non-blocking). Safe to call often —
        the flush itself takes a non-blocking lock and skips if one is already running."""
        if self.enabled and self._executor is not None:
            try:
                self._executor.submit(self.flush_outbox)
            except RuntimeError:
                # executor already shut down (during teardown) — files stay queued
                pass

    def _dead_letter(self, path: Path, reason: str) -> None:
        """Move a permanently-failed or corrupt event out of the retry loop."""
        try:
            self._ensure_dirs()
            os.replace(str(path), str(self.dead_dir / path.name))
            self.log.warning("[FlowGuard] Dead-lettered %s (%s)", path.name, reason)
        except OSError as exc:
            self.log.error("[FlowGuard] Could not dead-letter %s: %s", path.name, exc)

    def send_event(self, event: dict):
        """POST a single event synchronously. Returns (classification, status, body).
        Never raises — transport errors are classified as RETRYABLE."""
        url = f"{self.api_url}{self.endpoint_path}"
        body = json.dumps(event).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            response = self._urlopen(request, timeout=self.timeout)
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                status = response.getcode()
            try:
                raw = response.read()
            except Exception:
                raw = b""
            finally:
                if hasattr(response, "close"):
                    response.close()
            return classify_http_failure(status, None), status, raw
        except urllib.error.HTTPError as exc:
            return classify_http_failure(exc.code, exc), exc.code, None
        except urllib.error.URLError as exc:
            self.log.warning("[FlowGuard] Network error to %s: %s", mask_url(self.api_url), exc.reason)
            return RETRYABLE, None, None
        except Exception as exc:  # pragma: no cover - defensive
            self.log.warning("[FlowGuard] Unexpected send error to %s: %s", mask_url(self.api_url), exc)
            return RETRYABLE, None, None

    def flush_outbox(self) -> dict:
        """Attempt to send every queued event. Returns a small stats dict. SUCCESS
        removes the file; RETRYABLE leaves it; PERMANENT/corrupt is dead-lettered; a
        CONFIG_ERROR halts the whole flush so we don't spam a bad token/URL."""
        stats = {"sent": 0, "retry": 0, "dead": 0, "halted": False}
        if not self.enabled or self._halted:
            stats["halted"] = self._halted
            return stats
        # Non-blocking: if a flush is already running (e.g. queued twice), skip.
        if not self._flush_lock.acquire(blocking=False):
            return stats
        try:
            if not self.outbox_dir.exists():
                return stats
            for path in sorted(self.outbox_dir.glob("*.json")):
                if not path.is_file():
                    continue
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        event = json.load(handle)
                except (json.JSONDecodeError, ValueError, OSError) as exc:
                    # A truncated/corrupt file must not crash startup — retire it.
                    self._dead_letter(path, f"corrupt: {exc}")
                    stats["dead"] += 1
                    continue

                classification, status, _ = self.send_event(event)
                if classification == SUCCESS:
                    self._remove(path)
                    stats["sent"] += 1
                elif classification == CONFIG_ERROR:
                    # Token/URL is wrong for every event — stop now, keep the files.
                    self._halted = True
                    stats["halted"] = True
                    self.log.error(
                        "[FlowGuard] Auth/config error (HTTP %s) to %s - halting sends. "
                        "Check EDGE_INGEST_TOKEN / FLOWGUARD_API_URL. Events remain queued.",
                        status, mask_url(self.api_url),
                    )
                    break
                elif classification == PERMANENT:
                    self._dead_letter(path, f"permanent HTTP {status}")
                    stats["dead"] += 1
                else:  # RETRYABLE
                    stats["retry"] += 1
            return stats
        finally:
            self._flush_lock.release()

    def _remove(self, path: Path) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        """Flush anything left over from a previous run (crash/power-loss recovery)."""
        if not self.enabled:
            return
        self.log.info("[FlowGuard] Edge client enabled: %s", self.masked_config())
        if self._executor is not None and self.auto_flush:
            self._executor.submit(self.flush_outbox)

    def reset_halt(self) -> None:
        """Clear a CONFIG_ERROR halt (e.g. after the operator fixes the token)."""
        self._halted = False

    def stop(self, wait: bool = True) -> None:
        """Shut the background worker down cleanly (called on SIGTERM/KeyboardInterrupt)."""
        self._stopped = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=wait)
            except Exception:  # pragma: no cover - defensive
                pass
            self._executor = None
