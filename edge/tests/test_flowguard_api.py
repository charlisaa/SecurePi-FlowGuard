"""Off-device tests for the FlowGuard edge client + crash-safe outbox.

No real network: a fake ``urlopen`` is injected. Run with:
    python -m pytest edge/tests/test_flowguard_api.py -q
or standalone:
    python edge/tests/test_flowguard_api.py
"""

import json
import os
import sys
import tempfile
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # edge/ on path

import flowguard_api as fg  # noqa: E402


class FakeResp:
    def __init__(self, code, body=b"{}"):
        self._code = code
        self._body = body

    @property
    def status(self):
        return self._code

    def read(self):
        return self._body

    def close(self):
        pass


def urlopen_status(code, counter=None):
    def _u(req, timeout=None):
        if counter is not None:
            counter["n"] += 1
        if code >= 400:
            raise urllib.error.HTTPError(req.full_url, code, "err", {}, None)
        return FakeResp(code)
    return _u


def urlopen_network_error(counter=None):
    def _u(req, timeout=None):
        if counter is not None:
            counter["n"] += 1
        raise urllib.error.URLError("no route to host")
    return _u


def _client(tmp, urlopen, **kw):
    return fg.FlowGuardApiClient(
        "https://flowguard.example", "secret-token",
        enabled=True, device_id="securepi-test", camera_location="Test Cam",
        outbox_dir=os.path.join(tmp, "outbox"), auto_flush=False, urlopen=urlopen, **kw,
    )


def _event(client, **over):
    base = dict(event_type="unattended_object", alert_type="Unattended Object",
                zone_name="Lobby", object_class="backpack", severity="High",
                confidence=0.88, duration_seconds=35, track_id=12,
                snapshot_path="runtime/snapshots/lobby/x.jpg",
                timestamp="2026-07-29T08:46:00Z")
    base.update(over)
    return client.build_event(**base)


# ---- classification -------------------------------------------------------
def test_classify_success_retryable_config_permanent():
    assert fg.classify_http_failure(200) == fg.SUCCESS
    assert fg.classify_http_failure(201) == fg.SUCCESS
    assert fg.classify_http_failure(500) == fg.RETRYABLE
    assert fg.classify_http_failure(503) == fg.RETRYABLE
    assert fg.classify_http_failure(429) == fg.RETRYABLE
    assert fg.classify_http_failure(401) == fg.CONFIG_ERROR
    assert fg.classify_http_failure(403) == fg.CONFIG_ERROR
    assert fg.classify_http_failure(400) == fg.PERMANENT
    assert fg.classify_http_failure(422) == fg.PERMANENT
    assert fg.classify_http_failure(None, Exception("net")) == fg.RETRYABLE


# ---- event id -------------------------------------------------------------
def test_event_id_is_deterministic():
    a = fg.build_event_id("securepi-lobby-01", "unattended_object", 12, "2026-07-29T08:46:00Z")
    b = fg.build_event_id("securepi-lobby-01", "unattended_object", 12, "2026-07-29T08:46:00Z")
    assert a == b == "securepi-lobby-01:unattended_object:12:20260729T084600Z"


def test_build_event_uses_utc_iso_and_carries_id():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(201))
        event = _event(client)
        assert event["timestamp"] == "2026-07-29T08:46:00Z"
        assert event["event_id"] == "securepi-test:unattended_object:12:20260729T084600Z"
        assert event["alert_type"] == "Unattended Object"


# ---- outbox lifecycle -----------------------------------------------------
def test_enqueue_writes_atomically_no_temp_left():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(201))
        path = client.enqueue_event(_event(client))
        files = list(Path(client.outbox_dir).glob("*.json"))
        assert path.exists() and len(files) == 1
        # no stray temp files
        assert not list(Path(client.outbox_dir).glob(".tmp-*"))
        assert json.loads(path.read_text())["event_id"] == _event(client)["event_id"]


def test_http_201_removes_from_outbox():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(201))
        path = client.enqueue_event(_event(client))
        stats = client.flush_outbox()
        assert stats["sent"] == 1 and not path.exists()


def test_http_200_duplicate_removes_from_outbox():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(200))  # backend duplicate-success
        path = client.enqueue_event(_event(client))
        assert client.flush_outbox()["sent"] == 1 and not path.exists()


def test_http_500_keeps_for_retry():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(500))
        path = client.enqueue_event(_event(client))
        assert client.flush_outbox()["retry"] == 1 and path.exists()


def test_network_error_keeps_for_retry():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_network_error())
        path = client.enqueue_event(_event(client))
        assert client.flush_outbox()["retry"] == 1 and path.exists()


def test_http_400_dead_letters():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(400))
        path = client.enqueue_event(_event(client))
        stats = client.flush_outbox()
        assert stats["dead"] == 1 and not path.exists()
        assert (Path(client.outbox_dir) / "dead" / path.name).exists()


def test_http_401_does_not_spam_retries():
    with tempfile.TemporaryDirectory() as tmp:
        counter = {"n": 0}
        client = _client(tmp, urlopen_status(401, counter))
        client.enqueue_event(_event(client))
        client.flush_outbox()
        client.flush_outbox()
        client.flush_outbox()
        assert counter["n"] == 1  # halted after the first auth failure


def test_stable_event_id_across_retries():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(500))
        path = client.enqueue_event(_event(client))
        first = json.loads(path.read_text())["event_id"]
        client.flush_outbox()  # 500 -> kept
        # Same file, same id — a retry never mints a new id.
        assert path.exists() and json.loads(path.read_text())["event_id"] == first


def test_corrupt_queue_file_does_not_crash():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(201))
        client._ensure_dirs()
        (Path(client.outbox_dir) / "broken.json").write_text("{ not valid json")
        stats = client.flush_outbox()  # must not raise
        assert stats["dead"] == 1
        assert not (Path(client.outbox_dir) / "broken.json").exists()


def test_disabled_client_makes_no_outbox_and_no_calls():
    with tempfile.TemporaryDirectory() as tmp:
        counter = {"n": 0}
        client = fg.FlowGuardApiClient(
            "https://x", "tok", enabled=False,
            outbox_dir=os.path.join(tmp, "outbox"), urlopen=urlopen_status(201, counter),
        )
        assert client.enqueue_event({"event_id": "z"}) is None
        assert not os.path.exists(os.path.join(tmp, "outbox"))
        client.flush_outbox()
        assert counter["n"] == 0


def test_token_is_masked_in_config_and_logs():
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, urlopen_status(201))
        cfg = client.masked_config()
        assert cfg["token"] != "secret-token"
        assert "secret-token" not in str(cfg)
        assert fg.mask_token("secret-token") == "sec...(12 chars)"
        assert fg.mask_url("https://flowguard.example/api/edge/detection-alerts") == "https://flowguard.example"


def test_stop_closes_worker_cleanly():
    with tempfile.TemporaryDirectory() as tmp:
        client = fg.FlowGuardApiClient(
            "https://x", "tok", enabled=True,
            outbox_dir=os.path.join(tmp, "outbox"), auto_flush=True, urlopen=urlopen_status(201),
        )
        client.enqueue_event(_event(client))  # submits a background flush
        client.stop(wait=True)                 # must drain and not raise
        # After stop, further enqueues are no-ops (stopped) and don't crash.
        assert client.enqueue_event(_event(client)) is None


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
