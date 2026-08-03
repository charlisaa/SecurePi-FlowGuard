"""Send a FAKE SecurePi detection event to FlowGuard — WITHOUT the camera.

For local end-to-end validation of the FlowGuard integration. It builds one
synthetic event (pest / unattended / restricted-motion) and either prints it
(--dry-run) or POSTs it via the same edge client SecurePi uses.

Examples:
    # Just show the payload — no network, no config needed:
    python edge/send_test_event.py --dry-run --type pest

    # Actually POST to a running FlowGuard backend (reads env):
    FLOWGUARD_API_URL=http://localhost:5001 \
    EDGE_INGEST_TOKEN=test-edge-token \
    SECUREPI_DEVICE_ID=securepi-test \
    SECUREPI_CAMERA_LOCATION="Kitchen Camera 01" \
    python edge/send_test_event.py --type pest

The camera is never opened; this only exercises the HTTP path.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # edge/ on path
from flowguard_api import FlowGuardApiClient  # noqa: E402

# Fixed timestamp so the deterministic event_id is stable across runs (retry-safe).
FIXED_TS = "2026-07-29T09:00:00Z"

PRESETS = {
    "pest": dict(event_type="pest_detection", alert_type="Pest Detection",
                 zone_name="Kitchen", object_class="rat", severity="High",
                 confidence=0.92, track_id=1,
                 snapshot_path="runtime/snapshots/kitchen/pest_rat_1.jpg"),
    "unattended": dict(event_type="unattended_object", alert_type="Unattended Object",
                       zone_name="Lobby", object_class="backpack", severity="High",
                       confidence=0.88, duration_seconds=35, track_id=12,
                       snapshot_path="runtime/snapshots/lobby/alert_bag12.jpg"),
    "motion": dict(event_type="restricted_motion", alert_type="Restricted-Zone Motion",
                   zone_name="Chemical Storage", severity="High", track_id=None,
                   sensor_metadata={"distance_cm": 18.4, "object_close": True, "pir_ready": True}),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Send a fake SecurePi event to FlowGuard (no camera).")
    parser.add_argument("--type", choices=sorted(PRESETS), default="pest")
    parser.add_argument("--dry-run", action="store_true", help="Print the payload; do not send.")
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))

    # In dry-run we don't need a configured backend; otherwise read env.
    client = (FlowGuardApiClient.from_env(enabled=False) if args.dry_run
              else FlowGuardApiClient.from_env())
    event = client.build_event(timestamp=FIXED_TS, **PRESETS[args.type])

    if args.dry_run:
        print(json.dumps(event, indent=2))
        return 0

    if not client.api_url or not client.token:
        print("ERROR: set FLOWGUARD_API_URL and EDGE_INGEST_TOKEN (or use --dry-run).", file=sys.stderr)
        return 2

    classification, status, body = client.send_event(event)
    print(f"POST {client.api_url}/api/edge/detection-alerts -> HTTP {status} [{classification}]")
    if body:
        try:
            print(json.dumps(json.loads(body), indent=2))
        except (ValueError, TypeError):
            print(body[:500])
    return 0 if classification == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
