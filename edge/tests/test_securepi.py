"""Off-device tests for securePi tracking, matching, alerting, pests, and CLI.

Runs on any machine — cv2 and picamera2 are not required (cv2 is stubbed
before import). Usage:

    python edge/tests/test_securepi.py      # standalone
    python -m pytest edge/tests/            # or via pytest
"""
import contextlib
import io
import sys
import tempfile
import types
from pathlib import Path

# edge/ (parent of this tests/ dir) holds securePi.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Stub cv2 so the module imports without OpenCV. imwrite writes a placeholder
# file so the snapshot-pruning test can observe real files on disk; the drawing
# stubs record their calls so annotation tests can assert what was drawn.
_draw_calls = []
_cv2 = types.SimpleNamespace(
    imwrite=lambda path, img: Path(path).write_bytes(b"jpg") > 0,
    rectangle=lambda frame, p1, p2, color, thickness: _draw_calls.append(
        ("rect", p1, p2, color, thickness)),
    putText=lambda frame, text, org, font, scale, color, thickness: _draw_calls.append(
        ("text", text, color)),
    FONT_HERSHEY_SIMPLEX=0,
)
sys.modules.setdefault("cv2", _cv2)

from securePi import (Config, Detection, PersonTracker, BagTracker,  # noqa: E402
                      PestTracker, match_detections, smooth_box, parse_args,
                      _alert_due, _pest_alert_due, split_detections,
                      save_snapshot_worker, append_event_worker, log_event,
                      Renderer, _fire_alert, _fire_pest_alert, SNAPSHOT_EXECUTOR,
                      COLOR_ALERT, COLOR_PEST, dedup_detections, EVENT_HEADER)


class FakeFrame:
    """Minimal frame stand-in: drawing stubs ignore it, save_snapshot copies it."""

    def copy(self):
        return self

CFG = Config()


def _drain():
    """Block until the single-worker snapshot/log executor has flushed."""
    SNAPSHOT_EXECUTOR.submit(lambda: None).result()


# --------------------------------------------------------------------------
# Existing coverage (updated for the new layout/schema — not weakened)
# --------------------------------------------------------------------------

def test_smooth_box():
    old, new = (100, 100, 50, 50), (110, 110, 50, 50)
    assert smooth_box(old, new, 0.6) == (106, 106, 50, 50)
    far = (400, 400, 50, 50)
    assert smooth_box(old, far, 0.6) == far          # jump: snap, don't drag
    assert smooth_box(old, new, 1.0) == new          # 1.0 disables smoothing


def test_match_detections_order_independent():
    class T:
        def __init__(self, box):
            self.box = box
            self.centroid = (box[0] + box[2] / 2, box[1] + box[3] / 2)

    tA, tB = T((100, 100, 40, 40)), T((160, 100, 40, 40))
    d1 = Detection("bag", 0.9, (158, 100, 40, 40))   # clearly B, but first in list
    d2 = Detection("bag", 0.9, (102, 100, 40, 40))   # clearly A
    m = match_detections([d1, d2], [tA, tB], iou_gate=0.3, dist_gate=120)
    assert m[0] is tB and m[1] is tA


def test_person_tracker_stable_ids():
    pt = PersonTracker(CFG)
    pt.update([Detection("person", 0.9, (100, 100, 60, 120))], now=0.0)
    pt.update([Detection("person", 0.9, (104, 98, 62, 118))], now=0.1)
    assert len(pt.tracks) == 1
    track = next(iter(pt.tracks.values()))
    assert track.person_id == 0
    assert 100 <= track.box[0] <= 104                # smoothed between detections


def test_bag_tracker_no_swap():
    bt = BagTracker(CFG)
    bt.update([Detection("suitcase", 0.9, (100, 100, 40, 40)),
               Detection("suitcase", 0.9, (200, 100, 40, 40))], [], now=0.0)
    bt.update([Detection("suitcase", 0.9, (201, 101, 40, 40)),   # reversed order
               Detection("suitcase", 0.9, (99, 99, 40, 40))], [], now=0.1)
    assert len(bt.bags) == 2
    assert abs(bt.bags[0].centroid[0] - 120) < 5     # bag 0 stayed left
    assert abs(bt.bags[1].centroid[0] - 220) < 5     # bag 1 stayed right


def test_alert_due():
    bt = BagTracker(CFG)
    bt.update([Detection("suitcase", 0.9, (0, 0, 10, 10))], [], now=0.0)
    bag = bt.bags[0]
    bag.unattended_start = 0.0
    assert not _alert_due(bag, CFG, now=10.0)
    assert _alert_due(bag, CFG, now=CFG.unattended_time_sec + 1)
    bag.alerted = True
    bag.last_alert_time = CFG.unattended_time_sec + 1
    assert not _alert_due(bag, CFG, now=CFG.unattended_time_sec + 5)
    assert _alert_due(bag, CFG,
                      now=CFG.unattended_time_sec + 1 + CFG.alert_cooldown_sec)


def test_snapshot_pruning():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for i in range(5):
            (directory / f"alert_bag0_2026010{i}-000000.jpg").write_bytes(b"x")
        save_snapshot_worker(None, directory / "alert_bag1_new.jpg",
                             directory, keep=3)
        remaining = sorted(f.name for f in directory.glob("*.jpg"))
        assert len(remaining) == 3, remaining
        assert "alert_bag1_new.jpg" in remaining     # newest survives


def test_event_log_append_schema():
    """The event log carries the richer incident schema, header written once."""
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "events.csv"
        append_event_worker(log, ["2026-07-07 10:00:00", "unattended_object", "bag",
                                  "0.91", 0, "lobby", 12, "alert_bag0_x.jpg"])
        append_event_worker(log, ["2026-07-07 10:01:00", "pest", "rat",
                                  "0.95", 1, "kitchen", 3, "pest_rat_1_y.jpg"])
        lines = log.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0] == ",".join(EVENT_HEADER)        # header once
        assert len(lines) == 3
        assert lines[2].startswith("2026-07-07 10:01:00,pest,rat,")


def test_demo_preset():
    args = parse_args(["@demo.args"])
    assert args.unattended_time == 10.0              # demo override
    assert args.owner_claim_time == 5.0              # demo override
    assert args.alert_cooldown == 15.0               # demo override
    assert args.proximity == 150.0                   # inherited from common.args
    assert args.zone == "demo"                       # demo tags the zone


def test_presets_resolution_and_layering():
    args = parse_args(["@lobby.args"])               # short name, any CWD
    assert args.unattended_time == 60.0              # lobby override
    assert args.proximity == 120.0                   # lobby override
    assert args.alert_cooldown == 30.0               # inherited from common.args
    assert args.zone == "lobby"
    args = parse_args(["@kitchen.args"])
    assert args.unattended_time == 300.0
    assert args.proximity == 150.0                   # inherited
    args = parse_args(["@lobby.args", "--unattended-time", "15"])
    assert args.unattended_time == 15.0              # CLI beats preset


def test_preset_required():
    for argv in ([], ["--headless"]):
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                parse_args(argv)
            raise AssertionError(f"should have exited for argv={argv}")
        except SystemExit as e:
            assert e.code == 2
        assert "settings file is required" in err.getvalue()
        assert "@lobby.args" in err.getvalue()
    # -h still prints help without a preset
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            parse_args(["-h"])
    except SystemExit as e:
        assert e.code == 0
    assert "--unattended-time" in out.getvalue()


def test_bag_labels_alias():
    """--bag-labels stays a backward-compatible alias of --unattended-object-labels."""
    args = parse_args(["@lobby.args", "--bag-labels", "backpack", "laptop"])
    assert set(args.unattended_object_labels) == {"backpack", "laptop"}


def test_dedup_cross_label_detections():
    """One bag reported as backpack AND handbag in the same frame -> one detection."""
    dets = [Detection("backpack", 0.7, (100, 100, 40, 40)),
            Detection("handbag", 0.9, (102, 101, 40, 42)),
            Detection("suitcase", 0.8, (300, 100, 40, 40))]   # a genuinely separate bag
    kept = dedup_detections(dets)
    assert len(kept) == 2
    assert kept[0].label == "handbag"                          # highest score wins
    assert any(d.label == "suitcase" for d in kept)


def test_duplicate_bag_detections_one_track():
    bt = BagTracker(CFG)
    bt.update([Detection("backpack", 0.7, (100, 100, 40, 40)),
               Detection("handbag", 0.9, (102, 101, 40, 42))], [], now=0.0)
    assert len(bt.bags) == 1


def test_duplicate_bag_tracks_merge():
    """A stale flicker track stacked on the same bag collapses into the oldest,
    which keeps its id and unattended timer."""
    bt = BagTracker(CFG)
    bt.update([Detection("suitcase", 0.9, (100, 100, 40, 40))], [], now=0.0)
    bt._register(Detection("suitcase", 0.9, (104, 102, 40, 40)), now=5.0)
    assert len(bt.bags) == 2
    bt.update([], [], now=6.0)
    assert list(bt.bags) == [0]                                # oldest id survives
    assert bt.bags[0].unattended_start == 0.0                  # timer not reset
    assert bt.bags[0].last_seen == 5.0                         # fresher sighting adopted


def test_alert_snapshot_has_bag_box():
    """The saved alert snapshot must mark the offending bag, even if the
    renderer skipped its box (track coasting past draw_grace_sec)."""
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(runtime_dir=Path(td))
        bt = BagTracker(cfg)
        bt.update([Detection("suitcase", 0.9, (50, 60, 40, 40))], [], now=0.0)
        bag = bt.bags[0]
        bag.unattended_start = 0.0

        _draw_calls.clear()
        import securePi
        securePi.LOGGER.disabled = True
        try:
            _fire_alert(FakeFrame(), bag, cfg, now=cfg.unattended_time_sec + 5,
                        renderer=Renderer(cfg))
        finally:
            securePi.LOGGER.disabled = False
        _drain()   # snapshot + log writes have finished

        rects = [c for c in _draw_calls if c[0] == "rect"]
        assert any(c[3] == COLOR_ALERT and c[4] == 3 for c in rects), \
            "no thick red alert box was drawn on the frame"
        texts = [c for c in _draw_calls if c[0] == "text"]
        assert any("ALERT" in c[1] and f"Bag #{bag.bag_id}" in c[1] for c in texts)
        assert len(list((Path(td) / "snapshots").glob("alert_*.jpg"))) == 1


# --------------------------------------------------------------------------
# NEW: functional category routing (rat/mouse are pests, never bags)
# --------------------------------------------------------------------------

def test_rat_not_unattended_object():          # requirement 1
    cfg = Config()
    persons, objects, pests = split_detections([Detection("rat", 0.9, (10, 10, 20, 20))], cfg)
    assert objects == [] and len(pests) == 1
    bt = BagTracker(cfg)
    bt.update(objects, [], now=0.0)
    assert len(bt.bags) == 0                    # a rat never enters bag tracking


def test_mouse_not_unattended_object():        # requirement 2
    cfg = Config()
    persons, objects, pests = split_detections([Detection("mouse", 0.9, (10, 10, 20, 20))], cfg)
    assert objects == [] and len(pests) == 1
    bt = BagTracker(cfg)
    bt.update(objects, [], now=0.0)
    assert len(bt.bags) == 0                    # a mouse never enters bag tracking


def test_unattended_object_still_tracked():    # requirement 7
    cfg = Config()
    persons, objects, pests = split_detections([
        Detection("backpack", 0.9, (10, 10, 30, 30))], cfg)
    assert len(objects) == 1 and pests == []
    bt = BagTracker(cfg)
    bt.update(objects, [], now=0.0)             # no owner present
    bag = bt.bags[0]
    assert bag.unattended_start == 0.0          # unattended timer started
    assert _alert_due(bag, cfg, now=cfg.unattended_time_sec + 1)


def test_person_owner_association():           # requirement 8
    cfg = Config()
    pt = PersonTracker(cfg)
    pt.update([Detection("person", 0.9, (100, 100, 40, 80))], now=0.0)
    persons = list(pt.tracks.values())
    bt = BagTracker(cfg)
    bt.update([Detection("suitcase", 0.9, (120, 120, 30, 30))], persons, now=0.0)
    bag = bt.bags[0]
    assert bag.owner_id == persons[0].person_id  # nearby person adopted as owner
    assert bag.unattended_start is None          # owner present -> timer not running


# --------------------------------------------------------------------------
# NEW: pest detection logic
# --------------------------------------------------------------------------

def test_pest_requires_confirmation_time():    # requirement 3 (duration)
    cfg = Config(pest_confirmation_time=2.0, pest_confirmation_frames=0)
    pt = PestTracker(cfg)
    pt.update([Detection("rat", 0.9, (10, 10, 20, 20))], now=0.0)
    pest = next(iter(pt.pests.values()))
    assert not _pest_alert_due(pest, cfg, now=1.0)   # before the threshold: no alert
    assert _pest_alert_due(pest, cfg, now=2.5)       # after: due


def test_pest_requires_confirmation_frames():  # requirement 3 (frames)
    cfg = Config(pest_confirmation_time=0, pest_confirmation_frames=3)
    pt = PestTracker(cfg)
    pt.update([Detection("mouse", 0.9, (10, 10, 20, 20))], now=0.00)
    pt.update([Detection("mouse", 0.9, (10, 10, 20, 20))], now=0.01)
    pest = next(iter(pt.pests.values()))
    assert pest.frames == 2
    assert not _pest_alert_due(pest, cfg, now=0.01)  # 2 frames < 3: no alert
    pt.update([Detection("mouse", 0.9, (10, 10, 20, 20))], now=0.02)
    assert pest.frames == 3
    assert _pest_alert_due(pest, cfg, now=0.02)      # 3 frames: due


def test_pest_cooldown():                      # requirement 4
    cfg = Config(pest_confirmation_time=0, pest_confirmation_frames=1,
                 pest_alert_cooldown_sec=30)
    pt = PestTracker(cfg)
    pt.update([Detection("rat", 0.9, (10, 10, 20, 20))], now=0.0)
    pest = next(iter(pt.pests.values()))
    assert _pest_alert_due(pest, cfg, now=0.0)       # confirmed on frame 1
    pest.alerted = True
    pest.last_alert_time = 0.0
    assert not _pest_alert_due(pest, cfg, now=10.0)  # within cooldown: suppressed
    assert _pest_alert_due(pest, cfg, now=31.0)      # after cooldown: due again


def test_pest_low_confidence_filtered():
    """Pests below --pest-confidence never reach the pest tracker."""
    cfg = Config(pest_confidence=0.5)
    persons, objects, pests = split_detections([
        Detection("rat", 0.40, (10, 10, 20, 20)),    # below pest threshold
        Detection("mouse", 0.80, (60, 60, 20, 20))], cfg)
    assert [p.label for p in pests] == ["mouse"]


def test_pest_alert_class_and_log():           # requirements 5 & 6
    with tempfile.TemporaryDirectory() as td:
        cfg = Config(runtime_dir=Path(td), pest_confirmation_time=0,
                     pest_confirmation_frames=1)
        pt = PestTracker(cfg)
        pt.update([Detection("rat", 0.95, (30, 30, 20, 20))], now=0.0)
        pest = next(iter(pt.pests.values()))

        _draw_calls.clear()
        import securePi
        securePi.LOGGER.disabled = True
        try:
            _fire_pest_alert(FakeFrame(), pest, cfg, now=1.0, renderer=Renderer(cfg))
        finally:
            securePi.LOGGER.disabled = False
        _drain()

        # Requirement 5: the alert carries the detected class.
        texts = [c for c in _draw_calls if c[0] == "text"]
        assert any("rat" in c[1].lower() and "PEST" in c[1] for c in texts)
        rects = [c for c in _draw_calls if c[0] == "rect"]
        assert any(c[3] == COLOR_PEST for c in rects)
        # A snapshot named for the pest class was created.
        snaps = list((Path(td) / "snapshots").glob("pest_rat_*.jpg"))
        assert len(snaps) == 1
        # Requirement 6: the event log records class + timestamp on the pest path.
        rows = (Path(td) / "logs" / "events.csv").read_text(encoding="utf-8").splitlines()
        assert rows[0] == ",".join(EVENT_HEADER)
        assert any(r.split(",")[1] == "pest" and r.split(",")[2] == "rat"
                   for r in rows[1:])


def test_pest_continuous_visibility_single_alert():
    """A pest that stays continuously visible does not re-alert every frame."""
    cfg = Config(pest_confirmation_time=0, pest_confirmation_frames=1,
                 pest_alert_cooldown_sec=30)
    pt = PestTracker(cfg)
    alerts = 0
    for i in range(10):                              # 10 consecutive sightings
        now = i * 0.1
        pt.update([Detection("rat", 0.9, (10, 10, 20, 20))], now=now)
        pest = next(iter(pt.pests.values()))
        if _pest_alert_due(pest, cfg, now=now):
            alerts += 1
            pest.alerted = True
            pest.last_alert_time = now
    assert alerts == 1                               # only the first fires (cooldown holds)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"{test.__name__} OK")
    print(f"ALL {len(tests)} TESTS PASSED")
