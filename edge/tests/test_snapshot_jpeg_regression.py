"""Regression test for the ".jpg.tmp" / cv2.imwrite() extension bug.

Every other test file in this suite stubs cv2 out (so tests run without a real
OpenCV install), which is exactly why this specific bug went undetected: a stub
doesn't care what extension you hand it. This file deliberately does NOT stub
cv2 -- it runs save_snapshot_worker() in a fresh subprocess against whatever
real cv2 is installed, so it actually exercises OpenCV's real codec-selection
behaviour (which picks the encoder from the destination filename's extension).

Skips itself if cv2 isn't installed in this environment (e.g. some CI images).
Run: python -m pytest edge/tests/test_snapshot_jpeg_regression.py -q
"""

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

EDGE_DIR = Path(__file__).resolve().parents[1]

pytest.importorskip("cv2", reason="real OpenCV not installed in this environment")


def test_save_snapshot_worker_writes_valid_decodable_jpeg_with_real_cv2():
    """save_snapshot_worker() must encode explicitly (cv2.imencode) rather than
    calling cv2.imwrite() on a ".jpg.tmp" path -- imwrite() picks its codec from
    the destination filename's extension, ".tmp" isn't a recognised one, and it
    silently fails to write anything at all. This test proves the final ".jpg"
    file actually exists and is a real, decodable JPEG of the right shape."""
    with tempfile.TemporaryDirectory() as tmp:
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(EDGE_DIR)!r})
            import numpy as np
            import securePi
            from pathlib import Path

            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            frame[:, :, 1] = 200  # distinguishable from an all-zero/blank buffer

            out_dir = Path({tmp!r})
            path = out_dir / "alert_regress.jpg"
            securePi.save_snapshot_worker(frame, path, out_dir, keep=10)

            assert path.exists(), "final .jpg was never written"
            assert not path.with_name(path.name + ".tmp").exists(), "temp file leaked"
            assert path.stat().st_size > 0, "final .jpg is empty"

            import cv2
            decoded = cv2.imread(str(path))
            assert decoded is not None, "final file is not a valid, decodable JPEG"
            assert decoded.shape[0] == 48 and decoded.shape[1] == 64, (
                f"decoded shape mismatch: {{decoded.shape}}"
            )
            print("REGRESSION_TEST_OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"save_snapshot_worker failed under real cv2:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "REGRESSION_TEST_OK" in result.stdout


def test_cv2_imwrite_on_dot_tmp_extension_fails_documenting_the_root_cause():
    """Documents the exact root cause with real cv2: imwrite() on a filename
    ending in ".jpg.tmp" raises/fails because ".tmp" isn't a registered codec
    extension -- this is why writing straight to the temp path via imwrite()
    (the old implementation) silently produced no file at all.

    Runs in a subprocess (like the test above) so it exercises the REAL cv2 --
    every other test file in this suite stubs "cv2" into sys.modules, and since
    that's a process-wide cache, a plain top-of-function `import cv2` here in
    the same pytest process would silently get one of those stubs instead.
    """
    with tempfile.TemporaryDirectory() as tmp:
        script = textwrap.dedent(f"""
            import cv2
            import numpy as np
            from pathlib import Path

            bad_path = str(Path({tmp!r}) / "alert_regress.jpg.tmp")
            frame = np.zeros((10, 10, 3), dtype=np.uint8)
            try:
                ok = cv2.imwrite(bad_path, frame)
            except cv2.error:
                ok = False  # OpenCV raises on some builds instead of returning False
            assert ok is False or not Path(bad_path).exists(), (
                "expected cv2.imwrite to fail (or write nothing) for a "
                "'.jpg.tmp' destination -- if this fails, OpenCV's behaviour "
                "has changed and the .tmp-suffix bug this file guards against "
                "may no longer apply"
            )
            print("ROOT_CAUSE_DOCUMENTED_OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
        assert "ROOT_CAUSE_DOCUMENTED_OK" in result.stdout
