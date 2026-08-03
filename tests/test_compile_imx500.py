"""Off-device tests for the IMX500 compilation failure handling.

No Raspberry Pi, camera, or actual imx500-converter required: the converter is
injected via `converter_bin`/`runner`, so no real subprocess runs. Usage:

    python tests/test_compile_imx500.py     # standalone
    python -m pytest tests/                 # or via pytest
"""
import sys
import tempfile
import types
from pathlib import Path

# training/ holds compile_imx500.py
TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))

import compile_imx500 as C  # noqa: E402
from compile_imx500 import compile_rpk, CompilationError  # noqa: E402


def _fake_runner(returncode, make_output, content=b"rpk-bytes"):
    """Return a subprocess.run stand-in that optionally writes the --output file."""
    def runner(cmd, stdout=None, stderr=None, text=None):
        if make_output:
            out = cmd[cmd.index("--output") + 1]
            Path(out).write_bytes(content)
        return types.SimpleNamespace(returncode=returncode, stdout="",
                                     stderr="converter error" if returncode else "")
    return runner


def _with_onnx(fn):
    """Run fn(onnx_path, out_path) inside a temp dir holding a fake .onnx input."""
    with tempfile.TemporaryDirectory() as td:
        onnx = Path(td) / "best.onnx"
        onnx.write_bytes(b"fake-onnx")
        out = Path(td) / "model.rpk"
        return fn(str(onnx), str(out))


def test_compilation_failure_is_not_reported_as_success():   # requirement 10
    """A non-zero converter exit raises — never returns a 'success'."""
    def case(onnx, out):
        try:
            compile_rpk(onnx_path=onnx, output_rpk=out, converter_bin="fake-conv",
                        runner=_fake_runner(returncode=2, make_output=False))
        except CompilationError as exc:
            assert "code 2" in str(exc)
            return "raised"
        return "returned"
    assert _with_onnx(case) == "raised"


def test_missing_output_rpk_is_failure():                    # requirement 11a
    """Exit code 0 but no .rpk produced -> failure."""
    def case(onnx, out):
        try:
            compile_rpk(onnx_path=onnx, output_rpk=out, converter_bin="fake-conv",
                        runner=_fake_runner(returncode=0, make_output=False))
        except CompilationError as exc:
            assert "no .rpk" in str(exc)
            return "raised"
        return "returned"
    assert _with_onnx(case) == "raised"


def test_empty_output_rpk_is_failure():                      # requirement 11b
    """Exit code 0 but a 0-byte .rpk -> failure (no empty placeholder accepted)."""
    def case(onnx, out):
        try:
            compile_rpk(onnx_path=onnx, output_rpk=out, converter_bin="fake-conv",
                        runner=_fake_runner(returncode=0, make_output=True, content=b""))
        except CompilationError as exc:
            assert "empty" in str(exc)
            return "raised"
        return "returned"
    assert _with_onnx(case) == "raised"


def test_successful_compilation_returns_nonempty_rpk():
    """The happy path returns a Path to a real, non-empty .rpk."""
    def case(onnx, out):
        result = compile_rpk(onnx_path=onnx, output_rpk=out, converter_bin="fake-conv",
                             runner=_fake_runner(returncode=0, make_output=True))
        assert result.exists() and result.stat().st_size > 0
        return "ok"
    assert _with_onnx(case) == "ok"


def test_missing_converter_is_failure():
    """When no converter can be found, compilation fails with setup help."""
    orig_find = C.find_converter_bin
    C.find_converter_bin = lambda: None
    try:
        def case(onnx, out):
            try:
                compile_rpk(onnx_path=onnx, output_rpk=out,
                            runner=_fake_runner(returncode=0, make_output=True))
            except CompilationError as exc:
                assert "converter" in str(exc).lower()
                return "raised"
            return "returned"
        assert _with_onnx(case) == "raised"
    finally:
        C.find_converter_bin = orig_find


def test_missing_onnx_input_is_failure():
    """A missing ONNX input raises before any converter is invoked."""
    with tempfile.TemporaryDirectory() as td:
        missing = str(Path(td) / "nope.onnx")
        try:
            compile_rpk(onnx_path=missing, output_rpk=str(Path(td) / "o.rpk"),
                        converter_bin="fake-conv",
                        runner=_fake_runner(returncode=0, make_output=True))
        except FileNotFoundError:
            return
        raise AssertionError("expected FileNotFoundError for a missing ONNX input")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"{test.__name__} OK")
    print(f"ALL {len(tests)} TESTS PASSED")
