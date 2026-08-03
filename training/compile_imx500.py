"""Compile a trained ONNX model to the Sony IMX500 hardware format (.rpk).

Runs OFF the Raspberry Pi (computer / Colab). Fails loudly and safely:

* detects when the imx500-converter tool is missing and prints setup help;
* validates the converter's return code (never prints success on failure);
* verifies the output .rpk actually exists and is non-empty;
* never creates an empty placeholder .rpk;
* returns a non-zero exit code on any failure.

A *successful compilation* is NOT the same as *runtime compatibility*: the
custom YOLOv8 output tensor may still not be decodable by the IMX500 parser in
edge/securePi.py. See docs/MODELS.md and models/README.md.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


class CompilationError(RuntimeError):
    """Raised when compilation cannot be performed or did not produce a valid .rpk."""


# Candidate converter executable names, in preference order.
CONVERTER_NAMES = ["imx500-converter", "imxconv-pt", "sdspconv"]

SETUP_HELP = (
    "The Sony IMX500 converter was not found.\n"
    "  • It requires Python 3.8–3.11 (NOT 3.12+).\n"
    "  • Install with:  pip install imx500-converter[pt]\n"
    "  • Or run training + compilation on Google Colab (T4 GPU, Python 3.10).\n"
    "  • Searched names: " + ", ".join(CONVERTER_NAMES) + " (PATH and the active venv)."
)


def get_latest_onnx(base_dir: str = "runs/detect") -> str:
    """Find the most recent exported best.onnx under runs/detect/."""
    detect_path = Path(base_dir)
    if detect_path.exists():
        matching_dirs = [
            d for d in detect_path.iterdir()
            if d.is_dir() and (d / "weights" / "best.onnx").exists()
        ]
        if matching_dirs:
            latest_dir = sorted(
                matching_dirs,
                key=lambda d: (d / "weights" / "best.onnx").stat().st_mtime,
                reverse=True,
            )[0]
            best_onnx = latest_dir / "weights" / "best.onnx"
            print(f"💡 Found latest ONNX model at: {best_onnx}")
            return str(best_onnx)
    return "runs/detect/securepi_model/weights/best.onnx"


def find_converter_bin():
    """Return the path to a usable converter executable, or None if not found.

    Checks the active interpreter's Scripts/bin folder, a local venv311, then
    the system PATH. Returns None (rather than a bare name) when nothing is
    found so callers can fail with a clear setup message instead of hitting an
    opaque FileNotFoundError from subprocess.
    """
    exe_names = [f"{n}.exe" for n in CONVERTER_NAMES] + CONVERTER_NAMES

    env_dir = Path(sys.executable).parent
    for bin_name in exe_names:
        target = env_dir / bin_name
        if target.exists():
            return str(target)

    workspace_venv = Path("./venv311/Scripts")
    for bin_name in exe_names:
        target = workspace_venv / bin_name
        if target.exists():
            return str(target)

    for bin_name in CONVERTER_NAMES:
        found = shutil.which(bin_name)
        if found:
            return found

    return None


def compile_rpk(
    onnx_path: str = None,
    output_rpk: str = "imx500_custom_securepi.rpk",
    quantization_ds: str = "./dataset/images/val",
    imgsz: int = 320,
    converter_bin: str = None,
    runner=None,
) -> Path:
    """Compile an ONNX model to .rpk. Returns the output Path on success.

    Raises FileNotFoundError if the ONNX input is missing, or CompilationError
    for any other failure (converter missing, non-zero exit, missing/empty
    output). Never returns on failure, and never writes a placeholder .rpk.

    ``converter_bin`` and ``runner`` are injection points for testing.
    """
    runner = runner or subprocess.run

    if onnx_path is None or onnx_path == "":
        onnx_path = get_latest_onnx()

    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        raise FileNotFoundError(
            f"ONNX model file not found at '{onnx_path}'. Export it first with export_onnx.py."
        )

    converter = converter_bin or find_converter_bin()
    if not converter:
        raise CompilationError(SETUP_HELP)

    output_path = Path(output_rpk)
    cmd = [
        converter, "convert",
        "--input", str(onnx_file),
        "--output", str(output_path),
        "--input-shape", "1", "3", str(imgsz), str(imgsz),
        "--quantization-dataset", quantization_ds,
        "--target-device", "imx500",
    ]

    print("⚙️  Compiling ONNX model to Sony IMX500 .rpk format...")
    print(f"    Running: {' '.join(cmd)}")

    try:
        res = runner(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        # The resolved converter path could not actually be executed.
        raise CompilationError(f"Converter '{converter}' could not be run: {exc}\n{SETUP_HELP}") from exc

    # Explicitly validate the return code — do NOT print success on failure.
    if res.returncode != 0:
        stderr = (res.stderr or "").strip()
        raise CompilationError(
            f"Converter exited with code {res.returncode}."
            + (f"\n--- converter stderr ---\n{stderr}" if stderr else "")
        )

    # A zero exit code is not proof of output: verify the artefact exists...
    if not output_path.exists():
        raise CompilationError(
            f"Converter reported success but no .rpk was produced at '{output_path}'."
        )
    # ...and that it is non-empty.
    size = output_path.stat().st_size
    if size == 0:
        raise CompilationError(f"Compiled .rpk is empty (0 bytes): '{output_path}'.")

    print(f"✅ Compilation succeeded: {output_path} ({size} bytes)")
    return output_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compile ONNX model to Raspberry Pi AI Camera (.rpk) format"
    )
    parser.add_argument("--input-onnx", type=str, default="",
                        help="Path to .onnx file (leave empty to auto-detect latest)")
    parser.add_argument("--output-rpk", type=str, default="imx500_custom_securepi.rpk",
                        help="Output .rpk filename")
    parser.add_argument("--val-dataset", type=str, default="./dataset/images/val",
                        help="Quantization dataset path")
    parser.add_argument("--imgsz", type=int, default=320, help="Input resolution")
    args = parser.parse_args(argv)

    try:
        compile_rpk(
            onnx_path=args.input_onnx,
            output_rpk=args.output_rpk,
            quantization_ds=args.val_dataset,
            imgsz=args.imgsz,
        )
    except (FileNotFoundError, CompilationError) as exc:
        print(f"❌ Compilation failed: {exc}", file=sys.stderr)
        return 1

    # Distinguish compilation success from runtime compatibility.
    print("⚠️  NOTE: a successful build does NOT guarantee the model runs on the "
          "IMX500.\n    The custom YOLOv8 output tensor may not be decodable by the "
          "parser in\n    edge/securePi.py — verify on the physical Pi. See docs/MODELS.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
