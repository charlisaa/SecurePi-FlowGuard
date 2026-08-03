import argparse
import sys

from dataset_prep import prepare_dataset
from train import train_model
from export_onnx import export_to_onnx
from compile_imx500 import compile_rpk, CompilationError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SecurePi End-to-End AI Model Pipeline")
    parser.add_argument("--skip-dataset", action="store_true", help="Skip dataset downloading & formatting")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=320, help="Image resolution")
    parser.add_argument("--output-rpk", type=str, default="imx500_custom_securepi.rpk",
                        help="Output .rpk filename")
    args = parser.parse_args(argv)

    print("🚀 Starting SecurePi AI Model Building Pipeline...")

    # Step 1: Prepare Dataset
    if not args.skip_dataset:
        print("\n=== STEP 1: PREPARING DATASET ===")
        prepare_dataset()

    # Step 2: Train Model
    if not args.skip_train:
        print("\n=== STEP 2: TRAINING MODEL ===")
        train_model(epochs=args.epochs, imgsz=args.imgsz)

    # Step 3: Export ONNX
    print("\n=== STEP 3: EXPORTING ONNX MODEL ===")
    onnx_file = export_to_onnx(imgsz=args.imgsz)

    # Step 4: Compile RPK — abort loudly (non-zero exit) on any failure, and
    # only report success once compile_rpk has verified a non-empty .rpk exists.
    print("\n=== STEP 4: COMPILING TO IMX500 (.rpk) ===")
    try:
        rpk = compile_rpk(onnx_path=onnx_file, output_rpk=args.output_rpk, imgsz=args.imgsz)
    except (FileNotFoundError, CompilationError) as exc:
        print(f"\n❌ Pipeline FAILED at compilation: {exc}", file=sys.stderr)
        return 1

    print(f"\n🎉 Pipeline complete! Generated model file: {rpk} ({rpk.stat().st_size} bytes)")
    print("⚠️  Compilation success does NOT guarantee IMX500 runtime compatibility — "
          "verify the custom model on the physical Pi (see docs/MODELS.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
