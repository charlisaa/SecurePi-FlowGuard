import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def get_latest_weights(base_dir: str = "runs/detect") -> str:
    """
    Finds the latest trained best.pt weights in runs/detect/ (e.g., securepi_model, securepi__model-2, etc.).
    """
    detect_path = Path(base_dir)
    if detect_path.exists():
        # Check all subdirectories in runs/detect/ for weights/best.pt
        matching_dirs = [
            d for d in detect_path.iterdir()
            if d.is_dir() and (d / "weights" / "best.pt").exists()
        ]
        if matching_dirs:
            # Sort by last modified time descending
            latest_dir = sorted(matching_dirs, key=lambda d: (d / "weights" / "best.pt").stat().st_mtime, reverse=True)[0]
            best_pt = latest_dir / "weights" / "best.pt"
            print(f"💡 Found latest trained model at: {best_pt}")
            return str(best_pt)

    return "runs/detect/securepi_model/weights/best.pt"

def export_to_onnx(weights_path: str = None, imgsz: int = 320):
    """
    Exports trained PyTorch weights (.pt) to ONNX format.
    """
    if weights_path is None or weights_path == "":
        weights_path = get_latest_weights()

    weights_file = Path(weights_path)
    if not weights_file.exists():
        raise FileNotFoundError(f"Model weights not found at '{weights_path}'. Train the model first using train.py!")

    print(f"🔄 Loading model weights from {weights_path}...")
    model = YOLO(str(weights_file))

    print(f"📦 Exporting to ONNX format (imgsz={imgsz}, opset=11)...")
    onnx_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=11
    )
    print(f"✅ ONNX model successfully saved to: {onnx_path}")
    return onnx_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained PyTorch model to ONNX")
    parser.add_argument("--weights", type=str, default="", help="Path to best.pt (leave empty to auto-detect latest)")
    parser.add_argument("--imgsz", type=int, default=320, help="Image resolution for export")
    args = parser.parse_args()

    export_to_onnx(weights_path=args.weights, imgsz=args.imgsz)
