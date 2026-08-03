import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

def train_model(data_yaml: str = "./dataset/dataset.yaml", epochs: int = 50, batch: int = 16, imgsz: int = 320, name: str = "securepi_model", device: str = None):
    """
    Trains YOLOv8 nano model on the 6-class dataset.
    Automatically detects CUDA GPU or falls back to CPU.
    """
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    print(f"🏋️ Starting YOLOv8 training on device '{device}' (data={data_yaml}, epochs={epochs}, imgsz={imgsz})...")
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        name=name
    )
    print("🎉 Model training finished!")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom SecurePi YOLOv8 model")
    parser.add_argument("--data", type=str, default="./dataset/dataset.yaml", help="Path to dataset.yaml")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=320, help="Image resolution")
    parser.add_argument("--name", type=str, default="securepi_model", help="Run name")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., 'cpu', '0')")
    args = parser.parse_args()

    train_model(data_yaml=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, name=args.name, device=args.device)

