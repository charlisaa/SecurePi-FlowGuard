# Simple Setup Guide: Custom AI Model for SecurePi

This guide explains how to build and deploy a **custom multi-class model** (`person`, `backpack`, `handbag`, `suitcase`, `rat`, `mouse`) for SecurePi using **Google Colab (Free GPU)** and your **Raspberry Pi**.

---

## 💡 FAQ: Does Downloading a Dataset Include BOTH Images and Labels?

> [!IMPORTANT]
> **YES!** The automated script in **Colab Cell 2** downloads the photos AND automatically generates the matching `.txt` bounding box label files for all 6 target classes.
> 
> You **do NOT need to draw boxes or write text files manually**!

---

## ❓ What Needs to be Put in the `train/` Folders?

There are **two separate `train` subfolders** inside your dataset:

### 1. Inside `dataset/images/train/`
Contains **80% of photo files** (`.jpg`, `.jpeg`, or `.png`):
- `person01.jpg`
- `bag01.jpg`
- `rat01.jpg`
- `mouse01.jpg`

### 2. Inside `dataset/labels/train/`
Contains the **matching text annotation files** (`.txt`):
- `person01.txt`
- `bag01.txt`
- `rat01.txt`
- `mouse01.txt`

---

## 📂 Complete Dataset Folder Structure

```text
dataset/
├── dataset.yaml              <-- Main Configuration File
├── images/
│   ├── train/                <-- Put 80% of photos here (.jpg / .png)
│   │   ├── rat01.jpg
│   │   ├── person01.jpg
│   │   └── ...
│   └── val/                  <-- Put remaining 20% of photos here
│       ├── rat101.jpg
│       └── ...
└── labels/
    ├── train/                <-- Put matching text answer files here (.txt)
    │   ├── rat01.txt
    │   ├── person01.txt
    │   └── ...
    └── val/                  <-- Put matching text answer files here (.txt)
        ├── rat101.txt
        └── ...
```

---

## 🚀 Step-by-Step Setup Instructions

### Step 1: Open Google Colab
1. Go to `https://colab.research.google.com` in your web browser.
2. Click **New Notebook**.
3. In the top menu, click **Runtime** ➡️ **Change runtime type** ➡️ Select **T4 GPU** ➡️ Click **Save**.

---

### Step 2: Run Colab Code Cells (Copy & Paste)

Create 6 code cells in your Colab notebook and run them sequentially:

#### 📍 Cell 1: Install Tools
```python
# Check GPU and install tools
!nvidia-smi
!pip install -q ultralytics onnx onnxruntime imx500-converter fiftyone
print("✅ Environment Ready!")
```

#### 📍 Cell 2: Automated 100% Bulletproof Dataset Download & Class Mapping
This cell downloads COCO (people & bags) and Open Images (rats & mice), maps their class IDs cleanly `0..5`, and splits them 80% train / 20% validation:

```python
import os
import shutil
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz

# Setup clean directories
dataset_dir = Path("./dataset")
if dataset_dir.exists():
    shutil.rmtree(dataset_dir)

(dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
(dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
(dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
(dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

# Define unified class mapping
CLASS_MAP = {
    "person": 0,
    "backpack": 1,
    "handbag": 2,
    "suitcase": 3,
    "Rat": 4,
    "rat": 4,
    "Mouse": 5,
    "mouse": 5
}

dataset_yaml = """
path: ./dataset
train: images/train
val: images/val

names:
  0: person
  1: backpack
  2: handbag
  3: suitcase
  4: rat
  5: mouse
"""

with open(dataset_dir / "dataset.yaml", "w") as f:
    f.write(dataset_yaml.strip())

# Download COCO subset using FiftyOne
print("📥 [1/2] Downloading COCO dataset (people & bags)...")
coco_ds = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person", "backpack", "handbag", "suitcase"],
    max_samples=600,
    dataset_name="coco_subset"
)

# Download Open Images v7 subset for Rodents
print("📥 [2/2] Downloading Open Images dataset (rats & mice)...")
rodent_ds = foz.load_zoo_dataset(
    "open-images-v7",
    split="validation",
    label_types=["detections"],
    classes=["Rat", "Mouse"],
    max_samples=400,
    dataset_name="rodent_subset"
)

# Export function to format images & labels cleanly in YOLO format
def export_samples_to_yolo(samples, split_name):
    img_out = dataset_dir / "images" / split_name
    lbl_out = dataset_dir / "labels" / split_name
    
    for sample in samples:
        src_img_path = sample.filepath
        filename = Path(src_img_path).name
        stem = Path(src_img_path).stem
        
        # Copy image file
        dest_img_path = img_out / filename
        shutil.copy(src_img_path, dest_img_path)
        
        # Create YOLO label file
        txt_path = lbl_out / f"{stem}.txt"
        
        yolo_lines = []
        if sample.ground_truth is not None:
            for det in sample.ground_truth.detections:
                label_str = det.label
                if label_str in CLASS_MAP:
                    cls_id = CLASS_MAP[label_str]
                    # FiftyOne detections bounding_box: [x_min, y_min, width, height] (normalized 0..1)
                    x_min, y_min, box_w, box_h = det.bounding_box
                    x_center = x_min + (box_w / 2.0)
                    y_center = y_min + (box_h / 2.0)
                    yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
        
        with open(txt_path, "w") as f:
            f.write("\n".join(yolo_lines))

print("⚡ Formatting & pairing images into YOLO format...")
coco_samples = list(coco_ds)
rodent_samples = list(rodent_ds)

coco_split = int(len(coco_samples) * 0.8)
rodent_split = int(len(rodent_samples) * 0.8)

# 80% train, 20% val
export_samples_to_yolo(coco_samples[:coco_split], "train")
export_samples_to_yolo(coco_samples[coco_split:], "val")
export_samples_to_yolo(rodent_samples[:rodent_split], "train")
export_samples_to_yolo(rodent_samples[rodent_split:], "val")

print("✅ Dataset successfully downloaded, mapped, and ready for training!")
```

#### 📍 Cell 3: Train AI Model on GPU (~15 Minutes)
```python
from ultralytics import YOLO

# Load lightweight AI model template
model = YOLO("yolov8n.pt")

# Train model on all 6 target classes
model.train(
    data="./dataset/dataset.yaml",
    epochs=50,
    imgsz=320,
    batch=16,
    device=0,
    name="securepi_model"
)

print("🎉 Model training finished!")
```

#### 📍 Cell 4: Export to ONNX Format
```python
from ultralytics import YOLO

# Load fine-tuned model
trained_model = YOLO("runs/detect/securepi_model/weights/best.pt")

# Export to ONNX format
onnx_file = trained_model.export(
    format="onnx",
    imgsz=320,
    simplify=True,
    opset=11
)

print(f"✅ ONNX model saved to: {onnx_file}")
```

#### 📍 Cell 5: Compile to IMX500 Camera Format (`.rpk`)
```python
# Convert ONNX model to Sony IMX500 camera format (.rpk)
!imx500-converter convert \
    --input runs/detect/securepi_model/weights/best.onnx \
    --output imx500_custom_securepi.rpk \
    --input-shape 1 3 320 320 \
    --quantization-dataset ./dataset/images/val \
    --target-device imx500

print("🎉 imx500_custom_securepi.rpk successfully created!")
```

#### 📍 Cell 6: Download `.rpk` File to Your Computer
```python
from google.colab import files

# Download the final model file to your PC downloads folder
files.download("imx500_custom_securepi.rpk")
```

---

### Step 3: Copy Model to Raspberry Pi
1. Locate `imx500_custom_securepi.rpk` in your computer's **Downloads** folder.
2. Copy it (and `models/labels.txt`) into the repo's `models/` folder on the Pi.
   Replace `<user>`/`<pi-host>` with your own — do **not** assume `pi`:
   ```bash
   scp ~/Downloads/imx500_custom_securepi.rpk <user>@<pi-host>:~/SecurePi_FlowGuard/models/
   ```

---

### Step 4: Run SecurePi on Raspberry Pi
From the repository root on the Raspberry Pi:

```bash
python edge/securePi.py @edge/presets/kitchen.args \
    --model models/imx500_custom_securepi.rpk \
    --labels models/labels.txt \
    --unattended-time 10 \
    --headless
```

> **Label categories — do NOT mix them up.** `rat`/`mouse` are **pests**, handled
> by the separate pest logic. They must go in `--pest-labels` (the default), and
> must **never** appear in `--unattended-object-labels`. The custom model's
> classes are:
>
> - `--person-labels person`
> - `--unattended-object-labels backpack handbag suitcase`
> - `--pest-labels rat mouse`
>
> These are already the defaults, so the command above needs no label flags.

> ⚠️ **Compatibility caveat:** the custom model is a YOLOv8 export. Producing an
> `.rpk` does **not** guarantee the IMX500 parser in `edge/securePi.py` can decode
> its output tensor. See [MODELS.md](MODELS.md) → *Known model-parser limitation*
> and verify on the physical Pi before relying on the custom model.
