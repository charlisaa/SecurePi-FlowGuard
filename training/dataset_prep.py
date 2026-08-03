import sys
import shutil
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def prepare_dataset(dataset_dir: Path = Path("./dataset")):
    """
    Downloads COCO and Open Images subsets with balanced class representation for:
    0: person, 1: backpack, 2: handbag, 3: suitcase, 4: rat, 5: mouse.
    """
    print("🧹 Initializing dataset directory structure...")
    if dataset_dir.exists():
        try:
            shutil.rmtree(dataset_dir)
        except Exception as e:
            print(f"Note clearing dataset dir: {e}")

    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)

    CLASS_MAP = {
        "person": 0,
        "Person": 0,
        "backpack": 1,
        "Backpack": 1,
        "handbag": 2,
        "Handbag": 2,
        "suitcase": 3,
        "Suitcase": 3,
        "Rat": 4,
        "rat": 4,
        "Rats": 4,
        "Hamster": 4,
        "hamster": 4,
        "Mouse": 5,
        "mouse": 5,
        "Mice": 5
    }

    dataset_yaml = """path: ./dataset
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
    with open(dataset_dir / "dataset.yaml", "w", encoding="utf-8") as f:
        f.write(dataset_yaml.strip())

    # Download COCO subset focused specifically on bags and people
    print("📥 [1/3] Downloading COCO subset (bags & people)...")
    coco_ds = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["backpack", "handbag", "suitcase", "person"],
        max_samples=600,
        dataset_name="coco_bags_subset"
    )

    # Download Open Images v7 subset for Rodents (Mouse & Hamster)
    print("📥 [2/3] Downloading Open Images v7 subset for Rodents (Mouse & Hamster)...")
    rodent_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Mouse", "Hamster"],
        max_samples=400,
        dataset_name="open_rodent_subset"
    )

    # Download Open Images v7 subset for Extra Bags (Backpack, Handbag, Suitcase)
    print("📥 [3/3] Downloading Open Images v7 subset for Extra Bags...")
    open_bags_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Backpack", "Handbag", "Suitcase"],
        max_samples=300,
        dataset_name="open_bags_subset"
    )

    def export_samples_to_yolo(samples, split_name):
        img_out = dataset_dir / "images" / split_name
        lbl_out = dataset_dir / "labels" / split_name

        person_only_count = 0

        for sample in samples:
            src_img_path = sample.filepath
            filename = Path(src_img_path).name
            stem = Path(src_img_path).stem

            yolo_lines = []
            has_non_person = False
            person_count = 0

            if sample.ground_truth is not None:
                for det in sample.ground_truth.detections:
                    label_str = det.label
                    if label_str in CLASS_MAP:
                        cls_id = CLASS_MAP[label_str]

                        if cls_id == 0:  # person
                            if person_count >= 2:  # Cap at max 2 person boxes per image
                                continue
                            person_count += 1
                        else:
                            has_non_person = True

                        x_min, y_min, box_w, box_h = det.bounding_box
                        x_center = x_min + (box_w / 2.0)
                        y_center = y_min + (box_h / 2.0)
                        yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

            # If image only has person boxes and no bags/rodents, skip 60% of them to balance classes
            if not has_non_person and person_count > 0:
                person_only_count += 1
                if person_only_count % 3 != 0:
                    continue

            if yolo_lines:
                dest_img_path = img_out / filename
                shutil.copy(src_img_path, dest_img_path)
                txt_path = lbl_out / f"{stem}.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))

    print("⚡ Converting & formatting dataset into YOLO format...")
    coco_samples = list(coco_ds)
    rodent_samples = list(rodent_ds)
    open_bags_samples = list(open_bags_ds)

    def split_data(samples, ratio=0.8):
        split_idx = int(len(samples) * ratio)
        return samples[:split_idx], samples[split_idx:]

    coco_train, coco_val = split_data(coco_samples)
    rodent_train, rodent_val = split_data(rodent_samples)
    bags_train, bags_val = split_data(open_bags_samples)

    export_samples_to_yolo(coco_train + rodent_train + bags_train, "train")
    export_samples_to_yolo(coco_val + rodent_val + bags_val, "val")

    print("✅ Balanced dataset successfully prepared in YOLO format at ./dataset!")

if __name__ == "__main__":
    prepare_dataset()
