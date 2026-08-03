import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CLASS_NAMES = {
    0: "person",
    1: "backpack",
    2: "handbag",
    3: "suitcase",
    4: "rat",
    5: "mouse"
}

def verify_dataset(dataset_dir: str = "./dataset"):
    ds_path = Path(dataset_dir)
    if not ds_path.exists():
        print(f"❌ Error: Dataset directory '{dataset_dir}' does not exist!")
        return

    splits = ["train", "val"]

    print("=" * 60)
    print("📊 SECUREPI DATASET BOUNDING BOX VERIFICATION REPORT")
    print("=" * 60)

    total_instances_all = {cls_id: 0 for cls_id in CLASS_NAMES}

    for split in splits:
        labels_dir = ds_path / "labels" / split
        images_dir = ds_path / "images" / split

        if not labels_dir.exists():
            print(f"⚠️ Warning: Directory '{labels_dir}' not found!")
            continue

        label_files = list(labels_dir.glob("*.txt"))
        image_files = list(images_dir.glob("*.*"))

        counts = {cls_id: 0 for cls_id in CLASS_NAMES}
        files_with_boxes = 0

        for txt_file in label_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    files_with_boxes += 1
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id in counts:
                            counts[cls_id] += 1
                            total_instances_all[cls_id] += 1

        print(f"\n📂 Split: '{split}'")
        print(f"  • Image files count: {len(image_files)}")
        print(f"  • Label files count: {len(label_files)}")
        print(f"  • Label files with >=1 box: {files_with_boxes}")
        print("  • Bounding box counts per class:")
        for cls_id, name in CLASS_NAMES.items():
            print(f"    - Class {cls_id} ({name:<10}): {counts[cls_id]:>5} boxes")

    print("\n" + "=" * 60)
    print("📈 TOTAL BOUNDING BOX INSTANCES ACROSS ENTIRE DATASET:")
    print("=" * 60)
    for cls_id, name in CLASS_NAMES.items():
        box_count = total_instances_all[cls_id]
        if box_count >= 50:
            status = "✅ Sufficient"
        elif box_count > 0:
            status = "⚠️ Very Low (Needs More Data)"
        else:
            status = "❌ ZERO BOXES (MISSING DATASET!)"
        print(f"  Class {cls_id} ({name:<10}): {box_count:>5} boxes | Status: {status}")

if __name__ == "__main__":
    verify_dataset()
