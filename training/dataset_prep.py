import sys
import shutil
from pathlib import Path
import fiftyone as fo
import fiftyone.zoo as foz
import yaml

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Default location for a manually-downloaded real-rat dataset (see
# import_external_rat_images below for why this exists and what format it
# expects). Not part of the repo — the user drops an export here.
DEFAULT_RAT_SOURCE_DIR = Path("./training/external/rat_dataset")


def prepare_dataset(dataset_dir: Path = Path("./dataset"),
                     rat_source_dir: Path = DEFAULT_RAT_SOURCE_DIR):
    """
    Downloads COCO and Open Images subsets with balanced class representation for:
    0: person, 1: backpack, 2: handbag, 3: suitcase, 4: rat, 5: mouse.

    Real rat images are NOT part of this automated download: Open Images v7 has
    no boxable "Rat" class (only "Mouse"), so the rat class is populated from a
    separately downloaded dataset merged in via import_external_rat_images().
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
        # Hamster deliberately NOT mapped here (previously "Hamster": 4): Open
        # Images has no boxable "Rat" class, and a prior version of this script
        # used Hamster images as a stand-in for class 4. That trained a model
        # that had never seen a real rat (see docs/MODELS.md). Real rat images
        # now come from import_external_rat_images() instead.
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
    print("📥 [1/4] Downloading COCO subset (bags & people)...")
    coco_ds = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["backpack", "handbag", "suitcase", "person"],
        max_samples=600,
        dataset_name="coco_bags_subset"
    )

    # Download Open Images v7 subset for Mouse. NOTE: Open Images has no
    # boxable "Rat" class, so this intentionally covers "mouse" only — real
    # rat images are merged separately below via import_external_rat_images().
    print("📥 [2/4] Downloading Open Images v7 subset for Mouse...")
    rodent_ds = foz.load_zoo_dataset(
        "open-images-v7",
        split="validation",
        label_types=["detections"],
        classes=["Mouse"],
        max_samples=400,
        dataset_name="open_mouse_subset"
    )

    # Download Open Images v7 subset for Extra Bags (Backpack, Handbag, Suitcase)
    print("📥 [3/4] Downloading Open Images v7 subset for Extra Bags...")
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

    # [4/4] Merge in a manually-downloaded real-rat dataset, if present.
    print("📥 [4/4] Merging external rat dataset (if present)...")
    import_external_rat_images(dataset_dir, rat_source_dir)

    print("✅ Balanced dataset successfully prepared in YOLO format at ./dataset!")


# Source class name -> SecurePi class id. Lookup is case-insensitive. Extend
# this if a chosen source dataset uses different label spellings/plurals.
RAT_IMPORT_CLASS_MAP = {
    "rat": 4, "rats": 4,
    "mouse": 5, "mice": 5,
}

# Splits are recognised by common Roboflow/YOLO export folder names and
# collapsed onto this project's own train/val split (test folders, if any,
# are treated as additional val data rather than dropped).
_SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "val"}


def _read_source_names(source_dir: Path) -> dict:
    """Best-effort read of a Roboflow-style data.yaml/dataset.yaml 'names' field
    -> {index: name}. Returns {} if no yaml is found (labels are then skipped
    with a warning, since a bare class index alone can't be trusted)."""
    for candidate in ("data.yaml", "dataset.yaml"):
        yaml_path = source_dir / candidate
        if not yaml_path.exists():
            continue
        with open(yaml_path, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}
        names = spec.get("names", {})
        if isinstance(names, list):
            return {i: n for i, n in enumerate(names)}
        if isinstance(names, dict):
            return {int(i): n for i, n in names.items()}
    return {}


def import_external_rat_images(dataset_dir: Path, source_dir: Path,
                               class_map: dict = None) -> None:
    """Merge a separately-downloaded YOLO-format rodent dataset into dataset_dir,
    remapping the source's own class ids onto the SecurePi 6-class scheme.

    Open Images v7 has no boxable "Rat" class (only "Mouse"), so real rat
    images can't come from the automated fiftyone download above — they have
    to be sourced separately (e.g. a Roboflow Universe rat-detection dataset,
    exported in YOLO format) and dropped into ``source_dir`` by hand. See
    docs/MODELS.md for specific dataset recommendations.

    Expected layout (matches a standard Roboflow YOLO export): a data.yaml (or
    dataset.yaml) at the top listing the source's own class names, plus one or
    more split folders (train/valid/val/test), each with images/ and labels/
    subfolders of matching stems. Any source class not found in ``class_map``
    is skipped (with a per-file warning) rather than silently mislabeled — a
    dataset that ONLY has a "rat" class works with the default map unchanged;
    one with mixed classes (e.g. rat + mouse together) also works as long as
    both names appear in ``class_map``.

    A missing ``source_dir`` is not an error: this is an optional, manual step,
    so prepare_dataset() must still run standalone without it.
    """
    class_map = {k.lower(): v for k, v in (class_map or RAT_IMPORT_CLASS_MAP).items()}
    source_dir = Path(source_dir)
    if not source_dir.exists():
        print(f"  ⚠️  No external rat dataset found at '{source_dir}' — skipping. "
              f"The 'rat' class will have zero training images until one is added "
              f"(see docs/MODELS.md).")
        return

    names = _read_source_names(source_dir)
    if not names:
        print(f"  ⚠️  No data.yaml/dataset.yaml found under '{source_dir}' — cannot "
              f"trust bare class indices, skipping.")
        return

    copied = 0
    skipped_classes = set()
    for split_dir in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        dest_split = _SPLIT_ALIASES.get(split_dir.name.lower())
        if dest_split is None:
            continue  # not a recognised split folder (e.g. a 'README' dir) — ignore
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        img_out = dataset_dir / "images" / dest_split
        lbl_out = dataset_dir / "labels" / dest_split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = next((images_dir / f"{label_path.stem}{ext}"
                              for ext in (".jpg", ".jpeg", ".png")
                              if (images_dir / f"{label_path.stem}{ext}").exists()), None)
            if image_path is None:
                continue

            remapped_lines = []
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                src_name = names.get(int(parts[0]), "").lower()
                if src_name not in class_map:
                    skipped_classes.add(src_name or parts[0])
                    continue
                remapped_lines.append(" ".join([str(class_map[src_name]), *parts[1:]]))

            if not remapped_lines:
                continue

            # Prefix to avoid any filename collision with the fiftyone-sourced images.
            dest_stem = f"rat_ext_{label_path.stem}"
            shutil.copy(image_path, img_out / f"{dest_stem}{image_path.suffix}")
            (lbl_out / f"{dest_stem}.txt").write_text("\n".join(remapped_lines), encoding="utf-8")
            copied += 1

    if skipped_classes:
        print(f"  ⚠️  Ignored unmapped source classes: {sorted(skipped_classes)} "
              f"(add them to class_map if they should count).")
    print(f"  ✅ Merged {copied} externally-sourced image(s) into the dataset.")


if __name__ == "__main__":
    prepare_dataset()
