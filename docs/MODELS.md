# Available IMX500 AI Models (`.rpk`)

When you install the `imx500-all` package on a Raspberry Pi using `sudo apt install imx500-all`, it automatically downloads a suite of pre-compiled neural network models to the `/usr/share/imx500-models/` directory.

The SecurePi application is designed to work with **Object Detection** models. However, the IMX500 camera supports several different types of AI tasks. Here is a summary of the common models included in that directory and what they do.

> [!IMPORTANT]
> **Supported Models for SecurePi**
> SecurePi requires **Object Detection** models that output bounding boxes. If you pass a Pose Estimation, Image Classification, or Semantic Segmentation model to SecurePi, the script will crash.
> 
> **Safe to use without crashing:**
> - `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`
> - `imx500_network_nanodet_plus_416x416_pp.rpk`

---

## 📦 1. Object Detection Models
These models draw "bounding boxes" around objects they recognize. **These are the only models that work with SecurePi.**

- **`imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`**
  - **Function:** This is the **default model** used by SecurePi. It provides a great balance of speed and accuracy. 
  - **Detects:** 80 COCO categories (people, bags, animals, vehicles, etc.).

- **`imx500_network_nanodet_plus_416x416_pp.rpk`**
  - **Function:** An alternative object detection model based on NanoDet Plus. It uses a different post-processing pipeline but is fully supported by SecurePi. It is often faster and better at detecting smaller objects.
  - **Detects:** 80 COCO categories.
  - **Usage in SecurePi:** `python edge/securePi.py @edge/presets/lobby.args --model /usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk`

---

## 🦴 2. Pose Estimation Models
These models do not draw boxes. Instead, they locate human joints (eyes, nose, shoulders, elbows, knees) to map out a human skeleton. *(Will crash if passed to SecurePi)*.

- **`imx500_network_posenet_mobilenet_v1_100_257x257_ptq_pp.rpk`**
  - **Function:** PoseNet model for tracking human movement and posture.

- **`imx500_network_movenet_single_pose_lightning_192x192_ptq_pp.rpk`**
  - **Function:** MoveNet Lightning. An extremely fast model designed specifically for tracking the high-speed movements of a single person (e.g., fitness tracking, gesture control).

---

## 🖼️ 3. Image Classification Models
These models do not output coordinates. They analyze the *entire* image and output a single label describing the dominant object in the scene. *(Will crash if passed to SecurePi)*.

- **`imx500_network_mobilenet_v1_1.0_224_quant_pp.rpk`**
  - **Function:** MobileNet v1 classifier.
  - **Detects:** 1,000 specific ImageNet categories (e.g., dog breeds, car models, distinct plant species).

- **`imx500_network_efficientnet_lite0_int8_pp.rpk`**
  - **Function:** EfficientNet Lite0 classifier. Similar to MobileNet but often yields slightly higher accuracy.
  - **Detects:** 1,000 ImageNet categories.

---

## ✂️ 4. Semantic Segmentation Models
These models process the image pixel-by-pixel, coloring and grouping pixels into categories to create a mask. *(Will crash if passed to SecurePi)*.

- **`imx500_network_deeplabv3_mnv2_257x257_pp.rpk`**
  - **Function:** DeepLabV3. Used for foreground/background separation (like video call background blurring) or detailed scene analysis.

---

## ✅ Custom FlowGuard model — parser support confirmed on hardware

> **TL;DR:** the stock SSD and NanoDet models work, and so does the custom
> 6-class YOLOv8 model. `edge/securePi.py`'s `IMX500Detector.detect()` has a
> decode branch for the Ultralytics IMX500 in-network-NMS export (4 output
> tensors: boxes/scores/classes/valid-count), and a real capture from the
> physical Pi (`imx_debug.log`, repo root) confirms the compiled
> `models/imx500_custom_securepi.rpk` actually emits that exact shape —
> `[(1,100,4), (1,100), (1,100), (1,1)]` — and was decoded into a correct
> `person` detection at 0.777 confidence. Compiling an `.rpk` successfully
> still does not *by itself* prove it runs (still verify any newly retrained
> model on hardware), but the parser support itself is no longer in question.

### What the edge parser (`edge/securePi.py` → `IMX500Detector.detect`) supports

The parser only understands **two** output formats, matching the official
Raspberry Pi object-detection demo:

| Format | Trigger | Output tensors it reads |
|--------|---------|-------------------------|
| **SSD "_pp"** | `intrinsics.postprocess != "nanodet"` | 3 tensors: `outputs[0]=boxes`, `outputs[1]=scores`, `outputs[2]=classes` (NMS already applied on-sensor) |
| **NanoDet** | `intrinsics.postprocess == "nanodet"` | raw tensor → `postprocess_nanodet_detection(...)` then `scale_boxes(...)` |

In both cases boxes are mapped to pixels with `convert_inference_coords`, and
the class **index** is looked up in the labels list **by position**.

### What a stock YOLOv8 export actually produces

`training/export_onnx.py` runs `YOLO(...).export(format="onnx", ...)`. A stock
YOLOv8 detection head exports a **single** output tensor of shape roughly:

```
[1, 4 + num_classes, num_anchors]   e.g. [1, 10, 2100] for 6 classes @ 320×320
```

i.e. **raw** predictions (box + per-class scores per anchor) that would need
YOLO-specific decoding and NMS on the host if exported this way. That is
**not** what actually ships, though: `models/install_model_on_pi.sh` /
`training/compile_imx500.py` compile with Sony's IMX500 converter using
Ultralytics' IMX500-specific export path, which bakes NMS into the network
itself and emits **4 output tensors** (boxes/scores/classes/valid-count) —
matching the `elif len(np_outputs) >= 4` branch in `IMX500Detector.detect`,
not the raw single-tensor YOLO head described above.

**Confirmed on hardware:** `imx_debug.log` (repo root) captured a real run of
the bundled `models/imx500_custom_securepi.rpk` on the physical Pi and shows
exactly the expected 4-tensor shape — `[(1,100,4), (1,100), (1,100), (1,1)]` —
decoded into a correct `person` detection at 0.777 confidence. The label order
in `models/labels.txt` (`person, backpack, handbag, suitcase, rat, mouse`)
matches the training index order in `training_artifacts/dataset.yaml`.

### Still worth checking after any future retrain

1. **Output tensor format** — re-run `imx500.get_outputs(metadata)` on the Pi
   after any retrain/recompile and confirm it's still the 4-tensor shape above;
   a different export path (e.g. a raw single-tensor YOLO head) would need a
   new decode branch in `IMX500Detector.detect`, which does not exist.
2. **Label ordering** — if you retrain with a different class order, regenerate
   `models/labels.txt` to match, or every label will be wrong.
3. **Per-class accuracy** — a successful decode is not the same as a good
   detector. See the root cause notes on the `rat` class (trained on Hamster
   images, not real rats) before assuming a parser problem for any future
   accuracy issue.

### Confirmed on the physical Raspberry Pi (imx_debug.log)

- Firmware upload + `.rpk` load without error.
- `get_outputs()` returns the expected 4-tensor shape and is decoded by the
  Ultralytics branch in `IMX500Detector.detect`.
- A `person` detection decoded with a correct label and plausible confidence.

### Still to verify end-to-end on hardware

- `rat` and `mouse` detections appear with correct labels via `labels.txt`
  (real deployment logs so far show pest alerts firing as `mouse` only, never
  `rat` — expected, given the `rat` class's training-data problem above).
- Person/bag/pest routing behaves as in the off-device tests.
- An unattended bag alerts; a confirmed pest alerts on the **pest** path
  (never the bag path).

---

## Retraining the `rat` class with real images

Real deployment logs (`runtime/logs/events.csv`) show pest alerts firing as
`mouse` only — never `rat` — across 250+ events. Root cause: the previous
`training/dataset_prep.py` mapped `"Hamster"` to class 4 (rat) because **Open
Images v7 has no boxable "Rat" class** (only "Mouse"), so the shipped model
never saw a real rat during training. `dataset_prep.py` no longer maps Hamster
to `rat`, so this class currently has **zero training images** until a real
source is added.

### Getting real rat images

Download a YOLO-format rat-detection export from a source such as:

- [Rat Object Detection Dataset (v2)](https://universe.roboflow.com/rat-jgq0z/rat-5ll0r-tbdic/dataset/2) — ~2,948 images
- [Rodent Detection and Repulsion](https://universe.roboflow.com/rodent-detection-6ctxr/rodent-detection-and-repulsion) — ~1,000 images, rat *and* mouse together (useful for keeping the two classes visually distinct)
- [Yolov5(Rat)](https://universe.roboflow.com/rat-detection/yolov5-rat) — ~692 images

Export in **YOLO format** (Roboflow supports this natively) and unzip into
`training/external/rat_dataset/` at the repo root, preserving its own
`data.yaml`/`dataset.yaml` and `train`/`valid`/`test` split folders as
downloaded — do not hand-edit the class ids.

### How the merge works

`import_external_rat_images()` in `dataset_prep.py` (called automatically as
step 4/4 of `prepare_dataset()`):

1. Reads the *source's own* class names from its `data.yaml`/`dataset.yaml` —
   it never trusts a bare class index alone, since a different dataset may
   order its classes differently.
2. Remaps each source class name to the SecurePi id via `RAT_IMPORT_CLASS_MAP`
   (`rat`/`rats` → 4, `mouse`/`mice` → 5); any other class in the source
   (e.g. a "background" class) is skipped with a printed warning, never
   silently mislabeled.
3. Collapses whatever split names the source uses (`train`/`valid`/`val`/
   `test`) onto this project's own `train`/`val` split.
4. Copies images and remapped labels into `dataset/`, prefixed `rat_ext_` to
   avoid any filename collision with the COCO/Open-Images-sourced files.

A missing `training/external/rat_dataset/` is **not an error** — it's a
manual, optional step, so `prepare_dataset()` still runs standalone (with a
warning that `rat` will have zero images) for anyone who hasn't downloaded a
dataset yet.

### Before shipping a retrained model

1. `python training/verify_dataset.py` — confirm `rat` clears "sufficient"
   box count (this only counts boxes, not accuracy).
2. After `python training/train.py`, check YOLO's own per-class
   precision/recall/mAP50 output for the `rat` row specifically — a good
   overall mAP can still hide a badly-performing single class.
3. Repeat the hardware verification done for `person` (see `imx_debug.log`
   above): put a real rat in front of the camera and confirm it decodes as
   `rat` — not `mouse`, not missed entirely — at a workable confidence.
