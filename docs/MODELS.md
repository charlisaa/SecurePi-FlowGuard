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

## ⚠️ Known model-parser limitation — the custom FlowGuard model

> **TL;DR:** the stock SSD and NanoDet models work today. The **custom
> 6-class YOLOv8 model is NOT known to be decodable** by the current edge
> parser. Compiling an `.rpk` successfully does **not** prove it runs. This is
> an **open item that must be verified on the physical Raspberry Pi.**

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

i.e. **raw** predictions (box + per-class scores per anchor) that still require
**YOLO-specific decoding and Non-Max-Suppression on the host.** This is:

- **not** the SSD 3-tensor format (there is only one tensor → the SSD branch
  would raise `IndexError`/mis-index on `outputs[1]`/`outputs[2]`), and
- **not** NanoDet (a different layout → `postprocess_nanodet_detection` will not
  produce correct boxes).

**Conclusion: the custom YOLOv8 `.rpk` is not expected to decode with the
current parser as-is.** Do not assume an SSD, NanoDet, or "standard YOLO"
parser will just work.

### To document / decide before the model is usable on hardware

1. **Expected output tensor format** — confirm on the Pi what the compiled
   `.rpk` actually emits: run the model and inspect
   `imx500.get_outputs(metadata)` shapes/count, and
   `imx500.network_intrinsics.postprocess`.
2. **Required metadata** — `network_intrinsics` must report
   `task == "object detection"`, and the correct `bbox_order`,
   `bbox_normalization`, and `postprocess` values. If the converter did not
   embed post-processing, these will be wrong/empty.
3. **Required post-processing** — if the tensor is raw YOLOv8, a **new decode
   branch** must be added to `IMX500Detector.detect` (sigmoid/scores, xywh→xyxy,
   confidence filter, class-wise NMS). This code does **not** exist yet and is
   deliberately not faked.
4. **Label ordering** — `models/labels.txt` must list the 6 classes in the
   training index order (`person, backpack, handbag, suitcase, rat, mouse`).
   A wrong order silently mislabels every detection.
5. **Unresolved compatibility issue** — whether Sony's `imx500-converter` can
   emit an SSD-style post-processed head for a YOLOv8 network (so the existing
   SSD branch works), or whether host-side YOLO decoding is required.

### Must be tested on the physical Raspberry Pi

- Firmware upload + `.rpk` load without error.
- `get_outputs()` returns tensors in a shape the parser handles (or the new
  decode branch is exercised).
- `rat` and `mouse` detections appear with correct labels via `labels.txt`.
- Person/bag/pest routing behaves as in the off-device tests.
- End-to-end: an unattended bag alerts; a confirmed rat/mouse alerts on the
  **pest** path (never the bag path).
