# models/

Place the **compiled model** and its **labels file** here. These are the only
two files the Raspberry Pi edge runtime (`edge/securePi.py`) needs at inference
time.

```
models/
├── imx500_custom_securepi.rpk   # compiled IMX500 network — NOT committed (see below)
├── labels.txt                   # class names in output-index order (committed)
└── README.md                    # this file
```

## What is committed vs. what you provide

| File | Committed? | How you get it |
|------|-----------|----------------|
| `labels.txt` | ✅ yes | Shipped in the repo (class index order below). |
| `imx500_custom_securepi.rpk` | ❌ **no** — git-ignored (`*.rpk`) | Generated **off-device** by the training pipeline, then copied here manually. |

`.rpk`, `.pt`, and `.onnx` files are intentionally excluded from Git (see the
root `.gitignore`). Model artefacts are large and are produced separately — the
repository never stores them and never fabricates a placeholder.

## `labels.txt` — order matters

`edge/securePi.py` maps each detection's numeric class index to a label **by
line position** in this file, then routes it by name (person / unattended
object / pest). The order MUST match the training class ids
(`training/dataset_prep.py` → `dataset.yaml`):

```
0  person
1  backpack
2  handbag
3  suitcase
4  rat
5  mouse
```

If you retrain with a different class order, regenerate `labels.txt` to match,
or every label will be wrong.

## Deploying the compiled model

After compiling on a computer/Colab (see the root `README.md` → *Training
workflow*), copy both files onto the Pi and run:

```bash
python edge/securePi.py @edge/presets/lobby.args \
  --model models/imx500_custom_securepi.rpk \
  --labels models/labels.txt \
  --headless
```

## Ultralytics YOLOv8 IMX500 output support

The bundled model was exported with in-network NMS and returns four
output tensors:

1. bounding boxes,
2. confidence scores,
3. class indices,
4. number of valid detections.

`edge/securePi.py` reads the fourth tensor and ignores padded output
slots. This prevents an empty scene from creating dozens of false bag
tracks.
