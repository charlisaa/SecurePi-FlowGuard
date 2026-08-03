# Trained SecurePi Model Included

This project includes the trained six-class SecurePi model artefacts.

## Classes

0. person
1. backpack
2. handbag
3. suitcase
4. rat
5. mouse

## Included files

- `models/export/packerOut.zip` — Sony IMX500 packaging input
- `models/export/model_imx.onnx` — exported quantised model
- `models/labels.txt` — class order used by the runtime
- `training_artifacts/best.pt` — original trained YOLO weights
- `training_artifacts/dataset.yaml` — class/path configuration only

The full training image and label dataset is not included. It is not needed
for inference on the Raspberry Pi.

## Generate the RPK on the Raspberry Pi

```bash
cd SecurePi_FlowGuard-main
sudo apt update
sudo apt install -y imx500-tools imx500-all
bash models/install_model_on_pi.sh
```

This generates:

```text
models/imx500_custom_securepi.rpk
```

## Run a local test

```bash
python3 edge/securePi.py @edge/presets/demo.args           --model models/imx500_custom_securepi.rpk           --labels models/labels.txt           -v
```

## Accuracy note

Copying dataset images into the project does not make the deployed model
more accurate. Accuracy changes only after retraining and replacing the
compiled model.

The current runtime documentation also warns that custom YOLOv8 output
parsing must be verified on the physical IMX500 camera. Incorrect boxes or
labels may be a parser/output-format issue rather than missing dataset files.
