# Supported Object Labels (COCO Dataset)

The default IMX500 neural network (`imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`) is trained on the COCO (Common Objects in Context) dataset. This means you can track and monitor any of the 80 object categories listed below.

You assign labels to one of the three **functional categories** with these arguments:

- `--person-labels` — people (owner association). Default: `person`.
- `--unattended-object-labels` (alias: `--bag-labels`) — unattended-object candidates. Default: `backpack handbag suitcase`.
- `--pest-labels` — rodents handled by the separate pest logic. Default: `rat mouse`.

> ⚠️ **Never put `rat` or `mouse` in `--unattended-object-labels`.** Rodents use
> the pest path (own confidence, confirmation and cooldown, no owner). See the
> root README → *Supported classes*.
>
> ⚠️ **COCO vs. the custom model:** in the default COCO model `mouse` means a
> **computer mouse** and `rat` is **not a class at all**. Pest detection is only
> meaningful with the **custom FlowGuard model** (`person, backpack, handbag,
> suitcase, rat, mouse`) plus `models/labels.txt`.

## 🎒 Bags & Accessories
- `backpack`
- `handbag`
- `suitcase`
- `umbrella`
- `tie`

## 💻 Electronics & Indoor Items
- `laptop`
- `tv`
- `mouse`
- `keyboard`
- `cell phone`
- `remote`
- `clock`
- `book`
- `scissors`
- `teddy bear`
- `hair drier`
- `vase`
- `toothbrush`

## 🐕 Animals (Great for "attendants")
- `dog`
- `cat`
- `bird`
- `horse`
- `sheep`
- `cow`
- `elephant`
- `bear`
- `zebra`
- `giraffe`

## 🪑 Furniture & Home
- `chair`
- `couch`
- `potted plant`
- `bed`
- `dining table`
- `toilet`
- `microwave`
- `oven`
- `toaster`
- `sink`
- `refrigerator`

## 🚗 Vehicles & Outdoor
- `bicycle`
- `car`
- `motorcycle`
- `airplane`
- `bus`
- `train`
- `truck`
- `boat`
- `traffic light`
- `fire hydrant`
- `stop sign`
- `parking meter`
- `bench`

## ⚾ Sports
- `frisbee`
- `skis`
- `snowboard`
- `sports ball`
- `kite`
- `baseball bat`
- `baseball glove`
- `skateboard`
- `surfboard`
- `tennis racket`

## 🍎 Food & Dining
- `bottle`
- `wine glass`
- `cup`
- `fork`
- `knife`
- `spoon`
- `bowl`
- `banana`
- `apple`
- `sandwich`
- `orange`
- `broccoli`
- `carrot`
- `hot dog`
- `pizza`
- `donut`
- `cake`

## 🧍 People
- `person`
