# YOLO IMX500 Post-Processing Fix

The custom Ultralytics IMX500 model allocates 100 output detection slots.
Only the first `valid_detection_count` slots are real detections.

The previous parser read all 100 slots and could accumulate dozens of
false bag tracks. Raising confidence to 0.90 hid the padding but also hid
genuine detections.

The corrected parser:

- reads the fourth valid-count tensor;
- slices boxes, scores and classes to that count;
- normalises Ultralytics XYXY boxes;
- converts them to the YXYX format expected by Picamera2;
- ignores invalid class indices.

Recommended first test:

```bash
python3 edge/securePi.py @demo.args           --model models/imx500_custom_securepi.rpk           --labels models/labels.txt           --min-confidence 0.25           --pest-confidence 0.25           --max-detections 10           -v
```
