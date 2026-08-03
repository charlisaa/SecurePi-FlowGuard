import cv2
import numpy as np

from modlib.apps import Annotator
from modlib.devices import AiCamera
from modlib.models import COLOR_FORMAT, MODEL_TYPE, Model
from modlib.models.post_processors import pp_od_yolo_ultralytics


class SecurePiYOLO(Model):
    def __init__(self):
        super().__init__(
            model_file="models/export/packerOut.zip",
            model_type=MODEL_TYPE.CONVERTED,
            color_format=COLOR_FORMAT.RGB,
            preserve_aspect_ratio=False,
        )

        self.labels = np.genfromtxt(
            "models/labels.txt",
            dtype=str,
            delimiter="\n",
        )

    def post_process(self, output_tensors):
        return pp_od_yolo_ultralytics(output_tensors)


device = AiCamera(frame_rate=16)
model = SecurePiYOLO()

device.deploy(model)

annotator = Annotator()

with device as stream:
    for frame in stream:
        detections = frame.detections[
            frame.detections.confidence > 0.55
        ]

        labels = [
            f"{model.labels[class_id]}: {score:.2f}"
            for _, score, class_id, _ in detections
        ]

        annotator.annotate_boxes(
            frame,
            detections,
            labels=labels,
            alpha=0.3,
            corner_radius=10,
        )

        frame.display()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
