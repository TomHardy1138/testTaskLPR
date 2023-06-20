import easyocr
import numpy as np
import torch
from openvino.runtime import Core

YOLO_CAR = [2]
YOLO_MODELS = ["n", "s", "m", "l", "x"]


# Abstract class
class Inference:
    def __init__(self):
        pass

    def forward(self, data: np.array):
        pass


# Class for YOLOv5 PyTorch inference
# We use yolov5 for car detection
class TorchInference(Inference):
    def __init__(self, mode):
        assert (
            mode in YOLO_MODELS
        ), f"Wrong mode {mode}, need to be one of {YOLO_MODELS}"
        self.model = torch.hub.load("ultralytics/yolov5", f"yolov5{mode}")
        self.model.classes = YOLO_CAR

    def forward(self, data: np.array):
        """Return YOLOv5 detections in tensor([xyxy, confidence, label]) format"""
        return self.model(data).tolist()


# Class for OpenVINO inference
# Here we can use various models, but only vehicle attributes recognition for now
class OpenvinoInference(Inference):
    def __init__(self, filename):
        self.ie = Core()
        self.network = self.ie.read_model(filename, filename.replace(".xml", ".bin"))
        self.executable_network = self.ie.compile_model(self.network, device_name="CPU")

    def forward(self, data: np.array):
        """Return two tensors with color and type attributes of object\n
        Shapes: [1, 7] for colors and [1,4] for types"""
        return list(self.executable_network([data]).values())


# Class for EasyOCR inference
class OCRInference(Inference):
    def __init__(self):
        self.reader = easyocr.Reader(["ru", "en"])

    def forward(self, data: np.array):
        "Return list with predicted text from EasyOCR"
        return self.reader.readtext(data, detail=0)
