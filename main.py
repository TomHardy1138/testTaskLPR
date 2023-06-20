import argparse
import os

import cv2
import numpy as np

from inference import OCRInference, OpenvinoInference, TorchInference

MODES = ["bgr", "rgb", "gray"]
ATTR_SIZE = (72, 72)

ATTR_COLOR = ["white", "gray", "yellow", "red", "green", "blue", "black"]
ATTR_TYPE = ["car", "van", "truck", "bus"]


def read_image(filename, mode="bgr"):
    assert mode in MODES, f"Unknown mode {mode}. Need one from {' '.join(MODES)}"

    image = cv2.imread(filename)

    if mode == "rgb":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif mode == "gray":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def prepare_image(data, size):
    """Resize, transpose (NCHW) and unsqueeze image before inference"""
    data = cv2.resize(data, size)
    data = data.transpose((2, 0, 1))
    data = np.expand_dims(data, axis=0)

    return data


class ImageConsumer:
    def __init__(self, args):
        self.attr_net = OpenvinoInference(args.attr_net)
        self.detect_car_net = TorchInference(mode=args.mode)
        self.lpr = OCRInference()

        self.debug = args.debug

    def _get_attributes(self, data):
        data = prepare_image(data, ATTR_SIZE)

        return self.attr_net.forward(data)

    def _get_car_detections(self, data):
        res = self.detect_car_net.forward(data)

        return [r.pred for r in res][0][0]

    def process_frame(self, img_path):
        assert os.path.exists(img_path), f"Couldn't find image {img_path}"

        image = read_image(img_path)

        cars = self._get_car_detections(image)
        for car in cars:
            car_np = car.cpu().numpy()

            xmin, ymin, xmax, ymax = car_np[:4].astype(np.int64)

            crop = image[ymin:ymax, xmin:xmax]

            plate = self.lpr.forward(crop)

            if plate:
                print(plate)
                plate = "".join(plate).replace(" ", "")
            else:
                plate = "Cannot recognize LP"

            attr = self._get_attributes(crop)
            color = ATTR_COLOR[np.argmax(attr[0])]
            type = ATTR_TYPE[np.argmax(attr[1])]

            result = f"{color.capitalize()} {type} with {plate.upper()} LP"

            if self.debug:
                dbg = crop.copy()
                h, w, _ = dbg.shape
                dbg = cv2.putText(
                    dbg,
                    result,
                    (w // 2, h // 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.55,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(result, dbg)
                cv2.waitKey(0)

            print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attr_net",
        type=str,
        default=os.path.join(
            "attribute_net", "vehicle-attributes-recognition-barrier-0042.xml"
        ),
        help="Path to attr net",
    )
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--mode", type=str, default="s", help="Size of YOLOv5")
    parser.add_argument("--debug", action="store_true", help="Use for crop debugging")

    args = parser.parse_args()

    i_s = ImageConsumer(args)
    i_s.process_frame(args.image)
