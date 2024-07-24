import threading
import cv2
import numpy as np
import sys
import os

def resize(image, target_size=640):
    height, width, _ = image.shape

    scale = target_size / max(height, width)
    scaled_width, scaled_height = int(width * scale), int(height * scale)

    image = cv2.resize(image, (scaled_width, scaled_height))

    pad_width = target_size - scaled_width
    pad_height = target_size - scaled_height

    top, bottom = pad_height // 2, pad_height - pad_height // 2
    left, right = pad_width // 2, pad_width - pad_width // 2

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return image

def check_args(args, help):
    if "mode" not in args.keys() \
        or args["mode"] not in ("segment", "classify") \
        or "image_path" not in args.keys() \
        or "image_export" not in args.keys() \
        or type(args["image_path"]) is not str \
        or type(args["image_export"]) is not str:
        print(help)
        sys.exit()
    
    if args["mode"] == "segment" and (
        "label_path" not in args.keys() \
        or "label_export" not in args.keys() \
        or type(args["label_path"]) is not str \
        or type(args["label_export"]) is not str
    ):
        print(help)
        sys.exit()

class LoadImage(threading.Thread):
    def __init__(self, filename, arr, index, image_size):
        super().__init__()
        self.filename = filename
        self.arr = arr
        self.index = index
        self.image_size = image_size
    
    def run(self):
        if not os.path.isfile(self.filename):
            self.arr[self.index] = None
            return
        
        image = cv2.imread(self.filename, cv2.IMREAD_COLOR)

        if image is None:
            self.arr[self.index] = None
            return
        
        image = resize(image, self.image_size)
        self.arr[self.index] = image.astype(np.uint8)

class SaveImage(threading.Thread):
    def __init__(self, filename, image):
        super().__init__()
        self.filename = filename
        self.image = image
    
    def run(self):
        cv2.imwrite(self.filename, self.image)

