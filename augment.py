import os
import sys
import cv2
import numpy as np
import yaml

from modules.augment import Augmentation
from modules.utils import check_args, LoadImage, SaveImage

args = sys.argv
help = """
Data Augmentation Tool

Usage: python3 augment.py cfg.yaml

Format of cfg.yaml

```
mode: segment or classify

image_path: path/to/images
label_path: path/to/labels # only for segment mode
image_export: export/to/path/to/images
label_export: export/to/path/to/labels # only for segment mode
batch_size: augment_images_at_one_time
epochs: repeat_count
resize: resize_to_NxN
```
"""

if len(args) != 2 or args[1] == "help":
    print(help)
    sys.exit()

if not os.path.isfile(args[1]):
    print(help)
    sys.exit()

cfg = yaml.full_load(open(args[1], "r"))

check_args(cfg, help)

MODE = cfg["mode"]
IMAGE_PATH = cfg["image_path"]
IMAGE_EXPORT = cfg["image_export"]
LABEL_PATH = cfg["label_path"] if MODE == "segment" else None
LABEL_EXPORT = cfg["label_export"] if MODE == "segment" else None
IMAGE_SIZE = cfg["resize"] if "resize" in cfg.keys() else 640
EPOCHS = cfg["epochs"] if "epochs" in cfg.keys() else 50
BATCH_SIZE = cfg["batch_size"] if "batch_size" in cfg.keys() else 32

image_filename_list = []
label_filename_list = []
images_load = []
labels_load = []
images = []
labels = []
load_threads = []

augmentation = Augmentation(
    flip_vertical=cfg["flip_vertical"] if "flip_vertical" in cfg.keys() else 0.5,
    flip_horizontal=cfg["flip_horizontal"] if "flip_horizontal" in cfg.keys() else 0.5,
    rotate_degree=cfg["rotate_degree"] if "rotate_degree" in cfg.keys() else 0.0,
    rotate_prob=cfg["rotate_prob"] if "rotate_prob" in cfg.keys() else 0.0,
    brightness_range_add=cfg["brightness_range_add"] if "brightness_range_add" in cfg.keys() else [0.9, 1.1],
    brightness_range_mul=cfg["brightness_range_mul"] if "brightness_range_mul" in cfg.keys() else [1, 1],
    brightness_range_ratio=cfg["brightness_range_ratio"] if "brightness_range_ratio" in cfg.keys() else [1, 1],
    noise_opacity_range=cfg["noise_opacity_range"] if "noise_opacity_range" in cfg.keys() else [0, 0.125],
    translate_vertical_range=cfg["translate_vertical_range"] if "translate_vertical_range" in cfg.keys() else [0, 0.125],
    translate_horizontal_range=cfg["translate_horizontal_range"] if "translate_horizontal_range" in cfg.keys() else [0, 0],
    rescale_ratio_range=cfg["rescale_ratio_range"] if "rescale_ratio_range" in cfg.keys() else [0.8, 1.2],
)

if MODE == "segment":
    file_list = os.listdir(IMAGE_PATH)

    print("File scanning...")

    for file in file_list:
        image_file = os.path.join(IMAGE_PATH, file)
        label_file = os.path.join(LABEL_PATH, file)

        if os.path.isdir(image_file):
            for subfile in os.listdir(image_file):
                sub_image_file = os.path.join(image_file, subfile)
                sub_label_file = os.path.join(label_file, subfile)[:-4] + ".png"
                
                image_filename_list.append(sub_image_file)
                label_filename_list.append(sub_label_file)

            continue

        image_filename_list.append(image_file)
        label_filename_list.append(label_file)[:-4] + ".png"
    
    print("File loading...")

    images_load = [None for _ in image_filename_list]
    labels_load = [None for _ in label_filename_list]

    for i, (image, label) in enumerate(zip(image_filename_list, label_filename_list)):
        load_threads.append(LoadImage(image, images_load, i, IMAGE_SIZE))
        load_threads.append(LoadImage(label, labels_load, i, IMAGE_SIZE))

        load_threads[-2].start()
        load_threads[-1].start()
    
    for thread in load_threads:
        thread.join()

    for image, label in zip(images_load, labels_load):
        if image is None or label is None: continue
        images.append(image)
        labels.append(label)
    
    print("Start augmentation")
    augmentation.setMode("segment")
    augmentation.data(images, labels)

    if not os.path.isdir(IMAGE_EXPORT):
        os.makedirs(IMAGE_EXPORT)
    if not os.path.isdir(LABEL_EXPORT):
        os.makedirs(LABEL_EXPORT)
    
    for i in range(EPOCHS):
        batch = augmentation.augment()

        for j, (image, label) in enumerate(batch):
            filename = "%016d.png" % (i * BATCH_SIZE + j + 1)
            SaveImage(os.path.join(IMAGE_EXPORT, filename), image).start()
            SaveImage(os.path.join(LABEL_EXPORT, filename), label).start()

            print("saved %d/%d" % (i * BATCH_SIZE + j + 1, EPOCHS * BATCH_SIZE), end="\r")

elif MODE == "classify":
    class_list = os.listdir(IMAGE_PATH)

    print("File scanning...")
    for cls in class_list:
        class_path = os.path.join(IMAGE_PATH, cls)
        if not os.path.isdir(class_path): continue

        for file in os.listdir(class_path):
            image_filename_list.append(file)
            label_filename_list.append(cls)
    
    images_load = [None for _ in image_filename_list]
    labels_load = label_filename_list

    print("File loading...")
    for i, image in enumerate(image_filename_list):
        load_threads.append(LoadImage(image, images_load, i, IMAGE_SIZE))
        load_threads[-1].start()
    
    for thread in load_threads:
        thread.join()
    
    for image, label in (images_load, labels_load):
        if image is None: continue
        images.append(image)
        labels.append(label)
    
    print("Start augmentation")
    augmentation.setMode("classify")
    augmentation.data(images, labels)

    for i in range(EPOCHS):
        batch = augmentation.augment()

        for j, (image, label) in enumerate(batch):
            filename = "%016d.png" % (i * BATCH_SIZE + j + 1)
            filepath = os.path.join(IMAGE_EXPORT, label)

            if not os.path.isdir(filepath):
                os.mkdir(filepath)
            
            SaveImage(os.path.join(filepath, filename), image).start()
            print("saved %d/%d" % (i * BATCH_SIZE + j + 1, EPOCHS * BATCH_SIZE), end="\r")
