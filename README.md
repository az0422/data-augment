# Data Augmentation Library for Python

## Modules
 * `modules.classify.Augmentation`
 Augmenting data for classify

 * `modules.segment.Augmentation`
 Augmenting data for segmentation

## How to use
Example for segmentation

```
import cv2
import os
import threading
from modules.segmentat import Augmentation

IAMGE_ROOT = "path/to/images"
LABEL_ROOT = "path/to/labels"
EXPORT_IMAGE = "path/to/save/images"
EXPORT_LABEL = "path/to/save/labels"
BATCH_SIZE = size_for_batch
EPOCHS = count_for_loop

image_file_list = [[os.path.join(IMAGE_ROOT, file), os.path.join(LABEL_ROOT, file)] for file in os.listdir(IMAGE_ROOT)]
image_list = [None for _ in image_file_list]
label_list = [None for _ in image_file_list]

class LoadImage(threading.Thread):
    def __init__(self, paths, arrs, index):
        super().__init__()
        self.paths = paths
        self.arrs = arrs
        self.index = index
    
    def run(self):
        image_path, label_path = self.paths
        if not os.path.isfile(image_path) or not os.path.isfile(label_path): continue

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_COLOR)

        self.arr[0][self.index] = image
        self.arr[1][self.index] = label

class SaveImage(threading.Thread):
    def __init__(self, path, image):
        super().__init__()
        self.path = path
        self.image = image
    
    def run(self):
        cv2.imwrite(self.path, self.image)

load_threads = []
for index, (paths) in enumerate(image_file_list):
    load_threads.append(LoadImage(paths, [image_list, label_list], index))
    load_threads[-1].start()

for thread in load_threads:
    thread.join()

images_no_none = []
labels_no_none = []

for image, label in zip(image_list, label_list):
    if image is None or label is None: continue
    images_no_none.append(image)
    labels_no_none.append(label)

images = images_no_none
labels = labels_no_none

augmentation = Augmentation(
    flip_horizontal=0.5, # probability for fliping by horizon. range: 0 to 1.0
    flip_vertical=0.5, # probability for fliping by vertical. range: 0 to 1.0
    rotate_degree=90, # degree for rotating in range -degree to degree
    rotate_prob=0.5, # probability for rotation. range: 0 to 1.0
    brightness_range_add=[0.8, 1.2], # brightness range with add method. range: 0 to 2.0
    brightness_range_mul=[0.8, 1.2], # brightness range with muliply method. range: 0 to 2.0
    brightness_range_ratio=[0.8, 1.2], # brightness range with ratio multiply method. range: 0 to 2.0
    noise_opacity_range=[0, 0.5], # noise opacity range. range: 0 to 1.0
    translate_horizontal_range=[-0.5, 0.5], # translate to horizontal. range: -1.0 to 1.0
    translate_vertical_range=[-0.5, 0.5] # translate to vertical. range: -1.0 to 1.0
)

augmentation.data(images, labels, batch_size=BATCH_SIZE)

for i in range(EPOCHS):
    batch = augmentation.augment()

    for j, (image, label) in enumerate(batch):
        image_filename = "%016d.png" % (i * BATCH_SIZE + j + 1)
        SaveImage(os.path.join(EXPORT_IMAGE, image_filename), image).start()
        SaveImage(os.path.join(EXPORT_LABEL, image_filename), label).start()
```

## TODO list
 * Implement augmentation tool
