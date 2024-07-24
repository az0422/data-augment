import numpy as np
import cv2
import random
import math

class Augmentation():
    def __init__(self,
                 flip_vertical=0.5,
                 flip_horizontal=0.5,
                 rotate_degree=0,
                 rotate_prob=0.5,
                 brightness_range_add=[1, 1],
                 brightness_range_mul=[1, 1],
                 brightness_range_ratio=[1, 1],
                 noise_opacity_range=[0, 0],
                 translate_vertical_range=[0, 0],
                 translate_horizontal_range=[0, 0],
                 rescale_ratio_range=[1, 1],
        ):
        '''
        * flip_vertical: (0 - 1); probability. ex) 0.5
        * flip_horizontal: (0 - 1); probability. ex) 0.5
        * rotate_rad: radian. ex) 1.57
        * rotate_prob: (0 - 1); probability for rotation. ex) 0.5
        * brightness_range_add: (0 - 2); img * brightness_mul + brightness_add. ex) [0.8, 1.2]
        * brightness_range_mul: (0 - 2); img * brightness_mul + brightness_add. ex) [0.8, 1.2]
        * beightness_range_ratio: (0 - 2); img + (img - 128) * brightness. ex) [0.8, 1.2]
        * noise_opacity_range: (0 - 1); np.random.rand([height, width, 3]) * opacity. ex) [0, 1e-3]  
        * translate_vertical_range: (-1 - 1); (x, y + translate * y) -> image center position. ex) [-0.5, 0.5]
        * transalte_horizontal_range: (-1 - 1); (x + translate * x, y) -> image center position. ex) [-0.5, 0.5]
        * rescale_ratio_range: (0 - ); zoom scale for image. ex) [0.5, 3.0]
        '''
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.rotate_degree = rotate_degree * (180 / math.pi)
        self.rotate_prob = rotate_prob
        self.brightness_range_add = brightness_range_add
        self.brightness_range_mul = brightness_range_mul
        self.brightness_range_ratio = brightness_range_ratio
        self.noise_opacity_range = noise_opacity_range
        self.translate_vertical_range = translate_vertical_range
        self.translate_horizontal_range = translate_horizontal_range
        self.rescale_ratio_range = rescale_ratio_range
        self.images = None
        self.labels = None
        self.mode = "segment"
    
    def setMode(self, mode):
        if mode not in ("segment", "classify"):
            raise Exception("invalid mode %s" % (mode))
        self.mode = mode
    
    def data(self, images, labels, batch_size=32):
        if len(images) != len(labels):
            raise Exception("not matched image and label counts")

        if self.mode == "segment":
            for image, label in zip(images, labels):
                if image.shape != label.shape:
                    raise Exception("not matched image and label shape")

        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.index_range = len(images)
    
    def _flip(self, image, label):
        if self.flip_horizontal > np.random.rand():
            image = image[:, ::-1, :]

            if self.mode == "segment":
                label = label[:, ::-1, :]

        if self.flip_vertical > np.random.rand():
            image = image[::-1, :, :]

            if self.mode == "segment":
                label = label[::-1, :, :]

        return image, label
    
    def _brightness(self, image, label):
        add_range = (self.brightness_range_add[1] - self.brightness_range_add[0]) * 2 - 1
        add_offset = self.brightness_range_add[0] * 2 - 1
        mul_range = self.brightness_range_mul[1] - self.brightness_range_mul[0]
        mul_offset = self.brightness_range_mul[0]
        ratio_range = self.brightness_range_ratio[1] - self.brightness_range_ratio[0]
        ratio_offset = self.brightness_range_ratio[0]

        brightness_add = np.random.rand() * add_range + add_offset
        brightness_mul = np.random.rand() * mul_range + mul_offset
        brightness_ratio = np.random.rand() * ratio_range + ratio_offset

        image *= brightness_mul
        image = image + (image - 128) * brightness_ratio
        image += brightness_add

        return image, label
    
    def _noise(self, image, label):
        opacity_range = self.noise_opacity_range[1] - self.noise_opacity_range[0]
        opacity_offset = self.noise_opacity_range[0]
        height, width, _ = image.shape
        noise = np.random.rand(height, width, 3) * 255.0 * opacity_range + opacity_offset
        image += noise

        return image, label
    
    def _clip(self, image, label):
        under = image < 0
        over = image > 255

        image[under] = 0
        image[over] = 255

        return image, label
    
    def _translate(self, image, label):
        height, width, _ = image.shape
        start_x, end_x = np.array(self.translate_horizontal_range) * width
        start_y, end_y = np.array(self.translate_vertical_range) * height
        degree = (np.random.rand() * 2 - 1) * self.rotate_degree if np.random.rand() < self.rotate_prob else 0
        scale_range = self.rescale_ratio_range[1] - self.rescale_ratio_range[0]
        scale_offset = self.rescale_ratio_range[0]

        translate_x = (np.random.rand() * (end_x - start_x) + start_x)
        translate_y = (np.random.rand() * (end_y - start_y) + start_y)

        matrix = cv2.getRotationMatrix2D((height // 2, width // 2), degree,np.random.rand() * scale_range + scale_offset)
        matrix[0, 2] += translate_x
        matrix[1, 2] += translate_y

        height, width, _ = image.shape

        image = cv2.warpAffine(image, matrix, [width, height])

        if self.mode == "segment":
            label = cv2.warpAffine(label, matrix, [width, height])

        return image, label

    def augment(self):
        if self.images is None or self.labels is None:
            raise Exception("image and label is None. Please set the data by data function")
        
        result = []
        for _ in range(self.batch_size):
            take_index = random.randint(0, self.index_range - 1)
            image = self.images[take_index].astype(np.float32)
            label = self.labels[take_index].astype(np.float32)

            image, label = self._flip(image, label)
            image, label = self._translate(image, label)
            image, label = self._brightness(image, label)
            image, label = self._noise(image, label)
            image, label = self._clip(image, label)

            image = image.astype(np.uint8)
            
            if self.mode == "segment":
                label = label.astype(np.uint8)

            result.append([image, label])
        
        return result
    
    def generator(self):
        while True:
            yield self.augment()
