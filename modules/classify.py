import numpy as np
import cv2
import random

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
        ):
        '''
        * flip_vertical: (0 - 1); probability. ex) 0.5
        * flip_horizontal: (0 - 1); probability. ex) 0.5
        * rotate_degree: degree. ex) 45
        * rotate_prob: (0 - 1); probability for rotation. ex) 0.5
        * brightness_range_add: (0 - 2); img * brightness_mul + brightness_add. ex) [0.8, 1.2]
        * brightness_range_mul: (0 - 2); img * brightness_mul + brightness_add. ex) [0.8, 1.2]
        * beightness_range_ratio: (0 - 2); img + (img - 128) * brightness. ex) [0.8, 1.2]
        * noise_opacity_range: (0 - 1); np.random.rand([height, width, 3]) * opacity. ex) [0, 1e-3]  
        * translate_vertical_range: (-1 - 1); (x, y + translate * y) -> image center position. ex) [-0.5, 0.5]
        * transalte_horizontal_range: (-1 - 1); (x + translate * x, y) -> image center position. ex) [-0.5, 0.5]
        '''
        self.flip_vertical = flip_vertical
        self.flip_horizontal = flip_horizontal
        self.rotate_degree = rotate_degree
        self.rotate_prob = rotate_prob
        self.brightness_range_add = brightness_range_add
        self.brightness_range_mul = brightness_range_mul
        self.brightness_range_ratio = brightness_range_ratio
        self.noise_opacity_range = noise_opacity_range
        self.translate_vertical_range = translate_vertical_range
        self.translate_horizontal_range = translate_horizontal_range
        self.image = None
        self.label = None
    
    def data(self, image, label, batch_size=32):
        self.image = image
        self.label = label
        self.batch_size = batch_size
        self.index_range = len(image)
    
    def _flip(self, image):
        if self.flip_horizontal > np.random.rand():
            image = image[:, ::-1, :]

        if self.flip_vertical > np.random.rand():
            image = image[::-1, :, :]

        return image
    
    def _brightness(self, image):
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
        image += brightness_add
        image = image + (image - 128) * brightness_ratio

        return image
    
    def _noise(self, image):
        opacity_range = self.noise_opacity_range[1] - self.noise_opacity_range[0]
        opacity_offset = self.noise_opacity_range[0]
        height, width, _ = image.shape
        noise = np.random.rand(height, width, 3) * 255.0 * opacity_range + opacity_offset
        image += noise

        return image
    
    def _clip(self, image):
        under = image < 0
        over = image > 255

        image[under] = 0
        image[over] = 255

        return image
    
    def _translate(self, image):
        height, width, _ = image.shape
        start_x, end_x = np.array(self.translate_horizontal_range) * width
        start_y, end_y = np.array(self.translate_vertical_range) * height

        translate_x = (np.random.rand() * (end_x - start_x) + start_x)
        translate_y = (np.random.rand() * (end_y - start_y) + start_y)

        matrix = np.array([
            [1, 0, int(translate_x)],
            [0, 1, int(translate_y)],
        ], dtype=np.float32)

        height, width, _ = image.shape

        image = cv2.warpAffine(image, matrix, [width, height])

        return image
    
    def _rotate(self, image):
        degree = (np.random.rand() * 2 - 1) * self.rotate_degree if np.random.rand() < self.rotate_prob else 0
        height, width, _ = image.shape
        matrix = cv2.getRotationMatrix2D((height // 2, width // 2), degree, 1.0)
        image = cv2.warpAffine(image, matrix, [width, height])
        return image

    def augment(self):
        if self.image is None or self.label is None:
            raise Exception("image and label is None. Please set the data by data function")
        
        result = []
        for _ in range(self.batch_size):
            take_index = random.randint(0, self.index_range - 1)
            image = self.image[take_index].astype(np.float32)
            label = self.label[take_index]

            image = self._flip(image)
            image = self._brightness(image)
            image = self._noise(image)
            image = self._clip(image)
            image = self._translate(image)
            image = self._rotate(image)

            result.append([image, label])
        
        return result