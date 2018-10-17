from PIL import Image, ImageEnhance
import numpy as np
import scipy.ndimage.interpolation as interpolation
from scipy.misc import imresize
from scipy.ndimage import sobel
from typing import Union

MASK_COLOR = (255, 0, 220)


class ImageHelper:
    PINK_THRESHOLD = 0.5

    @staticmethod
    def open_image(image_path):
        image = Image.open(image_path)
        return image

    @staticmethod
    def convert_image_to_numpy_array(image):
        image = np.array(image)
        return image

    @staticmethod
    def convert_numpy_array_to_image(image):
        pil_image = Image.fromarray(image)
        return pil_image

    @staticmethod
    def enhance_color(image, factor):
        color = ImageEnhance.Color(image)
        modified_image = color.enhance(factor=factor)
        return modified_image

    @staticmethod
    def enhance_contrast(image, factor):
        contrast = ImageEnhance.Contrast(image)
        image = contrast.enhance(factor=factor)
        return image

    @staticmethod
    def enhance_brightness(image, factor):
        brightness = ImageEnhance.Brightness(image)
        image = brightness.enhance(factor=factor)
        return image

    @staticmethod
    def enhance_sharpness(image, factor):
        sharpness = ImageEnhance.Sharpness(image)
        image = sharpness.enhance(factor=factor)
        return image

    @staticmethod
    def flip_image(image, axis):
        new_image = np.flip(image, axis)
        return new_image

    @staticmethod
    def rotate_image(image, angle):
        colors = {'r': 255,
                  'g': 0,
                  'b': 220}
        sorted_colors = sorted(colors.keys(), reverse=True)
        image_array = np.array(image)
        image_shape = image_array.shape
        output_image = np.zeros(image_shape, dtype=np.uint8)
        for i, color in enumerate(sorted_colors):
            one_color_image = image_array[:, :, i]
            rotated_image = interpolation.rotate(one_color_image, angle=angle, reshape=False, cval=colors[color])
            output_image[:, :, i] += rotated_image
        return output_image

    @classmethod
    def apply_sobel_filter(cls, image):
        image = np.array(image)
        image = cls.numpy_array_to_grayscale(image)
        dx = sobel(image, 0)
        dy = sobel(image, 1)
        sobel_image = np.array(np.hypot(dx, dy))
        return sobel_image

    @staticmethod
    def numpy_array_to_grayscale(image):
        return np.dot(image[..., :3], [0.299, 0.587, 0.114])

    @classmethod
    def get_tiles_from_image(cls, image, tile_size, stride=10):
        image_size = (image.shape[0], image.shape[1])
        tiles = []
        for y_pos in range(0, int(image_size[0]), stride):
            for x_pos in range(0, int(image_size[1]), stride):
                tile = image[y_pos:y_pos + tile_size[0], x_pos: x_pos + tile_size[1]]
                if tile.shape[0] == tile_size[0] and tile.shape[1] == tile_size[1]:
                    tiles.append(tile)
        return tiles

    @staticmethod
    def get_tile_from_image(image, x_pos, y_pos, tile_size):
        assert image.shape[0] >= x_pos + tile_size[0]
        assert image.shape[1] >= y_pos + tile_size[1]
        tile = image[x_pos:x_pos + tile_size[0], y_pos: y_pos + tile_size[1]]
        return tile

    @staticmethod
    def resize_image(image, size):
        resized_image = imresize(image, size, interp='bicubic')
        return resized_image

    # Takes a free-form-selected image.
    # Returns int: 1 or 0 corresponding to 'snake' and 'not snake' respectively,
    # based on concentration of snake in the image.
    @staticmethod
    def classify_extracted_tile(image, threshold):
        # noinspection PyTypeChecker
        num_snake_pxs = np.count_nonzero(np.all(image != MASK_COLOR, axis=-1))

        concentration = num_snake_pxs / (image.shape[0] * image.shape[1])
        return 1 if concentration >= threshold else 0

    @classmethod
    def check_if_snake_exists(cls, image):
        red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
        mask = (red >= 250) & (green >= 0) & (blue >= 220)
        pink_pixel_count = np.sum(mask)
        total_pixel_count = image.size / image.shape[2]
        pink_ratio = pink_pixel_count / total_pixel_count
        if pink_ratio <= cls.PINK_THRESHOLD:
            return True
        else:
            return False

    @classmethod
    def get_snake_number(cls, image, snake_type):
        exists = cls.check_if_snake_exists(image)
        if exists:
            if snake_type == 'king_cobra':
                return 1
            elif snake_type == 'indian_rock_python':
                return 2
        else:
            return 0

    @staticmethod
    def blend_images(img1: Union[np.ndarray, Image.Image],
                     img2: Union[np.ndarray, Image.Image], alpha: float):

        if isinstance(img1, np.ndarray):
            img1 = Image.fromarray(img1)
        if isinstance(img2, np.ndarray):
            img2 = Image.fromarray(img2)

        return np.array(Image.blend(img1, img2, alpha=alpha))
