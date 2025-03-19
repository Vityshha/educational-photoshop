import cv2
import numpy as np


class Utils:

    @staticmethod
    def load_image(image_path: str):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def save_image(image_path: str, image: np.ndarray):
        cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    @staticmethod
    def get_rgb(image: np.ndarray, x, y):
        r = image[:,:,0][x, y]
        g = image[:,:,1][x, y]
        b = image[:,:,2][x, y]
        return r, g, b
