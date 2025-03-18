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