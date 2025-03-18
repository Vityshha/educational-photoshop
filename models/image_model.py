from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np

class ImageModel(QObject):

    signal_image_change = pyqtSignal(np.ndarray)

    def __init__(self, max_history_images=5):
        super().__init__()
        self.dict_images = {}
        self.max_history_images = max_history_images
        self.current_index = 0

    def image_set(self, image):
        if len(self.dict_images) > 1:
            self.current_index += 1

        if len(self.dict_images) >= self.max_history_images:
            oldest_key = min(self.dict_images.keys())
            del self.dict_images[oldest_key]

        self.dict_images[self.current_index] = image
        self.signal_image_change.emit(self.dict_images[self.current_index])


    def get_current_image(self):
        return self.dict_images[self.current_index]


    def get_pasted_image(self):
        if min(self.dict_images.keys()) > self.current_index - 1:
            self.current_index -= 1
            return self.dict_images[self.current_index]


    def get_following_image(self):
        if max(self.dict_images.keys()) < self.current_index + 1:
            self.current_index += 1
            return self.dict_images[self.current_index]