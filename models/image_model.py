from PyQt5.QtCore import QObject, pyqtSignal
import numpy as np


class ImageModel(QObject):
    signal_image_change = pyqtSignal(np.ndarray)

    def __init__(self, max_history_images=5):
        super().__init__()
        self.history_images = []
        self.max_history_images = max_history_images
        self.current_index = -1
        self.select_zone = None


    def image_set(self, image):
        if self.current_index != len(self.history_images) - 1:
            self.history_images = self.history_images[:self.current_index + 1]

        self.history_images.append(image)
        self.current_index += 1

        if len(self.history_images) > self.max_history_images:
            self.history_images.pop(0)
            self.current_index -= 1

        self.signal_image_change.emit(self.history_images[self.current_index])


    def get_current_image(self):
        if self.current_index >= 0:
            return self.history_images[self.current_index]
        return None


    def clear_select_zone(self):
        self.select_zone = None


    def put_select_zone(self, select_zone):
        self.select_zone = select_zone


    def get_select_zone(self):
        if self.select_zone is not None:
            return self.select_zone
        else:
            return self.history_images[self.current_index]


    def get_undo_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.signal_image_change.emit(self.history_images[self.current_index])


    def get_redo_image(self):
        if self.current_index < len(self.history_images) - 1:
            self.current_index += 1
            self.signal_image_change.emit(self.history_images[self.current_index])
