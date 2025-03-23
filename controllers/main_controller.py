import numpy as np
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject

from views.views_enums import ScaleMode
from views.main_view import MainWindow
from models.image_model import ImageModel
from controllers.utils import Utils


class MainController(QObject):

    signal_send_image = pyqtSignal(np.ndarray)
    signal_send_rgb = pyqtSignal(int, int, int)

    def __init__(self, image_model: ImageModel, view: MainWindow):
        super().__init__()
        self.image_model = image_model
        self.view = view
        self.connect_signals()
        self.connect_slot()


    def connect_signals(self):
        self.view.signal_open_image.connect(self.image_sender)
        self.view.signal_save_image.connect(self.save_image)
        self.view.signal_undo_image.connect(self.image_model.get_undo_image)
        self.view.signal_redo_image.connect(self.image_model.get_redo_image)
        self.view.signal_coordinates.connect(self.get_image_rgb)
        self.view.signal_scale_image.connect(self.scale_image)
        self.view.signal_grayscale_image.connect(self.convert_grayscale)
        self.view.signal_contrast_image.connect(self.contrast_stretching_metod)
        self.view.signal_quantized_image.connect(self.quantization_metod)

        self.image_model.signal_image_change.connect(self.view.put_image)

    def connect_slot(self):
        self.signal_send_image.connect(self.image_model.image_set)
        self.signal_send_rgb.connect(self.view.put_rgb_in_point)


#Methods
#-----------------------------------------------------------------


    def image_sender(self, file_name: str):
        image = Utils.load_image(file_name)
        if image is not None:
            self.signal_send_image.emit(image)


    def save_image(self, file_name: str):
        image = self.image_model.get_current_image()
        if image is not None:
            Utils.save_image(file_name, image)


    def get_image_rgb(self, x, y):
        image = self.image_model.get_current_image()
        if image is not None:
            r, g, b = Utils.get_rgb(image, x, y)
            self.signal_send_rgb.emit(r, g, b)

    def scale_image(self, metod, ratio):
        image = self.image_model.get_current_image()
        if image is not None:
            if metod == ScaleMode.BY_SELECTION.value:
                scaled_image = Utils.scale_image_subsampling(image, ratio)
            else:
                scaled_image = Utils.scale_image_interpolation(image, ratio)

            self.signal_send_image.emit(scaled_image)

    def convert_grayscale(self):
        image = self.image_model.get_current_image()
        if image is not None:
            grayscale_image = Utils.convert_to_24bit_grayscale(image)
            self.signal_send_image.emit(grayscale_image)

    def contrast_stretching_metod(self):
        image = self.image_model.get_current_image()
        if image is not None:
            contrast_stretching_image = Utils.contrast_stretching(image)
            self.signal_send_image.emit(contrast_stretching_image)

    def quantization_metod(self, levels=2):
        image = self.image_model.get_current_image()
        if image is not None:
            quantization_image = Utils.quantization(image, levels)
            self.signal_send_image.emit(quantization_image)