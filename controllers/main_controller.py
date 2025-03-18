import numpy as np
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QObject

from views.main_view import MainWindow
from models.image_model import ImageModel
from models.history_model import HistoryModel
from controllers.utils import Utils


class MainController(QObject):

    signal_send_image = pyqtSignal(np.ndarray)

    def __init__(self, image_model: ImageModel, history_model: HistoryModel, view: MainWindow):
        super().__init__()
        self.image_model = image_model
        self.history_model = history_model
        self.view = view
        self.connect_signals()
        self.connect_slot()


    def connect_signals(self):
        self.view.signal_open_image.connect(self.image_sender)
        self.view.signal_save_image.connect(self.save_image)
        self.view.signal_undo_image.connect(self.image_model.get_undo_image)
        self.view.signal_redo_image.connect(self.image_model.get_redo_image)

        self.image_model.signal_image_change.connect(self.view.put_image)


    def connect_slot(self):
        self.signal_send_image.connect(self.image_model.image_set)


    def image_sender(self, file_name: str):
        image = Utils.load_image(file_name)
        self.signal_send_image.emit(image)


    def save_image(self, file_name: str):
        image = self.image_model.get_current_image()
        Utils.save_image(file_name, image)

