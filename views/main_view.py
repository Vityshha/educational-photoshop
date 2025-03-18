from PyQt5.QtWidgets import QMainWindow
from views.ui.main_window import Ui_MainWindow



class MainWindow(QMainWindow):
    def __init__(self, image_model, history_model):
        super().__init__()
        self.image_model = image_model
        self.history_model = history_model
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)


