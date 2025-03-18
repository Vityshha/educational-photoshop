from models.image_model import ImageModel
from models.history_model import HistoryModel
from views.main_view import MainWindow

import sys
from PyQt5.QtWidgets import QApplication


class Application:
    def __init__(self):
        self.image_model = ImageModel()
        self.history_model = HistoryModel()
        self.view = MainWindow(self.image_model, self.history_model)
        # self.controller = MainController(self.image_model, self.history_model, self.view)
        pass



if __name__ == '__main__':
    app = QApplication(sys.argv)
    application = Application()
    application.view.show()
    sys.exit(app.exec_())