import sys
from PyQt5.QtWidgets import QApplication
from models.image_model import ImageModel
from views.main_view import MainWindow
from controllers.main_controller import MainController


class Application:
    def __init__(self):
        self.image_model = ImageModel()
        self.view = MainWindow(self.image_model)
        self.controller = MainController(self.image_model, self.view)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    application = Application()
    application.view.show()
    sys.exit(app.exec_())