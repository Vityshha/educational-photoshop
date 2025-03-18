from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QComboBox, QFileDialog
from views.ui.main_window import Ui_MainWindow
from views.views_enums import ComboBoxItem, StackedWidget
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QEvent
import numpy as np
import cv2


class MainWindow(QMainWindow):

    signal_open_image = pyqtSignal(str)
    signal_save_image = pyqtSignal(str)
    signal_undo_image = pyqtSignal()
    signal_redo_image = pyqtSignal()

    def __init__(self, image_model, history_model):
        super().__init__()
        self.image_model = image_model
        self.history_model = history_model
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.init_ui()
        self.init_connections()


    def init_ui(self):
        self.init_combo_box()
        self.ui.lbl_paint.installEventFilter(self)


    def init_connections(self):
        self.ui.btn_main.clicked.connect(self.switch_button_status)
        self.ui.btn_tools.clicked.connect(self.switch_button_status)
        self.ui.btn_redo.clicked.connect(self.switch_image)
        self.ui.btn_undo.clicked.connect(self.switch_image)


    def init_combo_box(self):
        self.combo_box = QComboBox(self)
        self.combo_box.addItems(["Открыть", "Сохранить"])
        self.combo_box.hide()
        self.combo_box.setGeometry(self.ui.cb_file.geometry())
        self.ui.cb_file.mousePressEvent = self.show_combo_box
        self.combo_box.activated.connect(self.on_combo_box_changed)


    def show_combo_box(self, event):
        self.combo_box.showPopup()


    def on_combo_box_changed(self, index):
        selected_index = self.combo_box.currentIndex()
        if selected_index == ComboBoxItem.OPEN.value:
            print('Отправить сигнал об отрытии изображения')
            self.open_image()
        elif selected_index == ComboBoxItem.SAVE.value:
            print('Отрпавить сигнал о сохранении изображения')
            self.save_image()


    def switch_button_status(self):
        sender = self.sender()
        buttons = {
            self.ui.btn_main: self.ui.btn_tools,
            self.ui.btn_tools: self.ui.btn_main
        }
        if sender in buttons:
            buttons[sender].setChecked(False)
            sender.setChecked(True)
        if sender == self.ui.btn_main:
            self.ui.stackedWidget.setCurrentIndex(StackedWidget.MAIN.value)
        else:
            self.ui.stackedWidget.setCurrentIndex(StackedWidget.TOOLS.value)


    def switch_image(self):
        sender = self.sender()
        if sender == self.ui.btn_undo:
            print('Переключаем на прошлое состояние изображения')
            self.signal_undo_image.emit()
        else:
            print('Переключаем на следующее состояние если есть')
            self.signal_redo_image.emit()

    def eventFilter(self, obj, event):
        if obj == self.ui.lbl_paint and event.type() == QEvent.Resize:
            new_size = self.ui.lbl_paint.size()
            self.put_holst_size(new_size.width(), new_size.height())
        return super().eventFilter(obj, event)


    def open_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "",
                                                   "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.signal_open_image.emit(file_name)


    def save_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Сохранить изображение", "",
                                                   "Images (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if file_name:
            self.signal_save_image.emit(file_name)


    @pyqtSlot(np.ndarray)
    def put_image(self, image: np.ndarray):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        self.ui.lbl_paint.setPixmap(pixmap)
        self.ui.lbl_paint.setScaledContents(True)


    def put_holst_size(self, width, height):
        self.ui.lbl_size.setText(f"{width}x{height} пкс")


    def put_select_size(self, width, height):
        self.ui.lbl_select.setText(f"{width}x{height} пкс")