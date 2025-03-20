from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from views.ui.main_window import Ui_MainWindow
from views.views_enums import ComboBoxItem, StackedWidget, ComboBoxSelect, SelectMode
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QEvent, Qt, QRect, QPoint
import numpy as np

from views.custom_combo_box import FileComboBox, SelectComboBox
from views.custom_dialog_window import ScaleMenu


class MainWindow(QMainWindow):

    signal_open_image = pyqtSignal(str)
    signal_save_image = pyqtSignal(str)
    signal_undo_image = pyqtSignal()
    signal_redo_image = pyqtSignal()
    signal_coordinates = pyqtSignal(int, int)

    def __init__(self, image_model):
        super().__init__()
        self.image_model = image_model
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.init_ui()
        self.init_connections()

        self.drawing = False
        self.selection_type = ComboBoxSelect.RECTANGLE.value
        self.start_point = None
        self.end_point = None
        self.selection_mask = None
        self.temp_pixmap = None
        self.is_selection_tool_active = False  # Флаг активности инструмента выделения
        self.original_pixmap = None  # Сохраняем исходное изображение


    def init_ui(self):
        self.init_combo_box()
        self.init_dialog_window()
        self.ui.lbl_paint.installEventFilter(self)
        self.ui.lbl_paint.setMouseTracking(True)
        self.ui.stackedWidget.setCurrentIndex(StackedWidget.MAIN.value)


    def init_connections(self):
        self.ui.btn_main.clicked.connect(self.switch_button_status)
        self.ui.btn_tools.clicked.connect(self.switch_button_status)
        self.ui.btn_redo.clicked.connect(self.switch_image)
        self.ui.btn_undo.clicked.connect(self.switch_image)
        self.ui.slider.valueChanged.connect(self.slider_changed)
        self.ui.btn_select_frame.clicked.connect(self.select_mode)
        self.ui.btn_resize.clicked.connect(self.resize_image)

        self.file_combo_box.activated.connect(self.on_combo_box_changed)
        self.select_combo_box.activated.connect(self.on_combo_box_select_change)



    def init_combo_box(self):
        self.file_combo_box = FileComboBox(self.ui.cb_file.geometry(), self)
        self.ui.cb_file.mousePressEvent = self.show_combo_box

        self.select_combo_box = SelectComboBox(self.ui.lbl_select_frame.geometry(), self.ui.frame_8.geometry(), self)
        self.ui.lbl_select_frame.mousePressEvent = self.show_combo_box_select


    def init_dialog_window(self):
        self.dialog_resize = ScaleMenu()
        self.dialog_resize.confirm_button.clicked.connect(self.apply_scaling)


    def show_combo_box(self, event):
        self.file_combo_box.showPopup()


    def show_combo_box_select(self, event):
        self.select_combo_box.showPopup()


    def on_combo_box_changed(self, index):
        selected_index = self.file_combo_box.currentIndex()
        if selected_index == ComboBoxItem.OPEN.value:
            print('Отправить сигнал об отрытии изображения')
            self.open_image()
        elif selected_index == ComboBoxItem.SAVE.value:
            print('Отрпавить сигнал о сохранении изображения')
            self.save_image()


    def on_combo_box_select_change(self, index):
        selected_index = self.select_combo_box.currentIndex()
        self.ui.btn_select_frame.click()
        if selected_index == ComboBoxSelect.RECTANGLE.value:
            print('Прямоугольная область')
            self.selection_type = ComboBoxSelect.RECTANGLE.value
            self.is_selection_tool_active = True
        else:
            print('Произвольная область')
            self.selection_type = ComboBoxSelect.FREEHAND.value
            self.is_selection_tool_active = True


    def select_mode(self):
        if self.ui.btn_select_frame.isChecked():
            self.ui.frame_8.setStyleSheet(SelectMode.SELECT.value)
            self.is_selection_tool_active = True
            self.selection_type = ComboBoxSelect.RECTANGLE.value
        else:
            self.ui.frame_8.setStyleSheet(SelectMode.UNSELECT.value)
            self.is_selection_tool_active = False
            self.clear_selection()


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

        if obj == self.ui.lbl_paint:
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton and self.is_selection_tool_active:
                    self.clear_selection()
                    self.drawing = True
                    self.start_point = self.scale_coordinates(event.x(), event.y())
                    self.end_point = self.start_point
                    self.original_pixmap = self.ui.lbl_paint.pixmap().copy()

                    width = abs(self.end_point[0] - self.start_point[0])
                    height = abs(self.end_point[1] - self.start_point[1])
                    self.put_select_size(width, height)

                    self.points = [self.start_point]

            elif event.type() == QEvent.MouseMove:
                self.end_point = self.scale_coordinates(event.x(), event.y())
                self.signal_coordinates.emit(self.end_point[1], self.end_point[0])
                self.put_position_mouse(self.end_point[0], self.end_point[1])
                if self.drawing and self.is_selection_tool_active:
                    self.update_temp_selection()
                    width = abs(self.end_point[0] - self.start_point[0])
                    height = abs(self.end_point[1] - self.start_point[1])
                    self.put_select_size(width, height)

                    if self.selection_type == ComboBoxSelect.FREEHAND.value:
                        self.points.append(self.end_point)

            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.is_selection_tool_active:
                    self.drawing = False
                    self.end_point = self.scale_coordinates(event.x(), event.y())
                    self.apply_selection()

                    width = abs(self.end_point[0] - self.start_point[0])
                    height = abs(self.end_point[1] - self.start_point[1])
                    self.put_select_size(width, height)

                    if self.selection_type == ComboBoxSelect.FREEHAND.value:
                        # Замыкаем путь
                        self.points.append(self.points[0])
                        self.update_temp_selection()

        return super().eventFilter(obj, event)


    def update_temp_selection(self):
        if self.start_point and self.end_point and self.original_pixmap:
            pixmap = self.original_pixmap.copy()
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))

            start_point = QPoint(*self.start_point)
            end_point = QPoint(*self.end_point)

            if self.selection_type == ComboBoxSelect.RECTANGLE.value:
                rect = QRect(start_point, end_point)
                painter.drawRect(rect)
            else:
                # Для произвольной области можно использовать QPainterPath
                path = QPainterPath()
                path.moveTo(QPoint(*self.points[0]))
                for point in self.points[1:]:
                    path.lineTo(QPoint(*point))
                painter.drawPath(path)

            painter.end()
            self.ui.lbl_paint.setPixmap(pixmap)


    def apply_selection(self):
        if self.start_point and self.end_point:
            pixmap = self.ui.lbl_paint.pixmap()
            if pixmap:
                start_point = QPoint(*self.start_point)
                end_point = QPoint(*self.end_point)

                width = abs(end_point.x() - start_point.x())
                height = abs(end_point.y() - start_point.y())
                # self.put_select_size(width, height)

                print(f"Выделенная область: {width}x{height}")


    def clear_selection(self):
        self.start_point = None
        self.end_point = None
        self.selection_mask = None
        if self.original_pixmap:
            self.ui.lbl_paint.setPixmap(self.original_pixmap)


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


    def put_position_mouse(self, x, y):
        self.ui.lbl_pos.setText(f"x: {x}, y: {y}")


    @pyqtSlot(int, int, int)
    def put_rgb_in_point(self, red, green, blue):
        self.ui.lbl_rgb.setText(f"R: {red}, G: {green}, B: {blue}")


    def get_scaled_image_size(self):
        scaled_width = self.ui.lbl_paint.width()
        scaled_height = self.ui.lbl_paint.height()
        return scaled_width, scaled_height


    def get_original_image_size(self):
        pixmap = self.ui.lbl_paint.pixmap()
        if pixmap:
            return pixmap.width(), pixmap.height()
        return None, None


    def scale_coordinates(self, x, y):
        original_width, original_height = self.get_original_image_size()
        scaled_width, scaled_height = self.get_scaled_image_size()

        if original_width is None or original_height is None:
            return x, y

        scale_x = original_width / scaled_width
        scale_y = original_height / scaled_height

        original_x = int(x * scale_x)
        original_y = int(y * scale_y)

        return original_x, original_y


    def put_lbl_scale(self, value):
        self.ui.lbl_scale.setText(str(value) + '%')


    def slider_changed(self):
        val_slider = self.ui.slider.value()
        scale_factor = val_slider / 100.0
        print('scale_factor:', scale_factor)
        self.put_lbl_scale(val_slider)


    def resize_image(self):
        if self.ui.btn_resize.isChecked():
            self.ui.frame_9.setStyleSheet(SelectMode.SELECT.value)
            self.dialog_resize.show()
        else:
            self.ui.frame_9.setStyleSheet(SelectMode.UNSELECT.value)
            print('sdasdsada')



    def apply_scaling(self):
        method = self.dialog_resize.method_combo.currentIndex()

        width = self.dialog_resize.width_input.text()
        height = self.dialog_resize.height_input.text()

        if not width or not height:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, заполните все поля.")
            return

        try:
            width = int(width)
            height = int(height)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Ширина и высота должны быть числами.")
            return

        self.dialog_resize.close()
        self.ui.btn_resize.setChecked(False)
        self.resize_image()

        print('scale: ', method, width, height)
        return method, width, height
