from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QPainterPath
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QMessageBox
from views.ui.main_window import Ui_MainWindow
from views.views_enums import ComboBoxItem, StackedWidget, ComboBoxSelect, SelectMode
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QEvent, Qt, QRect, QPoint
import numpy as np

from views.custom_combo_box import FileComboBox, SelectComboBox
from views.views_enums import ScaleMode
from views.custom_dialog_window import ScaleMenu


class MainWindow(QMainWindow):

    signal_open_image = pyqtSignal(str)
    signal_save_image = pyqtSignal(str)
    signal_undo_image = pyqtSignal()
    signal_redo_image = pyqtSignal()
    signal_coordinates = pyqtSignal(int, int)
    signal_scale_image = pyqtSignal(int, float)
    signal_grayscale_image = pyqtSignal()

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
        self.is_selection_tool_active = False
        self.original_pixmap = None


    def init_ui(self):
        self.init_combo_box()
        self.init_dialog_window()
        self.ui.lbl_paint.installEventFilter(self)
        self.ui.lbl_paint.setMouseTracking(True)
        self.ui.stackedWidget.setCurrentIndex(StackedWidget.MAIN.value)
        self.ui.lbl_select.clear()

        if self.ui.lbl_paint.pixmap() is None:
            self.ui.lbl_paint.setPixmap(self.create_white_pixmap(self.ui.lbl_paint.size()))

        self.selection_start_point, self.selection_end_point = None, None


    def init_connections(self):
        self.ui.btn_main.clicked.connect(self.switch_button_status)
        self.ui.btn_tools.clicked.connect(self.switch_button_status)
        self.ui.btn_redo.clicked.connect(self.switch_image)
        self.ui.btn_undo.clicked.connect(self.switch_image)
        self.ui.slider.valueChanged.connect(self.slider_changed)
        self.ui.btn_select_frame.clicked.connect(self.select_mode)
        self.ui.btn_resize.clicked.connect(self.resize_image)
        self.ui.btn_grayscale.clicked.connect(self.grayscale_image)

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
        self.dialog_resize.finished.connect(self.close_dialog_resize)


    def close_dialog_resize(self):
        self.ui.frame_9.setStyleSheet(SelectMode.UNSELECT.value)
        self.ui.btn_resize.setChecked(False)

    def create_white_pixmap(self, size):
        pixmap = QPixmap(size)
        pixmap.fill(Qt.white)
        return pixmap

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
        if not self.ui.btn_select_frame.isChecked():
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
            if event.type() == QEvent.MouseMove:
                self.put_position_mouse(event.x(), event.y())

            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton and self.is_selection_tool_active:
                    scaled_point = self.scale_coordinates(event.x(), event.y())
                    if scaled_point is None:
                        return super().eventFilter(obj, event)
                    self.clear_selection()
                    self.drawing = True
                    self.start_point = scaled_point
                    self.end_point = scaled_point
                    self.selection_start_point = scaled_point
                    self.original_pixmap = self.ui.lbl_paint.pixmap().copy()

                    width = abs(self.end_point[0] - self.start_point[0])
                    height = abs(self.end_point[1] - self.start_point[1])
                    self.put_select_size(width, height)

                    self.points = [self.start_point]

                elif event.button() == Qt.RightButton:  # Обработка правой кнопки мыши
                    scaled_point = self.scale_coordinates(event.x(), event.y())
                    if scaled_point is not None and self.selection_start_point and self.selection_end_point:

                        if self.selection_type == ComboBoxSelect.RECTANGLE.value:
                            rect = QRect(QPoint(*self.selection_start_point),
                                         QPoint(*self.selection_end_point)).normalized()
                            if rect.contains(QPoint(*scaled_point)):
                                print('YES')
                            else:
                                print('NO')

                        elif self.selection_type == ComboBoxSelect.FREEHAND.value:
                            path = QPainterPath()
                            path.moveTo(QPoint(*self.points[0]))
                            for point in self.points[1:]:
                                path.lineTo(QPoint(*point))
                            if path.contains(QPoint(*scaled_point)):
                                print('YES')
                            else:
                                print('NO')

            elif event.type() == QEvent.MouseMove:
                scaled_point = self.scale_coordinates(event.x(), event.y())
                if scaled_point is not None:
                    self.end_point = scaled_point
                    self.signal_coordinates.emit(self.end_point[1], self.end_point[0])
                    if self.drawing and self.is_selection_tool_active:
                        self.update_temp_selection()
                        width = abs(self.end_point[0] - self.start_point[0])
                        height = abs(self.end_point[1] - self.start_point[1])
                        self.put_select_size(width, height)

                        if self.selection_type == ComboBoxSelect.FREEHAND.value:
                            self.points.append(self.end_point)
                else:
                    self.end_point = None
                    self.put_rgb_in_point('', '', '')

            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.is_selection_tool_active:
                    self.drawing = False
                    scaled_point = self.scale_coordinates(event.x(), event.y())
                    if scaled_point is None:
                        return super().eventFilter(obj, event)
                    self.end_point = scaled_point
                    self.selection_end_point = scaled_point
                    self.apply_selection()

                    width = abs(self.end_point[0] - self.start_point[0])
                    height = abs(self.end_point[1] - self.start_point[1])
                    self.put_select_size(width, height)

                    if self.selection_type == ComboBoxSelect.FREEHAND.value:
                        self.points.append(self.points[0])
                        self.update_temp_selection()

        return super().eventFilter(obj, event)

    def update_temp_selection(self):
        if self.start_point and self.end_point:
            # Если оригинальное изображение не загружено, создаем пустой QPixmap
            if self.original_pixmap is None:
                self.original_pixmap = self.create_white_pixmap(self.ui.lbl_paint.size())

            pixmap = self.original_pixmap.copy()
            painter = QPainter(pixmap)
            painter.setPen(QPen(Qt.red, 3, Qt.DashLine))

            start_point = QPoint(*self.start_point)
            end_point = QPoint(*self.end_point)

            if self.selection_type == ComboBoxSelect.RECTANGLE.value:
                rect = QRect(start_point, end_point)
                painter.drawRect(rect)
                print(rect.width(), rect.height())
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
            start_point = QPoint(*self.start_point)
            end_point = QPoint(*self.end_point)

            width = abs(end_point.x() - start_point.x())
            height = abs(end_point.y() - start_point.y())

            print(f"Выделенная область: {width}x{height}")

    def clear_selection(self):
        self.start_point = None
        self.end_point = None
        self.selection_mask = None
        self.ui.lbl_select.clear()
        if self.original_pixmap:
            self.ui.lbl_paint.setPixmap(self.original_pixmap)
        else:
            self.ui.lbl_paint.setPixmap(self.create_white_pixmap(self.ui.lbl_paint.size()))


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
        if image is None or image.size == 0:
            print("Изображение пустое или не загружено.")
            return

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if len(image.shape) == 2:  # Grayscale
            height, width = image.shape
            bytes_per_line = width
            q_img = QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
        elif len(image.shape) == 3:  # Многоканальное изображение (RGB, RGBA)
            height, width, channel = image.shape
            if channel == 3:  # RGB
                bytes_per_line = 3 * width
                q_img = QImage(image.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif channel == 4:  # RGBA
                bytes_per_line = 4 * width
                q_img = QImage(image.tobytes(), width, height, bytes_per_line, QImage.Format_RGBA8888)
            else:
                print(f"Неподдерживаемое количество каналов: {channel}")
                return
        else:
            print(f"Неподдерживаемая форма изображения: {image.shape}")
            return

        pixmap = QPixmap.fromImage(q_img)
        self.ui.lbl_paint.setPixmap(pixmap)
        self.ui.lbl_paint.setScaledContents(False)
        self.original_pixmap = pixmap


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
        pixmap = self.ui.lbl_paint.pixmap()
        if not pixmap:
            return None

        original_width = pixmap.width()
        original_height = pixmap.height()
        widget_width = self.ui.lbl_paint.width()
        widget_height = self.ui.lbl_paint.height()

        dx = (widget_width - original_width) // 2
        dy = (widget_height - original_height) // 2

        if (dx <= x < dx + original_width) and (dy <= y < dy + original_height):
            return (x - dx, y - dy)
        return None


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


    def apply_scaling(self):
        method = self.dialog_resize.method_combo.currentIndex()

        ratio = self.dialog_resize.ratio_input.value()

        if not ratio:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, заполните все поля.")
            return

        if method == ScaleMode.BYSELECTION.value and ratio > 1:
            QMessageBox.warning(self, "Ошибка", "Коэффициент масштабирования должен быть в диапазоне (0, 1) для уменьшения изображения.")
            return

        self.signal_scale_image.emit(method, ratio)


    def grayscale_image(self):
        self.signal_grayscale_image.emit()
