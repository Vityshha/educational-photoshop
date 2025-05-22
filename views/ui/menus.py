from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QSpinBox
)


class CalcMenu(QDialog):

    finished = pyqtSignal()


    def __init__(self, parent=None):
        super(CalcMenu, self).__init__(parent)
        self.setWindowTitle("Вычисление статистик")

        layout = QVBoxLayout()

        self.min_max_button = QPushButton("Минимальная и максимальная амплитуда")
        layout.addWidget(self.min_max_button)

        self.mean_std_button = QPushButton("Среднее арифметическое и стандартное отклонение")
        layout.addWidget(self.mean_std_button)

        self.histogram_button = QPushButton("Построить гистограмму")
        layout.addWidget(self.histogram_button)

        self.setLayout(layout)

        self.is_selected_zone = False


    def set_pos(self, coords):
        self.setGeometry(coords)


    def switch_zone(self, is_selected: bool):
        self.is_selected_zone = is_selected


    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)



class SmoothingDialog(QDialog):

    signal_smoothing_image = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(SmoothingDialog, self).__init__(parent)
        self.setWindowTitle("Параметры сглаживания")

        layout = QVBoxLayout()

        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Радиус сглаживания:"))
        self.radius_input = QSpinBox()
        self.radius_input.setMinimum(1)
        self.radius_input.setMaximum(100)
        self.radius_input.setValue(3)
        radius_layout.addWidget(self.radius_input)
        layout.addLayout(radius_layout)

        self.confirm_button = QPushButton("Применить")
        self.confirm_button.clicked.connect(self.emit_radius)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def emit_radius(self):
        radius = self.radius_input.value()
        self.signal_smoothing_image.emit(radius)
        self.close()

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)


class DenoiseDialog(QDialog):
    signal_denoise_image = pyqtSignal(int)
    signal_estimate_noise = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(DenoiseDialog, self).__init__(parent)
        self.setWindowTitle("Параметры подавления шума")

        layout = QVBoxLayout()

        level_layout = QHBoxLayout()
        level_layout.addWidget(QLabel("Уровень подавления шума:"))
        self.level_input = QSpinBox()
        self.level_input.setMinimum(1)
        self.level_input.setMaximum(100)
        self.level_input.setValue(15)
        level_layout.addWidget(self.level_input)
        layout.addLayout(level_layout)

        button_layout = QHBoxLayout()

        self.confirm_button = QPushButton("Применить")
        self.confirm_button.clicked.connect(self.emit_level)
        button_layout.addWidget(self.confirm_button)

        self.measure_button = QPushButton("Измерить")
        self.measure_button.clicked.connect(self.emit_measure)
        button_layout.addWidget(self.measure_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def emit_level(self):
        level = self.level_input.value()
        self.signal_denoise_image.emit(level)
        self.close()

    def emit_measure(self):
        self.signal_estimate_noise.emit()

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)


class RotateDialog(QDialog):
    signal_rotate_image = pyqtSignal(float)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(RotateDialog, self).__init__(parent)
        self.setWindowTitle("Параметры поворота изображения")

        layout = QVBoxLayout()

        angle_layout = QHBoxLayout()
        angle_layout.addWidget(QLabel("Угол поворота (в градусах):"))
        self.angle_input = QSpinBox()
        self.angle_input.setMinimum(-360)
        self.angle_input.setMaximum(360)
        self.angle_input.setValue(90)
        angle_layout.addWidget(self.angle_input)
        layout.addLayout(angle_layout)

        button_layout = QHBoxLayout()

        self.confirm_button = QPushButton("Повернуть")
        self.confirm_button.clicked.connect(self.emit_angle)
        button_layout.addWidget(self.confirm_button)

        self.close_button = QPushButton("Отмена")
        self.close_button.clicked.connect(self.close)
        button_layout.addWidget(self.close_button)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def emit_angle(self):
        angle = self.angle_input.value()
        self.signal_rotate_image.emit(angle)
        self.close()

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)



class PixelEditDialog(QDialog):
    signal_get_amplitude = pyqtSignal(int, int)
    signal_set_amplitude = pyqtSignal(int, int, tuple)
    signal_build_piecewise = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(PixelEditDialog, self).__init__(parent)
        self.setWindowTitle("Редактирование амплитуды пикселя")

        layout = QVBoxLayout()

        # Координаты X и Y
        coord_layout = QHBoxLayout()
        coord_layout.addWidget(QLabel("X:"))
        self.x_input = QSpinBox()
        self.x_input.setMinimum(0)
        self.x_input.setMaximum(10000)
        coord_layout.addWidget(self.x_input)

        coord_layout.addWidget(QLabel("Y:"))
        self.y_input = QSpinBox()
        self.y_input.setMinimum(0)
        self.y_input.setMaximum(10000)
        coord_layout.addWidget(self.y_input)
        layout.addLayout(coord_layout)

        # Амплитуда (R, G, B)
        amp_layout = QHBoxLayout()
        amp_layout.addWidget(QLabel("Амплитуда R:"))
        self.r_input = QSpinBox()
        self.r_input.setRange(0, 255)
        amp_layout.addWidget(self.r_input)

        amp_layout.addWidget(QLabel("G:"))
        self.g_input = QSpinBox()
        self.g_input.setRange(0, 255)
        amp_layout.addWidget(self.g_input)

        amp_layout.addWidget(QLabel("B:"))
        self.b_input = QSpinBox()
        self.b_input.setRange(0, 255)
        amp_layout.addWidget(self.b_input)

        layout.addLayout(amp_layout)

        # Параметр block_size
        block_layout = QHBoxLayout()
        block_layout.addWidget(QLabel("Размер блока:"))
        self.block_input = QSpinBox()
        self.block_input.setRange(1, 500)
        self.block_input.setValue(32)
        block_layout.addWidget(self.block_input)
        layout.addLayout(block_layout)

        # Кнопки
        btn_layout = QHBoxLayout()
        self.get_button = QPushButton("Получить")
        self.get_button.clicked.connect(self.emit_get_amplitude)
        btn_layout.addWidget(self.get_button)

        self.set_button = QPushButton("Установить")
        self.set_button.clicked.connect(self.emit_set_amplitude)
        btn_layout.addWidget(self.set_button)

        layout.addLayout(btn_layout)

        # Построение мозаики
        self.build_button = QPushButton("Построить изображение с кусочно-постоянными амплитудами")
        self.build_button.clicked.connect(self.emit_build_piecewise)
        layout.addWidget(self.build_button)

        self.setLayout(layout)

    def emit_get_amplitude(self):
        x = self.x_input.value()
        y = self.y_input.value()
        self.signal_get_amplitude.emit(x, y)

    def emit_set_amplitude(self):
        x = self.x_input.value()
        y = self.y_input.value()
        rgb = (self.r_input.value(), self.g_input.value(), self.b_input.value())
        self.signal_set_amplitude.emit(x, y, rgb)

    def emit_build_piecewise(self):
        block_size = self.block_input.value()
        self.signal_build_piecewise.emit(block_size)

    def set_amplitude_value(self, rgb: tuple):
        r, g, b = rgb
        self.r_input.setValue(int(r))
        self.g_input.setValue(int(g))
        self.b_input.setValue(int(b))

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)

