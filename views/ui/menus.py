from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit, QFileDialog
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


class RandomSceneDialog(QDialog):
    signal_generate_scene = pyqtSignal(int, int, str, dict, int)  # h, w, mode, params, channels
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(RandomSceneDialog, self).__init__(parent)
        self.setWindowTitle("Генерация случайной сцены")

        layout = QVBoxLayout()

        # Размеры
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Высота (h):"))
        self.h_input = QSpinBox()
        self.h_input.setRange(1, 5000)
        self.h_input.setValue(512)
        size_layout.addWidget(self.h_input)

        size_layout.addWidget(QLabel("Ширина (w):"))
        self.w_input = QSpinBox()
        self.w_input.setRange(1, 5000)
        self.w_input.setValue(512)
        size_layout.addWidget(self.w_input)
        layout.addLayout(size_layout)

        # Выбор распределения
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Распределение:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["uniform", "normal"])
        self.mode_combo.currentTextChanged.connect(self.update_params_visibility)
        mode_layout.addWidget(self.mode_combo)
        layout.addLayout(mode_layout)

        # Параметры для равномерного распределения
        self.uniform_layout = QHBoxLayout()
        self.uniform_layout.addWidget(QLabel("a:"))
        self.a_input = QDoubleSpinBox()
        self.a_input.setRange(0, 255)
        self.a_input.setValue(0)
        self.uniform_layout.addWidget(self.a_input)

        self.uniform_layout.addWidget(QLabel("b:"))
        self.b_input = QDoubleSpinBox()
        self.b_input.setRange(0, 255)
        self.b_input.setValue(255)
        self.uniform_layout.addWidget(self.b_input)
        layout.addLayout(self.uniform_layout)

        # Параметры для нормального распределения
        self.normal_layout = QHBoxLayout()
        self.normal_layout.addWidget(QLabel("mean:"))
        self.mean_input = QDoubleSpinBox()
        self.mean_input.setRange(0, 255)
        self.mean_input.setValue(127)
        self.normal_layout.addWidget(self.mean_input)

        self.normal_layout.addWidget(QLabel("std:"))
        self.std_input = QDoubleSpinBox()
        self.std_input.setRange(0, 128)
        self.std_input.setValue(20)
        self.normal_layout.addWidget(self.std_input)
        layout.addLayout(self.normal_layout)

        # Каналы
        channels_layout = QHBoxLayout()
        channels_layout.addWidget(QLabel("Каналы:"))
        self.channels_input = QSpinBox()
        self.channels_input.setRange(1, 3)
        self.channels_input.setValue(1)
        channels_layout.addWidget(self.channels_input)
        layout.addLayout(channels_layout)

        # Кнопка генерации
        self.generate_button = QPushButton("Сгенерировать")
        self.generate_button.clicked.connect(self.emit_parameters)
        layout.addWidget(self.generate_button)

        self.setLayout(layout)
        self.update_params_visibility()

    def update_params_visibility(self):
        mode = self.mode_combo.currentText()
        self.uniform_layout.setEnabled(mode == "uniform")
        self.a_input.setEnabled(mode == "uniform")
        self.b_input.setEnabled(mode == "uniform")
        self.normal_layout.setEnabled(mode == "normal")
        self.mean_input.setEnabled(mode == "normal")
        self.std_input.setEnabled(mode == "normal")

    def emit_parameters(self):
        h = self.h_input.value()
        w = self.w_input.value()
        mode = self.mode_combo.currentText()
        channels = self.channels_input.value()

        if mode == "uniform":
            params = {"a": self.a_input.value(), "b": self.b_input.value()}
        else:
            params = {"mean": self.mean_input.value(), "std": self.std_input.value()}

        self.signal_generate_scene.emit(h, w, mode, params, channels)
        self.close()

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)


class SmoothedSceneDialog(QDialog):
    signal_generate_smoothed = pyqtSignal(int, int, int, float, float)  # h, w, r, mean, std
    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(SmoothedSceneDialog, self).__init__(parent)
        self.setWindowTitle("Генерация сглаженной сцены")

        layout = QVBoxLayout()

        # Размеры
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Высота (h):"))
        self.h_input = QSpinBox()
        self.h_input.setRange(1, 5000)
        self.h_input.setValue(512)
        size_layout.addWidget(self.h_input)

        size_layout.addWidget(QLabel("Ширина (w):"))
        self.w_input = QSpinBox()
        self.w_input.setRange(1, 5000)
        self.w_input.setValue(512)
        size_layout.addWidget(self.w_input)
        layout.addLayout(size_layout)

        # Радиус сглаживания
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Радиус сглаживания (r):"))
        self.r_input = QSpinBox()
        self.r_input.setRange(0, 100)
        self.r_input.setValue(3)
        radius_layout.addWidget(self.r_input)
        layout.addLayout(radius_layout)

        # Параметры нормального распределения
        dist_layout = QHBoxLayout()
        dist_layout.addWidget(QLabel("Среднее (m):"))
        self.mean_input = QDoubleSpinBox()
        self.mean_input.setRange(0, 255)
        self.mean_input.setValue(128)
        dist_layout.addWidget(self.mean_input)

        dist_layout.addWidget(QLabel("Ст. отклонение (σ):"))
        self.std_input = QDoubleSpinBox()
        self.std_input.setRange(0.1, 128)
        self.std_input.setValue(20)
        dist_layout.addWidget(self.std_input)
        layout.addLayout(dist_layout)

        # Кнопка генерации
        self.generate_button = QPushButton("Сгенерировать сцену")
        self.generate_button.clicked.connect(self.emit_parameters)
        layout.addWidget(self.generate_button)

        self.setLayout(layout)

    def emit_parameters(self):
        h = self.h_input.value()
        w = self.w_input.value()
        r = self.r_input.value()
        mean = self.mean_input.value()
        std = self.std_input.value()
        self.signal_generate_smoothed.emit(h, w, r, mean, std)
        self.close()

    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)


class ObjectProjectionDialog(QDialog):
    signal_run_projection = pyqtSignal(str, float, float, int, str)  # image_path, mask_path, mean, std, radius, distribution

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Добавление объекта по маске")
        self.setMinimumWidth(400)

        layout = QVBoxLayout()

        # Загрузка маски
        mask_layout = QHBoxLayout()
        self.mask_input = QLineEdit()
        btn_browse_mask = QPushButton("Обзор")
        btn_browse_mask.clicked.connect(self.browse_mask)
        mask_layout.addWidget(QLabel("Маска объекта:"))
        mask_layout.addWidget(self.mask_input)
        mask_layout.addWidget(btn_browse_mask)
        layout.addLayout(mask_layout)

        # Параметры распределения
        self.mean_input = QDoubleSpinBox()
        self.mean_input.setRange(0, 255)
        self.mean_input.setValue(128)
        self.mean_input.setSingleStep(1.0)

        self.std_input = QDoubleSpinBox()
        self.std_input.setRange(0, 128)
        self.std_input.setValue(20.0)
        self.std_input.setSingleStep(1.0)

        self.radius_input = QSpinBox()
        self.radius_input.setRange(0, 20)
        self.radius_input.setValue(2)

        dist_layout = QHBoxLayout()
        self.dist_select = QComboBox()
        self.dist_select.addItems(["normal", "uniform"])
        dist_layout.addWidget(QLabel("Распределение:"))
        dist_layout.addWidget(self.dist_select)
        layout.addLayout(dist_layout)

        layout.addWidget(QLabel("Среднее значение (mean):"))
        layout.addWidget(self.mean_input)

        layout.addWidget(QLabel("Стандартное отклонение (std):"))
        layout.addWidget(self.std_input)

        layout.addWidget(QLabel("Радиус сглаживания:"))
        layout.addWidget(self.radius_input)

        # Кнопка запуска
        run_btn = QPushButton("Добавить объект")
        run_btn.clicked.connect(self.emit_projection)
        layout.addWidget(run_btn)

        self.setLayout(layout)

    def browse_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", "", "Изображения (*.png *.jpg *.bmp)")
        if path:
            self.img_input.setText(path)

    def browse_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите маску", "", "Изображения (*.png *.jpg *.bmp)")
        if path:
            self.mask_input.setText(path)

    def emit_projection(self):
        mask_path = self.mask_input.text()
        mean = self.mean_input.value()
        std = self.std_input.value()
        radius = self.radius_input.value()
        dist = self.dist_select.currentText()

        if mask_path:
            self.signal_run_projection.emit(mask_path, mean, std, radius, dist)
            self.accept()
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Ошибка", "Укажите путь к изображению и маске.")