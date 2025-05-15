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