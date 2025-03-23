from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QPushButton
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

        self.smooth_button = QPushButton("Сглаживание амплитуд пикселей")
        layout.addWidget(self.smooth_button)

        self.setLayout(layout)

        self.is_selected_zone = False


    def set_pos(self, coords):
        self.setGeometry(coords)


    def switch_zone(self, is_selected: bool):
        self.is_selected_zone = is_selected


    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)