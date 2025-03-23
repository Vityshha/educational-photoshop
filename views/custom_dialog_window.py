from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QHBoxLayout, QMessageBox, QDoubleSpinBox
)


class ScaleMenu(QDialog):

    finished = pyqtSignal()

    def __init__(self, parent=None):
        super(ScaleMenu, self).__init__(parent)
        self.setWindowTitle("Масштабирование изображения")
        self.setGeometry(300, 300, 300, 150) # todo выставить центр

        layout = QVBoxLayout()

        self.method_label = QLabel("Выберите метод масштабирования:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Выборкой", "Интерполяцией"])
        layout.addWidget(self.method_label)
        layout.addWidget(self.method_combo)

        self.size_layout = QHBoxLayout()
        self.ratio = QLabel("Коэффициент:")
        self.ratio_input = QDoubleSpinBox()
        self.ratio_input.setRange(0.0, 10)
        self.ratio_input.setSingleStep(0.1)
        self.size_layout.addWidget(self.ratio)
        self.size_layout.addWidget(self.ratio_input)
        layout.addLayout(self.size_layout)

        self.confirm_button = QPushButton("Применить")
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)


    def closeEvent(self, event):
        self.finished.emit()
        super().closeEvent(event)