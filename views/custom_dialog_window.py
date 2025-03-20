from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QLineEdit, QPushButton, QHBoxLayout, QMessageBox
)


class ScaleMenu(QDialog):
    def __init__(self, parent=None):
        super(ScaleMenu, self).__init__(parent)
        self.setWindowTitle("Масштабирование изображения")
        self.setGeometry(300, 300, 300, 150)

        layout = QVBoxLayout()

        self.method_label = QLabel("Выберите метод масштабирования:")
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Выборкой", "Интерполяцией"])
        layout.addWidget(self.method_label)
        layout.addWidget(self.method_combo)

        self.size_layout = QHBoxLayout()
        self.width_label = QLabel("Ширина:")
        self.width_input = QLineEdit()
        self.height_label = QLabel("Высота:")
        self.height_input = QLineEdit()
        self.size_layout.addWidget(self.width_label)
        self.size_layout.addWidget(self.width_input)
        self.size_layout.addWidget(self.height_label)
        self.size_layout.addWidget(self.height_input)
        layout.addLayout(self.size_layout)

        self.confirm_button = QPushButton("Применить")
        # self.confirm_button.clicked.connect(self.apply_scaling)
        layout.addWidget(self.confirm_button)

        self.setLayout(layout)

    def apply_scaling(self):
        method = self.method_combo.currentIndex()

        width = self.width_input.text()
        height = self.height_input.text()

        if not width or not height:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, заполните все поля.")
            return

        try:
            width = int(width)
            height = int(height)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Ширина и высота должны быть числами.")
            return

        self.close()
        print('scale: ', method, width, height)
        return method, width, height

