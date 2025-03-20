from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import QComboBox




class FileComboBox(QComboBox):
    def __init__(self, qrect: QRect, parent=None):
        super().__init__(parent)
        self.addItems(["Открыть", 'Сохранить'])
        self.hide()
        self.setGeometry(qrect)


class SelectComboBox(QComboBox):
    def __init__(self, qrect: QRect, qrect_frame: QRect, parent=None):
        super().__init__(parent)
        self.addItems(["Прямоугольная область", "Произвольная область"])
        self.hide()
        self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.setGeometry(qrect.x(), qrect.y() + qrect_frame.width(), qrect.width(), qrect.height())
