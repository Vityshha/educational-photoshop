from enum import Enum


class ComboBoxItem(Enum):
    OPEN = 0
    SAVE = 1


class StackedWidget(Enum):
    MAIN = 0
    TOOLS = 1

class ComboBoxSelect(Enum):
    RECTANGLE = 0
    FREEHAND = 1