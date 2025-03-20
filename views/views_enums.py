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


class SelectMode(Enum):
    SELECT = "background-color: rgb(201, 224, 247);"
    UNSELECT = "background-color: rgb(245, 246, 247);"


class ScaleMode(Enum):
    BYSELECTION = 0
    BYINTERPOLATION = 1
