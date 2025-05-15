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
    BY_SELECTION = 0
    BY_INTERPOLATION = 1


class CalcMode(Enum):
    MIN_MAX_AMP = 0
    MEAN_STD = 1
    HISTOGRAM = 2
