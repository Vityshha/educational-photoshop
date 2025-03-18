# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\main_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.frame.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setLineWidth(0)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.cb_file = QtWidgets.QLabel(self.frame_3)
        self.cb_file.setStyleSheet("background-color: rgb(25, 121, 202); \n"
"color: rgb(255, 255, 255);\n"
"padding: 5px;\n"
"")
        self.cb_file.setObjectName("cb_file")
        self.horizontalLayout_6.addWidget(self.cb_file)
        self.btn_main = QtWidgets.QPushButton(self.frame_3)
        self.btn_main.setMouseTracking(True)
        self.btn_main.setStyleSheet("QPushButton {\n"
"    border-radius: 1px;       /* Закругленные углы */\n"
"    background-color: rgb(255, 255, 255);  /* Белый фон */\n"
"    min-width: 80px;          /* Минимальная ширина */\n"
"    padding: 5px;             /* Отступы внутри кнопки */\n"
"    border: none;             /* Убираем границу */\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"    background-color: rgb(245, 246, 247);  /* Серый фон при нажатии */\n"
"    padding: 5px;             /* Отступы внутри кнопки */\n"
"    border: none;             /* Убираем границу */\n"
"}")
        self.btn_main.setCheckable(True)
        self.btn_main.setChecked(True)
        self.btn_main.setObjectName("btn_main")
        self.horizontalLayout_6.addWidget(self.btn_main, 0, QtCore.Qt.AlignLeft)
        self.btn_tools = QtWidgets.QPushButton(self.frame_3)
        self.btn_tools.setStyleSheet("QPushButton {\n"
"    border-radius: 1px;       /* Закругленные углы */\n"
"    background-color: rgb(255, 255, 255);  /* Белый фон */\n"
"    min-width: 80px;          /* Минимальная ширина */\n"
"    padding: 5px;             /* Отступы внутри кнопки */\n"
"    border: none;             /* Убираем границу */\n"
"}\n"
"\n"
"QPushButton:checked {\n"
"    background-color: rgb(245, 246, 247);  /* Серый фон при нажатии */\n"
"    padding: 5px;             /* Отступы внутри кнопки */\n"
"    border: none;             /* Убираем границу */\n"
"}")
        self.btn_tools.setCheckable(True)
        self.btn_tools.setObjectName("btn_tools")
        self.horizontalLayout_6.addWidget(self.btn_tools, 0, QtCore.Qt.AlignLeft)
        self.verticalLayout_2.addWidget(self.frame_3, 0, QtCore.Qt.AlignLeft)
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 50))
        self.frame_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setLineWidth(0)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setSpacing(0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame_2)
        self.stackedWidget.setStyleSheet("background-color: rgb(245, 246, 247); ")
        self.stackedWidget.setLineWidth(0)
        self.stackedWidget.setObjectName("stackedWidget")
        self.main_page = QtWidgets.QWidget()
        self.main_page.setObjectName("main_page")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.main_page)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.btn_undo = QtWidgets.QPushButton(self.main_page)
        self.btn_undo.setStyleSheet("image: url(:/newPrefix/images/undo.png);")
        self.btn_undo.setText("")
        self.btn_undo.setObjectName("btn_undo")
        self.horizontalLayout_9.addWidget(self.btn_undo)
        self.btn_redo = QtWidgets.QPushButton(self.main_page)
        self.btn_redo.setStyleSheet("image: url(:/newPrefix/images/redo.png);")
        self.btn_redo.setText("")
        self.btn_redo.setObjectName("btn_redo")
        self.horizontalLayout_9.addWidget(self.btn_redo)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem)
        self.stackedWidget.addWidget(self.main_page)
        self.tools_page = QtWidgets.QWidget()
        self.tools_page.setObjectName("tools_page")
        self.stackedWidget.addWidget(self.tools_page)
        self.horizontalLayout_8.addWidget(self.stackedWidget)
        self.verticalLayout_2.addWidget(self.frame_2)
        self.verticalLayout.addWidget(self.frame)
        self.frame_paint = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_paint.sizePolicy().hasHeightForWidth())
        self.frame_paint.setSizePolicy(sizePolicy)
        self.frame_paint.setStyleSheet("background-color: rgb(201, 211, 226);")
        self.frame_paint.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_paint.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_paint.setObjectName("frame_paint")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.frame_paint)
        self.horizontalLayout_7.setContentsMargins(3, 3, 3, 3)
        self.horizontalLayout_7.setSpacing(3)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.lbl_paint = QtWidgets.QLabel(self.frame_paint)
        self.lbl_paint.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lbl_paint.setText("")
        self.lbl_paint.setScaledContents(True)
        self.lbl_paint.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_paint.setObjectName("lbl_paint")
        self.horizontalLayout_7.addWidget(self.lbl_paint)
        self.verticalLayout.addWidget(self.frame_paint)
        self.frame_lower_menu = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_lower_menu.sizePolicy().hasHeightForWidth())
        self.frame_lower_menu.setSizePolicy(sizePolicy)
        self.frame_lower_menu.setStyleSheet("background-color: rgb(240, 240, 240);")
        self.frame_lower_menu.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_lower_menu.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_lower_menu.setObjectName("frame_lower_menu")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_lower_menu)
        self.horizontalLayout.setContentsMargins(0, 0, 5, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_lower_menu)
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(5)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.btn_pos = QtWidgets.QPushButton(self.frame_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.btn_pos.sizePolicy().hasHeightForWidth())
        self.btn_pos.setSizePolicy(sizePolicy)
        self.btn_pos.setMaximumSize(QtCore.QSize(23, 23))
        self.btn_pos.setStyleSheet("background-color: transparent;\n"
"image: url(:/newPrefix/images/pos.png);")
        self.btn_pos.setText("")
        self.btn_pos.setIconSize(QtCore.QSize(12, 12))
        self.btn_pos.setObjectName("btn_pos")
        self.horizontalLayout_2.addWidget(self.btn_pos)
        self.lbl_pos = QtWidgets.QLabel(self.frame_4)
        self.lbl_pos.setObjectName("lbl_pos")
        self.horizontalLayout_2.addWidget(self.lbl_pos)
        self.lbl_rgb = QtWidgets.QLabel(self.frame_4)
        self.lbl_rgb.setObjectName("lbl_rgb")
        self.horizontalLayout_2.addWidget(self.lbl_rgb)
        self.horizontalLayout.addWidget(self.frame_4, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignBottom)
        self.line = QtWidgets.QFrame(self.frame_lower_menu)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout.addWidget(self.line)
        self.frame_5 = QtWidgets.QFrame(self.frame_lower_menu)
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(5)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.btn_select = QtWidgets.QPushButton(self.frame_5)
        self.btn_select.setMaximumSize(QtCore.QSize(23, 23))
        self.btn_select.setStyleSheet("background-color: transparent;\n"
"image: url(:/newPrefix/images/select.png);")
        self.btn_select.setText("")
        self.btn_select.setObjectName("btn_select")
        self.horizontalLayout_3.addWidget(self.btn_select, 0, QtCore.Qt.AlignLeft)
        self.lbl_select = QtWidgets.QLabel(self.frame_5)
        self.lbl_select.setObjectName("lbl_select")
        self.horizontalLayout_3.addWidget(self.lbl_select)
        self.horizontalLayout.addWidget(self.frame_5, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignBottom)
        self.line_2 = QtWidgets.QFrame(self.frame_lower_menu)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.horizontalLayout.addWidget(self.line_2)
        self.frame_6 = QtWidgets.QFrame(self.frame_lower_menu)
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_6)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(5)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.btn_size = QtWidgets.QPushButton(self.frame_6)
        self.btn_size.setMaximumSize(QtCore.QSize(23, 23))
        self.btn_size.setStyleSheet("background-color: transparent;\n"
"image: url(:/newPrefix/images/shape.png);")
        self.btn_size.setText("")
        self.btn_size.setObjectName("btn_size")
        self.horizontalLayout_4.addWidget(self.btn_size)
        self.lbl_size = QtWidgets.QLabel(self.frame_6)
        self.lbl_size.setObjectName("lbl_size")
        self.horizontalLayout_4.addWidget(self.lbl_size)
        self.horizontalLayout.addWidget(self.frame_6, 0, QtCore.Qt.AlignLeft|QtCore.Qt.AlignBottom)
        self.line_3 = QtWidgets.QFrame(self.frame_lower_menu)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.horizontalLayout.addWidget(self.line_3)
        self.frame_7 = QtWidgets.QFrame(self.frame_lower_menu)
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_7)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 4)
        self.horizontalLayout_5.setSpacing(5)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.lbl_scale = QtWidgets.QLabel(self.frame_7)
        self.lbl_scale.setObjectName("lbl_scale")
        self.horizontalLayout_5.addWidget(self.lbl_scale)
        self.slider = QtWidgets.QSlider(self.frame_7)
        self.slider.setMinimum(12)
        self.slider.setMaximum(800)
        self.slider.setProperty("value", 200)
        self.slider.setSliderPosition(200)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setInvertedAppearance(False)
        self.slider.setInvertedControls(False)
        self.slider.setTickPosition(QtWidgets.QSlider.NoTicks)
        self.slider.setTickInterval(0)
        self.slider.setObjectName("slider")
        self.horizontalLayout_5.addWidget(self.slider)
        self.horizontalLayout.addWidget(self.frame_7, 0, QtCore.Qt.AlignRight|QtCore.Qt.AlignBottom)
        self.verticalLayout.addWidget(self.frame_lower_menu)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.cb_file.setText(_translate("MainWindow", "Файл"))
        self.btn_main.setText(_translate("MainWindow", "Главная"))
        self.btn_tools.setText(_translate("MainWindow", "Инструменты"))
        self.lbl_pos.setText(_translate("MainWindow", "x:..., y:..."))
        self.lbl_rgb.setText(_translate("MainWindow", "R:255, G:255, B: 255"))
        self.lbl_select.setText(_translate("MainWindow", "620 x 514 пкс"))
        self.lbl_size.setText(_translate("MainWindow", "1024 x 1024 пкс"))
        self.lbl_scale.setText(_translate("MainWindow", "100%"))
import views.ui.res_rc