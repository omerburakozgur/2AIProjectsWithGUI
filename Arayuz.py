# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Arayuz.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1906, 1192)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(56, 56, 170, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(15, 7, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255, 202))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(56, 56, 170, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(15, 7, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(170, 170, 255, 202))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(56, 56, 170, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(125, 134, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(15, 7, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(15, 7, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        MainWindow.setPalette(palette)
        MainWindow.setAutoFillBackground(True)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 30, 1871, 1121))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setAutoFillBackground(True)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.groupBox_3 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_3.setGeometry(QtCore.QRect(300, 50, 1191, 381))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setAutoFillBackground(True)
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_3 = QtWidgets.QLabel(self.groupBox_3)
        self.label_3.setGeometry(QtCore.QRect(470, 40, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_3.setFont(font)
        self.label_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.cmbMimari = QtWidgets.QComboBox(self.groupBox_3)
        self.cmbMimari.setGeometry(QtCore.QRect(390, 100, 381, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.cmbMimari.setFont(font)
        self.cmbMimari.setObjectName("cmbMimari")
        self.cmbMimari.addItem("")
        self.cmbMimari.addItem("")
        self.cmbMimari.addItem("")
        self.cmbMimari.addItem("")
        self.cmbMimari.addItem("")
        self.cmbMimari.addItem("")
        self.btnModelEgit = QtWidgets.QPushButton(self.groupBox_3)
        self.btnModelEgit.setGeometry(QtCore.QRect(460, 170, 231, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnModelEgit.setFont(font)
        self.btnModelEgit.setObjectName("btnModelEgit")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(490, 240, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_6.setFont(font)
        self.label_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_6.setTextFormat(QtCore.Qt.PlainText)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.lblTahmin = QtWidgets.QLabel(self.groupBox_3)
        self.lblTahmin.setGeometry(QtCore.QRect(490, 300, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.lblTahmin.setFont(font)
        self.lblTahmin.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblTahmin.setTextFormat(QtCore.Qt.PlainText)
        self.lblTahmin.setAlignment(QtCore.Qt.AlignCenter)
        self.lblTahmin.setObjectName("lblTahmin")
        self.groupBox_5 = QtWidgets.QGroupBox(self.tab_3)
        self.groupBox_5.setGeometry(QtCore.QRect(300, 460, 1191, 401))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_5.sizePolicy().hasHeightForWidth())
        self.groupBox_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setAutoFillBackground(True)
        self.groupBox_5.setObjectName("groupBox_5")
        self.btnModelKullan = QtWidgets.QPushButton(self.groupBox_5)
        self.btnModelKullan.setGeometry(QtCore.QRect(60, 230, 171, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnModelKullan.setFont(font)
        self.btnModelKullan.setObjectName("btnModelKullan")
        self.cmbModel = QtWidgets.QComboBox(self.groupBox_5)
        self.cmbModel.setGeometry(QtCore.QRect(60, 170, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.cmbModel.setFont(font)
        self.cmbModel.setObjectName("cmbModel")
        self.cmbModel.addItem("")
        self.cmbModel.addItem("")
        self.cmbModel.addItem("")
        self.cmbModel.addItem("")
        self.cmbModel.addItem("")
        self.cmbModel.addItem("")
        self.label_4 = QtWidgets.QLabel(self.groupBox_5)
        self.label_4.setGeometry(QtCore.QRect(30, 110, 241, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.txtHamilelik = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtHamilelik.setGeometry(QtCore.QRect(410, 120, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtHamilelik.setFont(font)
        self.txtHamilelik.setObjectName("txtHamilelik")
        self.label_5 = QtWidgets.QLabel(self.groupBox_5)
        self.label_5.setGeometry(QtCore.QRect(340, 80, 191, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.groupBox_5)
        self.label_7.setGeometry(QtCore.QRect(540, 80, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_7.setTextFormat(QtCore.Qt.AutoText)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.txtKanSekeri = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtKanSekeri.setGeometry(QtCore.QRect(600, 120, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtKanSekeri.setFont(font)
        self.txtKanSekeri.setObjectName("txtKanSekeri")
        self.txtKanBasinci = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtKanBasinci.setGeometry(QtCore.QRect(770, 120, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtKanBasinci.setFont(font)
        self.txtKanBasinci.setObjectName("txtKanBasinci")
        self.label_8 = QtWidgets.QLabel(self.groupBox_5)
        self.label_8.setGeometry(QtCore.QRect(710, 80, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_8.setTextFormat(QtCore.Qt.AutoText)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox_5)
        self.label_9.setGeometry(QtCore.QRect(290, 200, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_9.setTextFormat(QtCore.Qt.AutoText)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.txtInsulin = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtInsulin.setGeometry(QtCore.QRect(360, 240, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtInsulin.setFont(font)
        self.txtInsulin.setObjectName("txtInsulin")
        self.label_10 = QtWidgets.QLabel(self.groupBox_5)
        self.label_10.setGeometry(QtCore.QRect(1010, 200, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_10.setFont(font)
        self.label_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_10.setTextFormat(QtCore.Qt.AutoText)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.txtYas = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtYas.setGeometry(QtCore.QRect(1040, 240, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtYas.setFont(font)
        self.txtYas.setObjectName("txtYas")
        self.label_16 = QtWidgets.QLabel(self.groupBox_5)
        self.label_16.setGeometry(QtCore.QRect(750, 200, 251, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_16.setFont(font)
        self.label_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_16.setTextFormat(QtCore.Qt.AutoText)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.txtGenetik = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtGenetik.setGeometry(QtCore.QRect(840, 240, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtGenetik.setFont(font)
        self.txtGenetik.setObjectName("txtGenetik")
        self.txtVucutKitle = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtVucutKitle.setGeometry(QtCore.QRect(590, 240, 61, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtVucutKitle.setFont(font)
        self.txtVucutKitle.setObjectName("txtVucutKitle")
        self.label_17 = QtWidgets.QLabel(self.groupBox_5)
        self.label_17.setGeometry(QtCore.QRect(500, 200, 241, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_17.setFont(font)
        self.label_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_17.setTextFormat(QtCore.Qt.AutoText)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.groupBox_5)
        self.label_18.setGeometry(QtCore.QRect(890, 80, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_18.setFont(font)
        self.label_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_18.setTextFormat(QtCore.Qt.AutoText)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.txtDeri = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtDeri.setGeometry(QtCore.QRect(950, 120, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtDeri.setFont(font)
        self.txtDeri.setObjectName("txtDeri")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.label_2 = QtWidgets.QLabel(self.tab_5)
        self.label_2.setGeometry(QtCore.QRect(830, 40, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_11 = QtWidgets.QLabel(self.tab_5)
        self.label_11.setGeometry(QtCore.QRect(840, 410, 161, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_11.setFont(font)
        self.label_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_11.setTextFormat(QtCore.Qt.AutoText)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.tblEgitim = QtWidgets.QTableWidget(self.tab_5)
        self.tblEgitim.setGeometry(QtCore.QRect(570, 90, 701, 291))
        self.tblEgitim.setObjectName("tblEgitim")
        self.tblEgitim.setColumnCount(0)
        self.tblEgitim.setRowCount(0)
        self.tblTest = QtWidgets.QTableWidget(self.tab_5)
        self.tblTest.setGeometry(QtCore.QRect(570, 460, 701, 291))
        self.tblTest.setObjectName("tblTest")
        self.tblTest.setColumnCount(0)
        self.tblTest.setRowCount(0)
        self.label_21 = QtWidgets.QLabel(self.tab_5)
        self.label_21.setGeometry(QtCore.QRect(200, 20, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_21.setFont(font)
        self.label_21.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_21.setTextFormat(QtCore.Qt.AutoText)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.imAccuracy_2 = QtWidgets.QLabel(self.tab_5)
        self.imAccuracy_2.setGeometry(QtCore.QRect(100, 70, 380, 380))
        self.imAccuracy_2.setText("")
        self.imAccuracy_2.setObjectName("imAccuracy_2")
        self.label_25 = QtWidgets.QLabel(self.tab_5)
        self.label_25.setGeometry(QtCore.QRect(200, 520, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_25.setFont(font)
        self.label_25.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_25.setTextFormat(QtCore.Qt.AutoText)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.imAccuracy_3 = QtWidgets.QLabel(self.tab_5)
        self.imAccuracy_3.setGeometry(QtCore.QRect(100, 570, 380, 380))
        self.imAccuracy_3.setText("")
        self.imAccuracy_3.setObjectName("imAccuracy_3")
        self.label_22 = QtWidgets.QLabel(self.tab_5)
        self.label_22.setGeometry(QtCore.QRect(1480, 20, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_22.setFont(font)
        self.label_22.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_22.setTextFormat(QtCore.Qt.AutoText)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.imAccuracy_4 = QtWidgets.QLabel(self.tab_5)
        self.imAccuracy_4.setGeometry(QtCore.QRect(1380, 570, 380, 380))
        self.imAccuracy_4.setText("")
        self.imAccuracy_4.setObjectName("imAccuracy_4")
        self.label_26 = QtWidgets.QLabel(self.tab_5)
        self.label_26.setGeometry(QtCore.QRect(1480, 520, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_26.setFont(font)
        self.label_26.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_26.setTextFormat(QtCore.Qt.AutoText)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.imAccuracy_5 = QtWidgets.QLabel(self.tab_5)
        self.imAccuracy_5.setGeometry(QtCore.QRect(1380, 70, 380, 380))
        self.imAccuracy_5.setText("")
        self.imAccuracy_5.setObjectName("imAccuracy_5")
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.label_12 = QtWidgets.QLabel(self.tab_6)
        self.label_12.setGeometry(QtCore.QRect(650, 60, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_12.setFont(font)
        self.label_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_12.setTextFormat(QtCore.Qt.AutoText)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.label_13 = QtWidgets.QLabel(self.tab_6)
        self.label_13.setGeometry(QtCore.QRect(800, 380, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_13.setFont(font)
        self.label_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_13.setTextFormat(QtCore.Qt.AutoText)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.label_15 = QtWidgets.QLabel(self.tab_6)
        self.label_15.setGeometry(QtCore.QRect(1000, 210, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_15.setFont(font)
        self.label_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_15.setTextFormat(QtCore.Qt.AutoText)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.imAccuracy = QtWidgets.QLabel(self.tab_6)
        self.imAccuracy.setGeometry(QtCore.QRect(30, 30, 650, 450))
        self.imAccuracy.setText("")
        self.imAccuracy.setObjectName("imAccuracy")
        self.imCM = QtWidgets.QLabel(self.tab_6)
        self.imCM.setGeometry(QtCore.QRect(590, 380, 650, 500))
        self.imCM.setText("")
        self.imCM.setObjectName("imCM")
        self.imLoss = QtWidgets.QLabel(self.tab_6)
        self.imLoss.setGeometry(QtCore.QRect(1190, 30, 650, 450))
        self.imLoss.setText("")
        self.imLoss.setObjectName("imLoss")
        self.tabWidget.addTab(self.tab_6, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Diyabet Tahmini"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Yeni Model Eğit"))
        self.label_3.setText(_translate("MainWindow", "Eğitilecek Mimari"))
        self.cmbMimari.setItemText(0, _translate("MainWindow", "Artificial Neural Network Holdout"))
        self.cmbMimari.setItemText(1, _translate("MainWindow", "Artificial Neural Network K-Fold"))
        self.cmbMimari.setItemText(2, _translate("MainWindow", "K-Nearest Neighbour Holdout"))
        self.cmbMimari.setItemText(3, _translate("MainWindow", "K-Nearest Neighbour K-Fold"))
        self.cmbMimari.setItemText(4, _translate("MainWindow", "Decision Tree Classifier Holdout"))
        self.cmbMimari.setItemText(5, _translate("MainWindow", "Decision Tree Classifier K-Fold"))
        self.btnModelEgit.setText(_translate("MainWindow", "Modeli Eğit ve Kullan"))
        self.label_6.setText(_translate("MainWindow", "Model Tahmini"))
        self.lblTahmin.setText(_translate("MainWindow", "Tahmin"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Modeli Test Et"))
        self.btnModelKullan.setText(_translate("MainWindow", "Modeli Kullan"))
        self.cmbModel.setItemText(0, _translate("MainWindow", "ANN Holdout"))
        self.cmbModel.setItemText(1, _translate("MainWindow", "ANN K-Fold"))
        self.cmbModel.setItemText(2, _translate("MainWindow", "KNN Holdout"))
        self.cmbModel.setItemText(3, _translate("MainWindow", "KNN K-Fold"))
        self.cmbModel.setItemText(4, _translate("MainWindow", "DTC Holdout"))
        self.cmbModel.setItemText(5, _translate("MainWindow", "DTC K-Fold"))
        self.label_4.setText(_translate("MainWindow", "Kullanılacak Model"))
        self.txtHamilelik.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "Hamilelik Sayısı (0-17)"))
        self.label_7.setText(_translate("MainWindow", "Kan Şekeri (0-199)"))
        self.txtKanSekeri.setText(_translate("MainWindow", "0"))
        self.txtKanBasinci.setText(_translate("MainWindow", "0"))
        self.label_8.setText(_translate("MainWindow", "Kan Basıncı (0-122)"))
        self.label_9.setText(_translate("MainWindow", "İnsülin Seviyesi (0-846)"))
        self.txtInsulin.setText(_translate("MainWindow", "0"))
        self.label_10.setText(_translate("MainWindow", "Yaş (21-81)"))
        self.txtYas.setText(_translate("MainWindow", "0"))
        self.label_16.setText(_translate("MainWindow", "Genetik Yatkınlık (0.08-2.42)"))
        self.txtGenetik.setText(_translate("MainWindow", "0"))
        self.txtVucutKitle.setText(_translate("MainWindow", "0"))
        self.label_17.setText(_translate("MainWindow", "Vücut Kitle Endeksi (0-67.1)"))
        self.label_18.setText(_translate("MainWindow", "Deri Kalınlığı (0-99)"))
        self.txtDeri.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Modelleri Test Et ve Yeni Model Eğit"))
        self.label_2.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_11.setText(_translate("MainWindow", "Test Verileri"))
        self.label_21.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_25.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_22.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_26.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Verisetini Görüntüle"))
        self.label_12.setText(_translate("MainWindow", "Accuracy"))
        self.label_13.setText(_translate("MainWindow", "Karışıklık Matrisi"))
        self.label_15.setText(_translate("MainWindow", "Loss"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "Grafikleri Görüntüle"))
