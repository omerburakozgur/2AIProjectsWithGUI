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
        MainWindow.resize(1833, 1080)
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
        self.tabWidget.setGeometry(QtCore.QRect(50, 10, 1811, 1041))
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
        self.groupBox_3.setGeometry(QtCore.QRect(10, 10, 1791, 1031))
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
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setGeometry(QtCore.QRect(720, 20, 321, 51))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_14.setFont(font)
        self.label_14.setAutoFillBackground(False)
        self.label_14.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_14.setTextFormat(QtCore.Qt.AutoText)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.lblSensitivity = QtWidgets.QLabel(self.groupBox_3)
        self.lblSensitivity.setGeometry(QtCore.QRect(660, 70, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblSensitivity.setFont(font)
        self.lblSensitivity.setAutoFillBackground(False)
        self.lblSensitivity.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblSensitivity.setTextFormat(QtCore.Qt.AutoText)
        self.lblSensitivity.setAlignment(QtCore.Qt.AlignCenter)
        self.lblSensitivity.setObjectName("lblSensitivity")
        self.txtEpoch = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtEpoch.setGeometry(QtCore.QRect(340, 270, 121, 41))
        self.txtEpoch.setObjectName("txtEpoch")
        self.label_19 = QtWidgets.QLabel(self.groupBox_3)
        self.label_19.setGeometry(QtCore.QRect(340, 210, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_19.setFont(font)
        self.label_19.setAutoFillBackground(False)
        self.label_19.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_19.setTextFormat(QtCore.Qt.AutoText)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.label_20 = QtWidgets.QLabel(self.groupBox_3)
        self.label_20.setGeometry(QtCore.QRect(210, 210, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_20.setFont(font)
        self.label_20.setAutoFillBackground(False)
        self.label_20.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_20.setTextFormat(QtCore.Qt.AutoText)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.lblSpecificity = QtWidgets.QLabel(self.groupBox_3)
        self.lblSpecificity.setGeometry(QtCore.QRect(970, 70, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblSpecificity.setFont(font)
        self.lblSpecificity.setAutoFillBackground(False)
        self.lblSpecificity.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblSpecificity.setTextFormat(QtCore.Qt.AutoText)
        self.lblSpecificity.setAlignment(QtCore.Qt.AlignCenter)
        self.lblSpecificity.setObjectName("lblSpecificity")
        self.lblPrecision = QtWidgets.QLabel(self.groupBox_3)
        self.lblPrecision.setGeometry(QtCore.QRect(810, 70, 161, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblPrecision.setFont(font)
        self.lblPrecision.setAutoFillBackground(False)
        self.lblPrecision.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblPrecision.setTextFormat(QtCore.Qt.AutoText)
        self.lblPrecision.setAlignment(QtCore.Qt.AlignCenter)
        self.lblPrecision.setObjectName("lblPrecision")
        self.txtBatch = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtBatch.setGeometry(QtCore.QRect(210, 270, 111, 41))
        self.txtBatch.setObjectName("txtBatch")
        self.lblAccuracy = QtWidgets.QLabel(self.groupBox_3)
        self.lblAccuracy.setGeometry(QtCore.QRect(510, 70, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblAccuracy.setFont(font)
        self.lblAccuracy.setAutoFillBackground(False)
        self.lblAccuracy.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblAccuracy.setTextFormat(QtCore.Qt.AutoText)
        self.lblAccuracy.setAlignment(QtCore.Qt.AlignCenter)
        self.lblAccuracy.setObjectName("lblAccuracy")
        self.lblF1 = QtWidgets.QLabel(self.groupBox_3)
        self.lblF1.setGeometry(QtCore.QRect(1120, 70, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblF1.setFont(font)
        self.lblF1.setAutoFillBackground(False)
        self.lblF1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblF1.setTextFormat(QtCore.Qt.AutoText)
        self.lblF1.setAlignment(QtCore.Qt.AlignCenter)
        self.lblF1.setObjectName("lblF1")
        self.lblError = QtWidgets.QLabel(self.groupBox_3)
        self.lblError.setGeometry(QtCore.QRect(640, 130, 501, 41))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        self.lblError.setPalette(palette)
        font = QtGui.QFont()
        font.setItalic(True)
        self.lblError.setFont(font)
        self.lblError.setAutoFillBackground(False)
        self.lblError.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblError.setTextFormat(QtCore.Qt.AutoText)
        self.lblError.setAlignment(QtCore.Qt.AlignCenter)
        self.lblError.setObjectName("lblError")
        self.cmbMimari = QtWidgets.QComboBox(self.groupBox_3)
        self.cmbMimari.setGeometry(QtCore.QRect(480, 270, 231, 41))
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
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setGeometry(QtCore.QRect(480, 210, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.cmbModel = QtWidgets.QComboBox(self.groupBox_3)
        self.cmbModel.setGeometry(QtCore.QRect(10, 50, 171, 41))
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
        self.btnModelEgit = QtWidgets.QPushButton(self.groupBox_3)
        self.btnModelEgit.setGeometry(QtCore.QRect(250, 320, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnModelEgit.setFont(font)
        self.btnModelEgit.setObjectName("btnModelEgit")
        self.btnModelKullan = QtWidgets.QPushButton(self.groupBox_3)
        self.btnModelKullan.setGeometry(QtCore.QRect(480, 320, 221, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnModelKullan.setFont(font)
        self.btnModelKullan.setObjectName("btnModelKullan")
        self.txtYas = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtYas.setGeometry(QtCore.QRect(1530, 340, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtYas.setFont(font)
        self.txtYas.setObjectName("txtYas")
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(850, 190, 191, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_5.setFont(font)
        self.label_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.txtKanSekeri = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtKanSekeri.setGeometry(QtCore.QRect(1110, 240, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtKanSekeri.setFont(font)
        self.txtKanSekeri.setObjectName("txtKanSekeri")
        self.txtKanBasinci = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtKanBasinci.setGeometry(QtCore.QRect(1280, 240, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtKanBasinci.setFont(font)
        self.txtKanBasinci.setObjectName("txtKanBasinci")
        self.txtHamilelik = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtHamilelik.setGeometry(QtCore.QRect(920, 240, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtHamilelik.setFont(font)
        self.txtHamilelik.setObjectName("txtHamilelik")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(1500, 290, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_10.setFont(font)
        self.label_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_10.setTextFormat(QtCore.Qt.AutoText)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.txtInsulin = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtInsulin.setGeometry(QtCore.QRect(860, 340, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtInsulin.setFont(font)
        self.txtInsulin.setObjectName("txtInsulin")
        self.label_17 = QtWidgets.QLabel(self.groupBox_3)
        self.label_17.setGeometry(QtCore.QRect(990, 290, 241, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_17.setFont(font)
        self.label_17.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_17.setTextFormat(QtCore.Qt.AutoText)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(1220, 190, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_8.setFont(font)
        self.label_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_8.setTextFormat(QtCore.Qt.AutoText)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.label_18 = QtWidgets.QLabel(self.groupBox_3)
        self.label_18.setGeometry(QtCore.QRect(1400, 190, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_18.setFont(font)
        self.label_18.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_18.setTextFormat(QtCore.Qt.AutoText)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(1050, 190, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_7.setTextFormat(QtCore.Qt.AutoText)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_16 = QtWidgets.QLabel(self.groupBox_3)
        self.label_16.setGeometry(QtCore.QRect(1240, 290, 251, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_16.setFont(font)
        self.label_16.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_16.setTextFormat(QtCore.Qt.AutoText)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.txtDeri = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtDeri.setGeometry(QtCore.QRect(1460, 240, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtDeri.setFont(font)
        self.txtDeri.setObjectName("txtDeri")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(780, 290, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_9.setFont(font)
        self.label_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_9.setTextFormat(QtCore.Qt.AutoText)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.txtVucutKitle = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtVucutKitle.setGeometry(QtCore.QRect(1090, 340, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtVucutKitle.setFont(font)
        self.txtVucutKitle.setObjectName("txtVucutKitle")
        self.txtGenetik = QtWidgets.QLineEdit(self.groupBox_3)
        self.txtGenetik.setGeometry(QtCore.QRect(1330, 340, 51, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.txtGenetik.setFont(font)
        self.txtGenetik.setObjectName("txtGenetik")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(770, 390, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_13.setFont(font)
        self.label_13.setAutoFillBackground(True)
        self.label_13.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_13.setTextFormat(QtCore.Qt.AutoText)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.imCM = QtWidgets.QLabel(self.groupBox_3)
        self.imCM.setGeometry(QtCore.QRect(570, 450, 620, 460))
        self.imCM.setAutoFillBackground(True)
        self.imCM.setText("")
        self.imCM.setObjectName("imCM")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setGeometry(QtCore.QRect(1400, 450, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_15.setFont(font)
        self.label_15.setAutoFillBackground(True)
        self.label_15.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_15.setTextFormat(QtCore.Qt.AutoText)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.imAccuracy = QtWidgets.QLabel(self.groupBox_3)
        self.imAccuracy.setGeometry(QtCore.QRect(30, 510, 480, 360))
        self.imAccuracy.setAutoFillBackground(True)
        self.imAccuracy.setText("")
        self.imAccuracy.setObjectName("imAccuracy")
        self.imLoss = QtWidgets.QLabel(self.groupBox_3)
        self.imLoss.setGeometry(QtCore.QRect(1270, 510, 480, 360))
        self.imLoss.setAutoFillBackground(True)
        self.imLoss.setText("")
        self.imLoss.setObjectName("imLoss")
        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(150, 450, 221, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_12.setFont(font)
        self.label_12.setAutoFillBackground(True)
        self.label_12.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_12.setTextFormat(QtCore.Qt.AutoText)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.groupBox_2 = QtWidgets.QGroupBox(self.tab_5)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 10, 1771, 971))
        self.groupBox_2.setAutoFillBackground(True)
        self.groupBox_2.setObjectName("groupBox_2")
        self.imAccuracy_5 = QtWidgets.QLabel(self.groupBox_2)
        self.imAccuracy_5.setGeometry(QtCore.QRect(1260, 70, 480, 360))
        self.imAccuracy_5.setAutoFillBackground(True)
        self.imAccuracy_5.setText("")
        self.imAccuracy_5.setObjectName("imAccuracy_5")
        self.label_26 = QtWidgets.QLabel(self.groupBox_2)
        self.label_26.setGeometry(QtCore.QRect(1410, 470, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_26.setFont(font)
        self.label_26.setAutoFillBackground(True)
        self.label_26.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_26.setTextFormat(QtCore.Qt.AutoText)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setGeometry(QtCore.QRect(790, 20, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_2.setFont(font)
        self.label_2.setAutoFillBackground(True)
        self.label_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.tblTest = QtWidgets.QTableWidget(self.groupBox_2)
        self.tblTest.setGeometry(QtCore.QRect(530, 530, 701, 341))
        self.tblTest.setAutoFillBackground(True)
        self.tblTest.setObjectName("tblTest")
        self.tblTest.setColumnCount(0)
        self.tblTest.setRowCount(0)
        self.imAccuracy_3 = QtWidgets.QLabel(self.groupBox_2)
        self.imAccuracy_3.setGeometry(QtCore.QRect(20, 520, 480, 359))
        self.imAccuracy_3.setAutoFillBackground(True)
        self.imAccuracy_3.setText("")
        self.imAccuracy_3.setObjectName("imAccuracy_3")
        self.tblEgitim = QtWidgets.QTableWidget(self.groupBox_2)
        self.tblEgitim.setGeometry(QtCore.QRect(530, 80, 701, 341))
        self.tblEgitim.setAutoFillBackground(True)
        self.tblEgitim.setObjectName("tblEgitim")
        self.tblEgitim.setColumnCount(0)
        self.tblEgitim.setRowCount(0)
        self.label_22 = QtWidgets.QLabel(self.groupBox_2)
        self.label_22.setGeometry(QtCore.QRect(1410, 20, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_22.setFont(font)
        self.label_22.setAutoFillBackground(True)
        self.label_22.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_22.setTextFormat(QtCore.Qt.AutoText)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.imAccuracy_4 = QtWidgets.QLabel(self.groupBox_2)
        self.imAccuracy_4.setGeometry(QtCore.QRect(1260, 520, 480, 360))
        self.imAccuracy_4.setAutoFillBackground(True)
        self.imAccuracy_4.setText("")
        self.imAccuracy_4.setObjectName("imAccuracy_4")
        self.label_25 = QtWidgets.QLabel(self.groupBox_2)
        self.label_25.setGeometry(QtCore.QRect(160, 470, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_25.setFont(font)
        self.label_25.setAutoFillBackground(True)
        self.label_25.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_25.setTextFormat(QtCore.Qt.AutoText)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.imAccuracy_2 = QtWidgets.QLabel(self.groupBox_2)
        self.imAccuracy_2.setGeometry(QtCore.QRect(20, 70, 480, 360))
        self.imAccuracy_2.setAutoFillBackground(True)
        self.imAccuracy_2.setText("")
        self.imAccuracy_2.setObjectName("imAccuracy_2")
        self.label_11 = QtWidgets.QLabel(self.groupBox_2)
        self.label_11.setGeometry(QtCore.QRect(800, 470, 161, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_11.setFont(font)
        self.label_11.setAutoFillBackground(True)
        self.label_11.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_11.setTextFormat(QtCore.Qt.AutoText)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.label_21 = QtWidgets.QLabel(self.groupBox_2)
        self.label_21.setGeometry(QtCore.QRect(170, 20, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label_21.setFont(font)
        self.label_21.setAutoFillBackground(True)
        self.label_21.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_21.setTextFormat(QtCore.Qt.AutoText)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.tabWidget.addTab(self.tab_5, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Diyabet Tahmini"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Yeni Model Eğit"))
        self.label_14.setText(_translate("MainWindow", "Sonuç:"))
        self.lblSensitivity.setText(_translate("MainWindow", "Sensitivity: "))
        self.txtEpoch.setText(_translate("MainWindow", "50"))
        self.label_19.setText(_translate("MainWindow", "Epoch Sayısı"))
        self.label_20.setText(_translate("MainWindow", "Batch Size"))
        self.lblSpecificity.setText(_translate("MainWindow", "Specificity:"))
        self.lblPrecision.setText(_translate("MainWindow", "Precision:"))
        self.txtBatch.setText(_translate("MainWindow", "2"))
        self.lblAccuracy.setText(_translate("MainWindow", "Accuracy:"))
        self.lblF1.setText(_translate("MainWindow", "F1 Score:"))
        self.lblError.setText(_translate("MainWindow", "..."))
        self.cmbMimari.setItemText(0, _translate("MainWindow", "MLP Holdout"))
        self.cmbMimari.setItemText(1, _translate("MainWindow", "MLP K-Fold"))
        self.cmbMimari.setItemText(2, _translate("MainWindow", "KNN Holdout"))
        self.cmbMimari.setItemText(3, _translate("MainWindow", "KNN K-Fold"))
        self.cmbMimari.setItemText(4, _translate("MainWindow", "DTC Holdout"))
        self.cmbMimari.setItemText(5, _translate("MainWindow", "DTC K-Fold"))
        self.label_4.setText(_translate("MainWindow", "Kullanılacak Model"))
        self.cmbModel.setItemText(0, _translate("MainWindow", "ANN Holdout"))
        self.cmbModel.setItemText(1, _translate("MainWindow", "ANN K-Fold"))
        self.cmbModel.setItemText(2, _translate("MainWindow", "KNN Holdout"))
        self.cmbModel.setItemText(3, _translate("MainWindow", "KNN K-Fold"))
        self.cmbModel.setItemText(4, _translate("MainWindow", "DTC Holdout"))
        self.cmbModel.setItemText(5, _translate("MainWindow", "DTC K-Fold"))
        self.btnModelEgit.setText(_translate("MainWindow", "Modeli Eğit ve Kullan"))
        self.btnModelKullan.setText(_translate("MainWindow", "Modeli Kullan"))
        self.txtYas.setText(_translate("MainWindow", "0"))
        self.label_5.setText(_translate("MainWindow", "Hamilelik Sayısı (0-17)"))
        self.txtKanSekeri.setText(_translate("MainWindow", "0"))
        self.txtKanBasinci.setText(_translate("MainWindow", "0"))
        self.txtHamilelik.setText(_translate("MainWindow", "0"))
        self.label_10.setText(_translate("MainWindow", "Yaş (21-81)"))
        self.txtInsulin.setText(_translate("MainWindow", "0"))
        self.label_17.setText(_translate("MainWindow", "Vücut Kitle Endeksi (0-67.1)"))
        self.label_8.setText(_translate("MainWindow", "Kan Basıncı (0-122)"))
        self.label_18.setText(_translate("MainWindow", "Deri Kalınlığı (0-99)"))
        self.label_7.setText(_translate("MainWindow", "Kan Şekeri (0-199)"))
        self.label_16.setText(_translate("MainWindow", "Genetik Yatkınlık (0.08-2.42)"))
        self.txtDeri.setText(_translate("MainWindow", "0"))
        self.label_9.setText(_translate("MainWindow", "İnsülin Seviyesi (0-846)"))
        self.txtVucutKitle.setText(_translate("MainWindow", "0"))
        self.txtGenetik.setText(_translate("MainWindow", "0"))
        self.label_13.setText(_translate("MainWindow", "Karışıklık Matrisi"))
        self.label_15.setText(_translate("MainWindow", "Loss"))
        self.label_12.setText(_translate("MainWindow", "Accuracy"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Modelleri Test Et ve Yeni Model Eğit"))
        self.groupBox_2.setTitle(_translate("MainWindow", "GroupBox"))
        self.label_26.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_2.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_22.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_25.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.label_11.setText(_translate("MainWindow", "Test Verileri"))
        self.label_21.setText(_translate("MainWindow", "Eğitim Verileri"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Verisetini Görüntüle"))