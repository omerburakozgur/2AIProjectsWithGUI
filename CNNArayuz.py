# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CNNArayuz.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1610, 893)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(213, 213, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(149, 149, 255, 150))
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
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
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
        brush = QtGui.QBrush(QtGui.QColor(213, 213, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(149, 149, 255, 150))
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
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
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
        brush = QtGui.QBrush(QtGui.QColor(213, 213, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(149, 149, 255, 150))
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
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 127, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(85, 85, 255, 150))
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
        self.groupBox_5 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_5.setGeometry(QtCore.QRect(20, 0, 1581, 871))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setAutoFillBackground(True)
        self.groupBox_5.setObjectName("groupBox_5")
        self.lblGoruntu = QtWidgets.QLabel(self.groupBox_5)
        self.lblGoruntu.setGeometry(QtCore.QRect(140, 100, 250, 250))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblGoruntu.setFont(font)
        self.lblGoruntu.setAutoFillBackground(True)
        self.lblGoruntu.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblGoruntu.setText("")
        self.lblGoruntu.setTextFormat(QtCore.Qt.AutoText)
        self.lblGoruntu.setObjectName("lblGoruntu")
        self.btnGoruntuSec = QtWidgets.QPushButton(self.groupBox_5)
        self.btnGoruntuSec.setGeometry(QtCore.QRect(150, 370, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnGoruntuSec.setFont(font)
        self.btnGoruntuSec.setObjectName("btnGoruntuSec")
        self.btnModelKullan = QtWidgets.QPushButton(self.groupBox_5)
        self.btnModelKullan.setGeometry(QtCore.QRect(1040, 230, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnModelKullan.setFont(font)
        self.btnModelKullan.setObjectName("btnModelKullan")
        self.label_4 = QtWidgets.QLabel(self.groupBox_5)
        self.label_4.setGeometry(QtCore.QRect(840, 230, 181, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_4.setFont(font)
        self.label_4.setAutoFillBackground(False)
        self.label_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox_5)
        self.label_5.setGeometry(QtCore.QRect(760, 50, 321, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_5.setFont(font)
        self.label_5.setAutoFillBackground(False)
        self.label_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_7 = QtWidgets.QLabel(self.groupBox_5)
        self.label_7.setGeometry(QtCore.QRect(190, 60, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_7.setFont(font)
        self.label_7.setAutoFillBackground(False)
        self.label_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_7.setTextFormat(QtCore.Qt.AutoText)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.cmbModel = QtWidgets.QComboBox(self.groupBox_5)
        self.cmbModel.setGeometry(QtCore.QRect(840, 290, 181, 41))
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
        self.lblPrecision = QtWidgets.QLabel(self.groupBox_5)
        self.lblPrecision.setGeometry(QtCore.QRect(840, 140, 161, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblPrecision.setFont(font)
        self.lblPrecision.setAutoFillBackground(False)
        self.lblPrecision.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblPrecision.setTextFormat(QtCore.Qt.AutoText)
        self.lblPrecision.setAlignment(QtCore.Qt.AlignCenter)
        self.lblPrecision.setObjectName("lblPrecision")
        self.lblSensitivity = QtWidgets.QLabel(self.groupBox_5)
        self.lblSensitivity.setGeometry(QtCore.QRect(690, 140, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblSensitivity.setFont(font)
        self.lblSensitivity.setAutoFillBackground(False)
        self.lblSensitivity.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblSensitivity.setTextFormat(QtCore.Qt.AutoText)
        self.lblSensitivity.setAlignment(QtCore.Qt.AlignCenter)
        self.lblSensitivity.setObjectName("lblSensitivity")
        self.lblF1 = QtWidgets.QLabel(self.groupBox_5)
        self.lblF1.setGeometry(QtCore.QRect(1150, 140, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblF1.setFont(font)
        self.lblF1.setAutoFillBackground(False)
        self.lblF1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblF1.setTextFormat(QtCore.Qt.AutoText)
        self.lblF1.setAlignment(QtCore.Qt.AlignCenter)
        self.lblF1.setObjectName("lblF1")
        self.lblAccuracy = QtWidgets.QLabel(self.groupBox_5)
        self.lblAccuracy.setGeometry(QtCore.QRect(540, 140, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblAccuracy.setFont(font)
        self.lblAccuracy.setAutoFillBackground(False)
        self.lblAccuracy.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblAccuracy.setTextFormat(QtCore.Qt.AutoText)
        self.lblAccuracy.setAlignment(QtCore.Qt.AlignCenter)
        self.lblAccuracy.setObjectName("lblAccuracy")
        self.btnModelEgit = QtWidgets.QPushButton(self.groupBox_5)
        self.btnModelEgit.setGeometry(QtCore.QRect(1040, 290, 231, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.btnModelEgit.setFont(font)
        self.btnModelEgit.setAutoRepeat(True)
        self.btnModelEgit.setObjectName("btnModelEgit")
        self.label_2 = QtWidgets.QLabel(self.groupBox_5)
        self.label_2.setGeometry(QtCore.QRect(220, 440, 91, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setAutoFillBackground(False)
        self.label_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_2.setObjectName("label_2")
        self.imCM = QtWidgets.QLabel(self.groupBox_5)
        self.imCM.setGeometry(QtCore.QRect(1070, 480, 480, 370))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.imCM.setFont(font)
        self.imCM.setAutoFillBackground(True)
        self.imCM.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imCM.setText("")
        self.imCM.setTextFormat(QtCore.Qt.AutoText)
        self.imCM.setObjectName("imCM")
        self.imAccuracy = QtWidgets.QLabel(self.groupBox_5)
        self.imAccuracy.setGeometry(QtCore.QRect(30, 480, 480, 370))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.imAccuracy.setFont(font)
        self.imAccuracy.setAutoFillBackground(True)
        self.imAccuracy.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imAccuracy.setText("")
        self.imAccuracy.setTextFormat(QtCore.Qt.AutoText)
        self.imAccuracy.setObjectName("imAccuracy")
        self.label_8 = QtWidgets.QLabel(self.groupBox_5)
        self.label_8.setGeometry(QtCore.QRect(760, 440, 51, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_8.setFont(font)
        self.label_8.setAutoFillBackground(False)
        self.label_8.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_8.setTextFormat(QtCore.Qt.AutoText)
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.groupBox_5)
        self.label_9.setGeometry(QtCore.QRect(1230, 440, 151, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_9.setFont(font)
        self.label_9.setAutoFillBackground(False)
        self.label_9.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_9.setTextFormat(QtCore.Qt.AutoText)
        self.label_9.setObjectName("label_9")
        self.imLoss = QtWidgets.QLabel(self.groupBox_5)
        self.imLoss.setGeometry(QtCore.QRect(550, 480, 480, 370))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.imLoss.setFont(font)
        self.imLoss.setAutoFillBackground(True)
        self.imLoss.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.imLoss.setText("")
        self.imLoss.setTextFormat(QtCore.Qt.AutoText)
        self.imLoss.setObjectName("imLoss")
        self.lblSpecificity = QtWidgets.QLabel(self.groupBox_5)
        self.lblSpecificity.setGeometry(QtCore.QRect(1000, 140, 151, 61))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.lblSpecificity.setFont(font)
        self.lblSpecificity.setAutoFillBackground(False)
        self.lblSpecificity.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.lblSpecificity.setTextFormat(QtCore.Qt.AutoText)
        self.lblSpecificity.setAlignment(QtCore.Qt.AlignCenter)
        self.lblSpecificity.setObjectName("lblSpecificity")
        self.label_6 = QtWidgets.QLabel(self.groupBox_5)
        self.label_6.setGeometry(QtCore.QRect(570, 230, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_6.setFont(font)
        self.label_6.setAutoFillBackground(False)
        self.label_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_6.setTextFormat(QtCore.Qt.AutoText)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.label_10 = QtWidgets.QLabel(self.groupBox_5)
        self.label_10.setGeometry(QtCore.QRect(700, 230, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_10.setFont(font)
        self.label_10.setAutoFillBackground(False)
        self.label_10.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.label_10.setTextFormat(QtCore.Qt.AutoText)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.lblError = QtWidgets.QLabel(self.groupBox_5)
        self.lblError.setGeometry(QtCore.QRect(680, 350, 501, 41))
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
        self.txtBatch = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtBatch.setGeometry(QtCore.QRect(570, 290, 111, 41))
        self.txtBatch.setObjectName("txtBatch")
        self.txtEpoch = QtWidgets.QLineEdit(self.groupBox_5)
        self.txtEpoch.setGeometry(QtCore.QRect(700, 290, 121, 41))
        self.txtEpoch.setObjectName("txtEpoch")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Orman Yangını Tespiti"))
        self.groupBox_5.setTitle(_translate("MainWindow", "Modeli Test Et"))
        self.btnGoruntuSec.setText(_translate("MainWindow", "Görüntü Seç"))
        self.btnModelKullan.setText(_translate("MainWindow", "Modeli Test Et"))
        self.label_4.setText(_translate("MainWindow", "Kullanılacak Model"))
        self.label_5.setText(_translate("MainWindow", "Sonuç:"))
        self.label_7.setText(_translate("MainWindow", "Seçilen Görüntü"))
        self.cmbModel.setItemText(0, _translate("MainWindow", "MLP Holdout"))
        self.cmbModel.setItemText(1, _translate("MainWindow", "MLP K-Fold"))
        self.cmbModel.setItemText(2, _translate("MainWindow", "ResNet50 Holdout"))
        self.cmbModel.setItemText(3, _translate("MainWindow", "ResNet50 K-Fold"))
        self.cmbModel.setItemText(4, _translate("MainWindow", "VGG-19 Holdout"))
        self.cmbModel.setItemText(5, _translate("MainWindow", "VGG-19 K-Fold"))
        self.lblPrecision.setText(_translate("MainWindow", "Precision:"))
        self.lblSensitivity.setText(_translate("MainWindow", "Sensitivity: "))
        self.lblF1.setText(_translate("MainWindow", "F1 Score:"))
        self.lblAccuracy.setText(_translate("MainWindow", "Accuracy:"))
        self.btnModelEgit.setText(_translate("MainWindow", "Modeli Eğit ve Test Et"))
        self.label_2.setText(_translate("MainWindow", "Accuracy"))
        self.label_8.setText(_translate("MainWindow", "Loss"))
        self.label_9.setText(_translate("MainWindow", "Confusion Matrix"))
        self.lblSpecificity.setText(_translate("MainWindow", "Specificity:"))
        self.label_6.setText(_translate("MainWindow", "Batch Size"))
        self.label_10.setText(_translate("MainWindow", "Epoch Sayısı"))
        self.lblError.setText(_translate("MainWindow", "..."))
        self.txtBatch.setText(_translate("MainWindow", "2"))
        self.txtEpoch.setText(_translate("MainWindow", "50"))
