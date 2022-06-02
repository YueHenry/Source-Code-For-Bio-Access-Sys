# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'IrisDlg.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_IrisDlg(object):
    def setupUi(self, IrisDlg):
        IrisDlg.setObjectName("IrisDlg")
        IrisDlg.resize(1021, 773)
        self.img_view = QtWidgets.QGraphicsView(IrisDlg)
        self.img_view.setGeometry(QtCore.QRect(10, 40, 811, 691))
        self.img_view.setObjectName("img_view")
        self.layoutWidget = QtWidgets.QWidget(IrisDlg)
        self.layoutWidget.setGeometry(QtCore.QRect(23, 0, 801, 41))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_file_name = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(18)
        self.label_file_name.setFont(font)
        self.label_file_name.setObjectName("label_file_name")
        self.horizontalLayout.addWidget(self.label_file_name)
        self.label_result = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(18)
        self.label_result.setFont(font)
        self.label_result.setObjectName("label_result")
        self.horizontalLayout.addWidget(self.label_result)
        self.layoutWidget1 = QtWidgets.QWidget(IrisDlg)
        self.layoutWidget1.setGeometry(QtCore.QRect(837, 30, 161, 271))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.radioBtn_resnet = QtWidgets.QRadioButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.radioBtn_resnet.setFont(font)
        self.radioBtn_resnet.setChecked(False)
        self.radioBtn_resnet.setObjectName("radioBtn_resnet")
        self.gridLayout.addWidget(self.radioBtn_resnet, 0, 0, 1, 1)
        self.radioBtn_transformer = QtWidgets.QRadioButton(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.radioBtn_transformer.setFont(font)
        self.radioBtn_transformer.setChecked(True)
        self.radioBtn_transformer.setObjectName("radioBtn_transformer")
        self.gridLayout.addWidget(self.radioBtn_transformer, 1, 0, 1, 1)
        self.btn_load = QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_load.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.btn_load.setFont(font)
        self.btn_load.setObjectName("btn_load")
        self.gridLayout.addWidget(self.btn_load, 2, 0, 1, 1)
        self.btn_import = QtWidgets.QPushButton(self.layoutWidget1)
        self.btn_import.setMinimumSize(QtCore.QSize(0, 50))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(16)
        self.btn_import.setFont(font)
        self.btn_import.setObjectName("btn_import")
        self.gridLayout.addWidget(self.btn_import, 3, 0, 1, 1)

        self.retranslateUi(IrisDlg)
        QtCore.QMetaObject.connectSlotsByName(IrisDlg)

    def retranslateUi(self, IrisDlg):
        _translate = QtCore.QCoreApplication.translate
        IrisDlg.setWindowTitle(_translate("IrisDlg", "Iris-Recognition"))
        self.label_file_name.setText(_translate("IrisDlg", "File Name:"))
        self.label_result.setText(_translate("IrisDlg", "Result："))
        self.radioBtn_resnet.setText(_translate("IrisDlg", "Resnet 34"))
        self.radioBtn_transformer.setText(_translate("IrisDlg", "Transformer"))
        self.btn_load.setText(_translate("IrisDlg", "Load Model"))
        self.btn_import.setText(_translate("IrisDlg", "Import Image"))

