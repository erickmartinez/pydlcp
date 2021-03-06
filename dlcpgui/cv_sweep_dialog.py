# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\cv_sweep_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(572, 301)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Dialog.sizePolicy().hasHeightForWidth())
        Dialog.setSizePolicy(sizePolicy)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Dialog)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_Acquisition = QtWidgets.QGroupBox(Dialog)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_Acquisition.sizePolicy().hasHeightForWidth())
        self.groupBox_Acquisition.setSizePolicy(sizePolicy)
        self.groupBox_Acquisition.setObjectName("groupBox_Acquisition")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_Acquisition)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.formLayout_Acquisition = QtWidgets.QFormLayout()
        self.formLayout_Acquisition.setObjectName("formLayout_Acquisition")
        self.label_Frequency = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_Frequency.setObjectName("label_Frequency")
        self.formLayout_Acquisition.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_Frequency)
        self.comboBox_Frequency = QtWidgets.QComboBox(self.groupBox_Acquisition)
        self.comboBox_Frequency.setObjectName("comboBox_Frequency")
        self.comboBox_Frequency.addItem("")
        self.comboBox_Frequency.addItem("")
        self.comboBox_Frequency.addItem("")
        self.comboBox_Frequency.addItem("")
        self.comboBox_Frequency.addItem("")
        self.formLayout_Acquisition.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.comboBox_Frequency)
        self.label_IntegrationTime = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_IntegrationTime.setObjectName("label_IntegrationTime")
        self.formLayout_Acquisition.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_IntegrationTime)
        self.comboBox_IntegrationTime = QtWidgets.QComboBox(self.groupBox_Acquisition)
        self.comboBox_IntegrationTime.setObjectName("comboBox_IntegrationTime")
        self.comboBox_IntegrationTime.addItem("")
        self.comboBox_IntegrationTime.addItem("")
        self.comboBox_IntegrationTime.addItem("")
        self.formLayout_Acquisition.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_IntegrationTime)
        self.label_NumberOfAverages = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_NumberOfAverages.setObjectName("label_NumberOfAverages")
        self.formLayout_Acquisition.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_NumberOfAverages)
        self.comboBox_NumberOfAverages = QtWidgets.QComboBox(self.groupBox_Acquisition)
        self.comboBox_NumberOfAverages.setObjectName("comboBox_NumberOfAverages")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.formLayout_Acquisition.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_NumberOfAverages)
        self.label_OscLevel = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_OscLevel.setObjectName("label_OscLevel")
        self.formLayout_Acquisition.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_OscLevel)
        self.doubleSpinBox_OscLevel = QtWidgets.QDoubleSpinBox(self.groupBox_Acquisition)
        self.doubleSpinBox_OscLevel.setDecimals(0)
        self.doubleSpinBox_OscLevel.setMinimum(10.0)
        self.doubleSpinBox_OscLevel.setMaximum(1000.0)
        self.doubleSpinBox_OscLevel.setSingleStep(1.0)
        self.doubleSpinBox_OscLevel.setProperty("value", 50.0)
        self.doubleSpinBox_OscLevel.setObjectName("doubleSpinBox_OscLevel")
        self.formLayout_Acquisition.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_OscLevel)
        self.label_BiasStart = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_BiasStart.setObjectName("label_BiasStart")
        self.formLayout_Acquisition.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_BiasStart)
        self.doubleSpinBox_BiasStart = QtWidgets.QDoubleSpinBox(self.groupBox_Acquisition)
        self.doubleSpinBox_BiasStart.setMinimum(-40.0)
        self.doubleSpinBox_BiasStart.setMaximum(0.0)
        self.doubleSpinBox_BiasStart.setProperty("value", -5.0)
        self.doubleSpinBox_BiasStart.setObjectName("doubleSpinBox_BiasStart")
        self.formLayout_Acquisition.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_BiasStart)
        self.label_SweepDirection = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_SweepDirection.setObjectName("label_SweepDirection")
        self.formLayout_Acquisition.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_SweepDirection)
        self.comboBox_SweepDirection = QtWidgets.QComboBox(self.groupBox_Acquisition)
        self.comboBox_SweepDirection.setObjectName("comboBox_SweepDirection")
        self.comboBox_SweepDirection.addItem("")
        self.comboBox_SweepDirection.addItem("")
        self.formLayout_Acquisition.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.comboBox_SweepDirection)
        self.label_BiasStep = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_BiasStep.setObjectName("label_BiasStep")
        self.formLayout_Acquisition.setWidget(6, QtWidgets.QFormLayout.LabelRole, self.label_BiasStep)
        self.doubleSpinBox_BiasStep = QtWidgets.QDoubleSpinBox(self.groupBox_Acquisition)
        self.doubleSpinBox_BiasStep.setMinimum(0.01)
        self.doubleSpinBox_BiasStep.setMaximum(10.0)
        self.doubleSpinBox_BiasStep.setProperty("value", 0.1)
        self.doubleSpinBox_BiasStep.setObjectName("doubleSpinBox_BiasStep")
        self.formLayout_Acquisition.setWidget(6, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_BiasStep)
        self.label_BiasStop = QtWidgets.QLabel(self.groupBox_Acquisition)
        self.label_BiasStop.setObjectName("label_BiasStop")
        self.formLayout_Acquisition.setWidget(7, QtWidgets.QFormLayout.LabelRole, self.label_BiasStop)
        self.doubleSpinBox_BiasStop = QtWidgets.QDoubleSpinBox(self.groupBox_Acquisition)
        self.doubleSpinBox_BiasStop.setMinimum(0.0)
        self.doubleSpinBox_BiasStop.setMaximum(40.0)
        self.doubleSpinBox_BiasStop.setObjectName("doubleSpinBox_BiasStop")
        self.formLayout_Acquisition.setWidget(7, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_BiasStop)
        self.verticalLayout_3.addLayout(self.formLayout_Acquisition)
        self.pushButton_Start = QtWidgets.QPushButton(self.groupBox_Acquisition)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/start.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Start.setIcon(icon)
        self.pushButton_Start.setObjectName("pushButton_Start")
        self.verticalLayout_3.addWidget(self.pushButton_Start)
        self.verticalLayout.addWidget(self.groupBox_Acquisition)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.groupBox_Graph = QtWidgets.QGroupBox(Dialog)
        self.groupBox_Graph.setObjectName("groupBox_Graph")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_Graph)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.widget_Plot = QtWidgets.QWidget(self.groupBox_Graph)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_Plot.sizePolicy().hasHeightForWidth())
        self.widget_Plot.setSizePolicy(sizePolicy)
        self.widget_Plot.setMinimumSize(QtCore.QSize(300, 250))
        self.widget_Plot.setObjectName("widget_Plot")
        self.verticalLayout_2.addWidget(self.widget_Plot)
        self.horizontalLayout.addWidget(self.groupBox_Graph)
        self.label_Frequency.setBuddy(self.comboBox_Frequency)
        self.label_IntegrationTime.setBuddy(self.comboBox_IntegrationTime)
        self.label_NumberOfAverages.setBuddy(self.comboBox_NumberOfAverages)
        self.label_OscLevel.setBuddy(self.doubleSpinBox_OscLevel)
        self.label_BiasStart.setBuddy(self.doubleSpinBox_BiasStart)
        self.label_SweepDirection.setBuddy(self.comboBox_SweepDirection)
        self.label_BiasStep.setBuddy(self.doubleSpinBox_BiasStep)
        self.label_BiasStop.setBuddy(self.doubleSpinBox_BiasStop)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.comboBox_Frequency, self.comboBox_IntegrationTime)
        Dialog.setTabOrder(self.comboBox_IntegrationTime, self.comboBox_NumberOfAverages)
        Dialog.setTabOrder(self.comboBox_NumberOfAverages, self.doubleSpinBox_OscLevel)
        Dialog.setTabOrder(self.doubleSpinBox_OscLevel, self.comboBox_SweepDirection)
        Dialog.setTabOrder(self.comboBox_SweepDirection, self.doubleSpinBox_BiasStart)
        Dialog.setTabOrder(self.doubleSpinBox_BiasStart, self.doubleSpinBox_BiasStep)
        Dialog.setTabOrder(self.doubleSpinBox_BiasStep, self.doubleSpinBox_BiasStop)
        Dialog.setTabOrder(self.doubleSpinBox_BiasStop, self.pushButton_Start)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Run Test"))
        self.groupBox_Acquisition.setTitle(_translate("Dialog", "Acquisition Parameters"))
        self.label_Frequency.setText(_translate("Dialog", "Frequency:"))
        self.comboBox_Frequency.setItemText(0, _translate("Dialog", "1 MHz"))
        self.comboBox_Frequency.setItemText(1, _translate("Dialog", "100 kHz"))
        self.comboBox_Frequency.setItemText(2, _translate("Dialog", "10 kHz"))
        self.comboBox_Frequency.setItemText(3, _translate("Dialog", "1 kHz"))
        self.comboBox_Frequency.setItemText(4, _translate("Dialog", "100 Hz"))
        self.label_IntegrationTime.setText(_translate("Dialog", "Integration Time:"))
        self.comboBox_IntegrationTime.setItemText(0, _translate("Dialog", "Short (500 us)"))
        self.comboBox_IntegrationTime.setItemText(1, _translate("Dialog", "Medium (5 ms)"))
        self.comboBox_IntegrationTime.setItemText(2, _translate("Dialog", "Long (100 ms)"))
        self.label_NumberOfAverages.setText(_translate("Dialog", "Number of Averages:"))
        self.comboBox_NumberOfAverages.setItemText(0, _translate("Dialog", "1"))
        self.comboBox_NumberOfAverages.setItemText(1, _translate("Dialog", "2"))
        self.comboBox_NumberOfAverages.setItemText(2, _translate("Dialog", "4"))
        self.comboBox_NumberOfAverages.setItemText(3, _translate("Dialog", "8"))
        self.comboBox_NumberOfAverages.setItemText(4, _translate("Dialog", "16"))
        self.comboBox_NumberOfAverages.setItemText(5, _translate("Dialog", "32"))
        self.comboBox_NumberOfAverages.setItemText(6, _translate("Dialog", "64"))
        self.comboBox_NumberOfAverages.setItemText(7, _translate("Dialog", "128"))
        self.label_OscLevel.setText(_translate("Dialog", "Osc Level:"))
        self.doubleSpinBox_OscLevel.setSuffix(_translate("Dialog", " (mV p-p)"))
        self.label_BiasStart.setText(_translate("Dialog", "Bias Start:"))
        self.doubleSpinBox_BiasStart.setSuffix(_translate("Dialog", " (V)"))
        self.label_SweepDirection.setText(_translate("Dialog", "Sweep Direction"))
        self.comboBox_SweepDirection.setItemText(0, _translate("Dialog", "Down"))
        self.comboBox_SweepDirection.setItemText(1, _translate("Dialog", "Up"))
        self.label_BiasStep.setText(_translate("Dialog", "Bias Step:"))
        self.doubleSpinBox_BiasStep.setSuffix(_translate("Dialog", " (V)"))
        self.label_BiasStop.setText(_translate("Dialog", "Bias Stop:"))
        self.doubleSpinBox_BiasStop.setSuffix(_translate("Dialog", " (V)"))
        self.pushButton_Start.setText(_translate("Dialog", "Acquire"))
        self.groupBox_Graph.setTitle(_translate("Dialog", "CV Plot"))

import cv_sweep_dialog_rc
