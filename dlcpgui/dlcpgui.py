# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dlcpgui.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(840, 566)
        MainWindow.setMinimumSize(QtCore.QSize(840, 0))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/images/scope.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setStyleSheet("")
        MainWindow.setInputMethodHints(QtCore.Qt.ImhNone)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMinimumSize(QtCore.QSize(840, 450))
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_10 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_10.setObjectName("verticalLayout_10")
        self.horizontalLayout_center = QtWidgets.QHBoxLayout()
        self.horizontalLayout_center.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_center.setObjectName("horizontalLayout_center")
        self.verticalLayout_Acquisition = QtWidgets.QVBoxLayout()
        self.verticalLayout_Acquisition.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_Acquisition.setObjectName("verticalLayout_Acquisition")
        self.groupBox_Devices = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_Devices.sizePolicy().hasHeightForWidth())
        self.groupBox_Devices.setSizePolicy(sizePolicy)
        self.groupBox_Devices.setMinimumSize(QtCore.QSize(265, 85))
        self.groupBox_Devices.setObjectName("groupBox_Devices")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.groupBox_Devices)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.horizontalLayout_LEDs = QtWidgets.QHBoxLayout()
        self.horizontalLayout_LEDs.setObjectName("horizontalLayout_LEDs")
        self.widgetImpedanceAnalyzerLamp = QtWidgets.QWidget(self.groupBox_Devices)
        self.widgetImpedanceAnalyzerLamp.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetImpedanceAnalyzerLamp.sizePolicy().hasHeightForWidth())
        self.widgetImpedanceAnalyzerLamp.setSizePolicy(sizePolicy)
        self.widgetImpedanceAnalyzerLamp.setMinimumSize(QtCore.QSize(24, 24))
        self.widgetImpedanceAnalyzerLamp.setStyleSheet("background-color: #ff0000;\n"
"margin: 4px;\n"
"border-radius: 8px;\n"
"border: 3px solid #333;\n"
"")
        self.widgetImpedanceAnalyzerLamp.setObjectName("widgetImpedanceAnalyzerLamp")
        self.horizontalLayout_LEDs.addWidget(self.widgetImpedanceAnalyzerLamp)
        self.label_ImpendanceAnalyzerLampLabel = QtWidgets.QLabel(self.groupBox_Devices)
        self.label_ImpendanceAnalyzerLampLabel.setObjectName("label_ImpendanceAnalyzerLampLabel")
        self.horizontalLayout_LEDs.addWidget(self.label_ImpendanceAnalyzerLampLabel)
        self.verticalLayout_8.addLayout(self.horizontalLayout_LEDs)
        self.horizontalLayout_Power = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Power.setObjectName("horizontalLayout_Power")
        self.pushButton_Power = QtWidgets.QPushButton(self.groupBox_Devices)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/icons/images/plug.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Power.setIcon(icon1)
        self.pushButton_Power.setObjectName("pushButton_Power")
        self.horizontalLayout_Power.addWidget(self.pushButton_Power)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_Power.addItem(spacerItem)
        self.verticalLayout_8.addLayout(self.horizontalLayout_Power)
        self.verticalLayout_Acquisition.addWidget(self.groupBox_Devices)
        self.groupBox_AcquisitionGeneral = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_AcquisitionGeneral.sizePolicy().hasHeightForWidth())
        self.groupBox_AcquisitionGeneral.setSizePolicy(sizePolicy)
        self.groupBox_AcquisitionGeneral.setMinimumSize(QtCore.QSize(240, 130))
        self.groupBox_AcquisitionGeneral.setObjectName("groupBox_AcquisitionGeneral")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.groupBox_AcquisitionGeneral)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.formLayout_acquisition = QtWidgets.QFormLayout()
        self.formLayout_acquisition.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.formLayout_acquisition.setObjectName("formLayout_acquisition")
        self.label_Frequency = QtWidgets.QLabel(self.groupBox_AcquisitionGeneral)
        self.label_Frequency.setObjectName("label_Frequency")
        self.formLayout_acquisition.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_Frequency)
        self.lineEdit_Frequency = QtWidgets.QLineEdit(self.groupBox_AcquisitionGeneral)
        self.lineEdit_Frequency.setInputMask("")
        self.lineEdit_Frequency.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.lineEdit_Frequency.setObjectName("lineEdit_Frequency")
        self.formLayout_acquisition.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.lineEdit_Frequency)
        self.label_IntegrationTime = QtWidgets.QLabel(self.groupBox_AcquisitionGeneral)
        self.label_IntegrationTime.setObjectName("label_IntegrationTime")
        self.formLayout_acquisition.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_IntegrationTime)
        self.comboBox_IntegrationTime = QtWidgets.QComboBox(self.groupBox_AcquisitionGeneral)
        self.comboBox_IntegrationTime.setCurrentText("Medium (5 ms)")
        self.comboBox_IntegrationTime.setObjectName("comboBox_IntegrationTime")
        self.comboBox_IntegrationTime.addItem("")
        self.comboBox_IntegrationTime.addItem("")
        self.comboBox_IntegrationTime.addItem("")
        self.formLayout_acquisition.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.comboBox_IntegrationTime)
        self.label_NumberOfAverages = QtWidgets.QLabel(self.groupBox_AcquisitionGeneral)
        self.label_NumberOfAverages.setObjectName("label_NumberOfAverages")
        self.formLayout_acquisition.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_NumberOfAverages)
        self.comboBox_NumberOfAverages = QtWidgets.QComboBox(self.groupBox_AcquisitionGeneral)
        self.comboBox_NumberOfAverages.setCurrentText("8")
        self.comboBox_NumberOfAverages.setMinimumContentsLength(0)
        self.comboBox_NumberOfAverages.setObjectName("comboBox_NumberOfAverages")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.comboBox_NumberOfAverages.addItem("")
        self.formLayout_acquisition.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.comboBox_NumberOfAverages)
        self.label_Delay = QtWidgets.QLabel(self.groupBox_AcquisitionGeneral)
        self.label_Delay.setObjectName("label_Delay")
        self.formLayout_acquisition.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_Delay)
        self.spinBox_Delay = QtWidgets.QSpinBox(self.groupBox_AcquisitionGeneral)
        self.spinBox_Delay.setSuffix(" (ms)")
        self.spinBox_Delay.setMaximum(1000)
        self.spinBox_Delay.setSingleStep(100)
        self.spinBox_Delay.setObjectName("spinBox_Delay")
        self.formLayout_acquisition.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.spinBox_Delay)
        self.verticalLayout_5.addLayout(self.formLayout_acquisition)
        self.verticalLayout_Acquisition.addWidget(self.groupBox_AcquisitionGeneral)
        self.groupBoxAcquisition_Sweep = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBoxAcquisition_Sweep.sizePolicy().hasHeightForWidth())
        self.groupBoxAcquisition_Sweep.setSizePolicy(sizePolicy)
        self.groupBoxAcquisition_Sweep.setMinimumSize(QtCore.QSize(265, 0))
        self.groupBoxAcquisition_Sweep.setObjectName("groupBoxAcquisition_Sweep")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.groupBoxAcquisition_Sweep)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.formLayout_DLCPSweep = QtWidgets.QFormLayout()
        self.formLayout_DLCPSweep.setObjectName("formLayout_DLCPSweep")
        self.label_OscLevelStart = QtWidgets.QLabel(self.groupBoxAcquisition_Sweep)
        self.label_OscLevelStart.setObjectName("label_OscLevelStart")
        self.formLayout_DLCPSweep.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.label_OscLevelStart)
        self.doubleSpinBox_OscLevelStart = QtWidgets.QDoubleSpinBox(self.groupBoxAcquisition_Sweep)
        self.doubleSpinBox_OscLevelStart.setPrefix("")
        self.doubleSpinBox_OscLevelStart.setDecimals(0)
        self.doubleSpinBox_OscLevelStart.setMinimum(10.0)
        self.doubleSpinBox_OscLevelStart.setMaximum(1000.0)
        self.doubleSpinBox_OscLevelStart.setSingleStep(5.0)
        self.doubleSpinBox_OscLevelStart.setProperty("value", 50.0)
        self.doubleSpinBox_OscLevelStart.setObjectName("doubleSpinBox_OscLevelStart")
        self.formLayout_DLCPSweep.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_OscLevelStart)
        self.label_OscLevelStep = QtWidgets.QLabel(self.groupBoxAcquisition_Sweep)
        self.label_OscLevelStep.setObjectName("label_OscLevelStep")
        self.formLayout_DLCPSweep.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_OscLevelStep)
        self.doubleSpinBox_OscLevelStep = QtWidgets.QDoubleSpinBox(self.groupBoxAcquisition_Sweep)
        self.doubleSpinBox_OscLevelStep.setPrefix("")
        self.doubleSpinBox_OscLevelStep.setDecimals(0)
        self.doubleSpinBox_OscLevelStep.setMinimum(10.0)
        self.doubleSpinBox_OscLevelStep.setSingleStep(5.0)
        self.doubleSpinBox_OscLevelStep.setObjectName("doubleSpinBox_OscLevelStep")
        self.formLayout_DLCPSweep.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_OscLevelStep)
        self.label_OscLevelStop = QtWidgets.QLabel(self.groupBoxAcquisition_Sweep)
        self.label_OscLevelStop.setObjectName("label_OscLevelStop")
        self.formLayout_DLCPSweep.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_OscLevelStop)
        self.doubleSpinBox_OscLevelStop = QtWidgets.QDoubleSpinBox(self.groupBoxAcquisition_Sweep)
        self.doubleSpinBox_OscLevelStop.setPrefix("")
        self.doubleSpinBox_OscLevelStop.setDecimals(0)
        self.doubleSpinBox_OscLevelStop.setMinimum(20.0)
        self.doubleSpinBox_OscLevelStop.setMaximum(1000.0)
        self.doubleSpinBox_OscLevelStop.setSingleStep(5.0)
        self.doubleSpinBox_OscLevelStop.setObjectName("doubleSpinBox_OscLevelStop")
        self.formLayout_DLCPSweep.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_OscLevelStop)
        self.label_NominalBiasStart = QtWidgets.QLabel(self.groupBoxAcquisition_Sweep)
        self.label_NominalBiasStart.setObjectName("label_NominalBiasStart")
        self.formLayout_DLCPSweep.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_NominalBiasStart)
        self.doubleSpinBox_NominalBiasStart = QtWidgets.QDoubleSpinBox(self.groupBoxAcquisition_Sweep)
        self.doubleSpinBox_NominalBiasStart.setPrefix("")
        self.doubleSpinBox_NominalBiasStart.setMinimum(-40.0)
        self.doubleSpinBox_NominalBiasStart.setMaximum(40.0)
        self.doubleSpinBox_NominalBiasStart.setSingleStep(0.01)
        self.doubleSpinBox_NominalBiasStart.setProperty("value", -5.0)
        self.doubleSpinBox_NominalBiasStart.setObjectName("doubleSpinBox_NominalBiasStart")
        self.formLayout_DLCPSweep.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_NominalBiasStart)
        self.label_NominalBiasStep = QtWidgets.QLabel(self.groupBoxAcquisition_Sweep)
        self.label_NominalBiasStep.setObjectName("label_NominalBiasStep")
        self.formLayout_DLCPSweep.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_NominalBiasStep)
        self.doubleSpinBox_NominalBiasStep = QtWidgets.QDoubleSpinBox(self.groupBoxAcquisition_Sweep)
        self.doubleSpinBox_NominalBiasStep.setPrefix("")
        self.doubleSpinBox_NominalBiasStep.setMinimum(0.01)
        self.doubleSpinBox_NominalBiasStep.setMaximum(1.0)
        self.doubleSpinBox_NominalBiasStep.setSingleStep(0.01)
        self.doubleSpinBox_NominalBiasStep.setObjectName("doubleSpinBox_NominalBiasStep")
        self.formLayout_DLCPSweep.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_NominalBiasStep)
        self.label_NominalBiasStop = QtWidgets.QLabel(self.groupBoxAcquisition_Sweep)
        self.label_NominalBiasStop.setObjectName("label_NominalBiasStop")
        self.formLayout_DLCPSweep.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_NominalBiasStop)
        self.doubleSpinBox_NominalBiasStop = QtWidgets.QDoubleSpinBox(self.groupBoxAcquisition_Sweep)
        self.doubleSpinBox_NominalBiasStop.setPrefix("")
        self.doubleSpinBox_NominalBiasStop.setMinimum(-40.0)
        self.doubleSpinBox_NominalBiasStop.setMaximum(40.0)
        self.doubleSpinBox_NominalBiasStop.setSingleStep(0.01)
        self.doubleSpinBox_NominalBiasStop.setProperty("value", -3.0)
        self.doubleSpinBox_NominalBiasStop.setObjectName("doubleSpinBox_NominalBiasStop")
        self.formLayout_DLCPSweep.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.doubleSpinBox_NominalBiasStop)
        self.verticalLayout_7.addLayout(self.formLayout_DLCPSweep)
        self.verticalLayout_Acquisition.addWidget(self.groupBoxAcquisition_Sweep)
        self.horizontalLayout_center.addLayout(self.verticalLayout_Acquisition)
        self.groupBox_Plot = QtWidgets.QGroupBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_Plot.sizePolicy().hasHeightForWidth())
        self.groupBox_Plot.setSizePolicy(sizePolicy)
        self.groupBox_Plot.setMinimumSize(QtCore.QSize(547, 417))
        self.groupBox_Plot.setObjectName("groupBox_Plot")
        self.verticalLayout_9 = QtWidgets.QVBoxLayout(self.groupBox_Plot)
        self.verticalLayout_9.setObjectName("verticalLayout_9")
        self.horizontalLayout_Status = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Status.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_Status.setSpacing(4)
        self.horizontalLayout_Status.setObjectName("horizontalLayout_Status")
        self.widgetStatusLamp = QtWidgets.QWidget(self.groupBox_Plot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetStatusLamp.sizePolicy().hasHeightForWidth())
        self.widgetStatusLamp.setSizePolicy(sizePolicy)
        self.widgetStatusLamp.setMinimumSize(QtCore.QSize(24, 24))
        self.widgetStatusLamp.setToolTipDuration(-1)
        self.widgetStatusLamp.setStyleSheet("background-color: #fff;\n"
"margin: 4px;\n"
"border-radius: 8px;\n"
"border: 3px solid #333;\n"
"")
        self.widgetStatusLamp.setObjectName("widgetStatusLamp")
        self.horizontalLayout_Status.addWidget(self.widgetStatusLamp)
        self.label_SystemStatus = QtWidgets.QLabel(self.groupBox_Plot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_SystemStatus.sizePolicy().hasHeightForWidth())
        self.label_SystemStatus.setSizePolicy(sizePolicy)
        self.label_SystemStatus.setMinimumSize(QtCore.QSize(0, 24))
        self.label_SystemStatus.setMaximumSize(QtCore.QSize(16777215, 24))
        self.label_SystemStatus.setObjectName("label_SystemStatus")
        self.horizontalLayout_Status.addWidget(self.label_SystemStatus)
        spacerItem1 = QtWidgets.QSpacerItem(40, 24, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_Status.addItem(spacerItem1)
        self.label_Progress = QtWidgets.QLabel(self.groupBox_Plot)
        self.label_Progress.setObjectName("label_Progress")
        self.horizontalLayout_Status.addWidget(self.label_Progress)
        self.lineEditProgress = QtWidgets.QLineEdit(self.groupBox_Plot)
        self.lineEditProgress.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEditProgress.sizePolicy().hasHeightForWidth())
        self.lineEditProgress.setSizePolicy(sizePolicy)
        self.lineEditProgress.setMinimumSize(QtCore.QSize(133, 0))
        self.lineEditProgress.setToolTipDuration(-1)
        self.lineEditProgress.setStyleSheet("font-weight: bold;")
        self.lineEditProgress.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEditProgress.setReadOnly(True)
        self.lineEditProgress.setObjectName("lineEditProgress")
        self.horizontalLayout_Status.addWidget(self.lineEditProgress)
        self.verticalLayout_9.addLayout(self.horizontalLayout_Status)
        self.widgetCVPlot = QtWidgets.QWidget(self.groupBox_Plot)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widgetCVPlot.sizePolicy().hasHeightForWidth())
        self.widgetCVPlot.setSizePolicy(sizePolicy)
        self.widgetCVPlot.setMinimumSize(QtCore.QSize(527, 325))
        self.widgetCVPlot.setStyleSheet("background: #fff;")
        self.widgetCVPlot.setObjectName("widgetCVPlot")
        self.verticalLayout_9.addWidget(self.widgetCVPlot)
        self.horizontalLayout_center.addWidget(self.groupBox_Plot)
        self.verticalLayout_10.addLayout(self.horizontalLayout_center)
        self.horizontalLayout_start = QtWidgets.QHBoxLayout()
        self.horizontalLayout_start.setObjectName("horizontalLayout_start")
        self.label_FileTag = QtWidgets.QLabel(self.centralwidget)
        self.label_FileTag.setObjectName("label_FileTag")
        self.horizontalLayout_start.addWidget(self.label_FileTag)
        self.lineEdit_FileTag = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_FileTag.setInputMask("")
        self.lineEdit_FileTag.setObjectName("lineEdit_FileTag")
        self.horizontalLayout_start.addWidget(self.lineEdit_FileTag)
        self.label_Storge = QtWidgets.QLabel(self.centralwidget)
        self.label_Storge.setObjectName("label_Storge")
        self.horizontalLayout_start.addWidget(self.label_Storge)
        self.plainTextEdit_SaveFolder = QtWidgets.QPlainTextEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.plainTextEdit_SaveFolder.sizePolicy().hasHeightForWidth())
        self.plainTextEdit_SaveFolder.setSizePolicy(sizePolicy)
        self.plainTextEdit_SaveFolder.setMinimumSize(QtCore.QSize(100, 40))
        self.plainTextEdit_SaveFolder.setReadOnly(True)
        self.plainTextEdit_SaveFolder.setObjectName("plainTextEdit_SaveFolder")
        self.horizontalLayout_start.addWidget(self.plainTextEdit_SaveFolder)
        self.pushButton_ChangeDir = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_ChangeDir.setMinimumSize(QtCore.QSize(0, 0))
        self.pushButton_ChangeDir.setStyleSheet("padding: 3.5px 8px;")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/icons/images/folder-into.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_ChangeDir.setIcon(icon2)
        self.pushButton_ChangeDir.setObjectName("pushButton_ChangeDir")
        self.horizontalLayout_start.addWidget(self.pushButton_ChangeDir)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_start.addItem(spacerItem2)
        self.pushButton_AbortMeasurement = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_AbortMeasurement.setStyleSheet("background: #ff0000;\n"
"font-weight: bold;")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/icons/images/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_AbortMeasurement.setIcon(icon3)
        self.pushButton_AbortMeasurement.setObjectName("pushButton_AbortMeasurement")
        self.horizontalLayout_start.addWidget(self.pushButton_AbortMeasurement)
        self.pushButton_StartMeasurement = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_StartMeasurement.setStyleSheet("background: #00ff00;\n"
"font-weight: bold;")
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/icons/images/start.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_StartMeasurement.setIcon(icon4)
        self.pushButton_StartMeasurement.setObjectName("pushButton_StartMeasurement")
        self.horizontalLayout_start.addWidget(self.pushButton_StartMeasurement)
        self.verticalLayout_10.addLayout(self.horizontalLayout_start)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 840, 21))
        self.menubar.setObjectName("menubar")
        self.menu_File = QtWidgets.QMenu(self.menubar)
        self.menu_File.setObjectName("menu_File")
        self.menu_Measurement = QtWidgets.QMenu(self.menubar)
        self.menu_Measurement.setObjectName("menu_Measurement")
        self.menu_System = QtWidgets.QMenu(self.menubar)
        self.menu_System.setObjectName("menu_System")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setStyleSheet("")
        self.toolBar.setMovable(False)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.action_Load_Acquisition_Settings = QtWidgets.QAction(MainWindow)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/icons/images/monitor.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Load_Acquisition_Settings.setIcon(icon5)
        self.action_Load_Acquisition_Settings.setShortcut("")
        self.action_Load_Acquisition_Settings.setObjectName("action_Load_Acquisition_Settings")
        self.actionLoad_System_Settings = QtWidgets.QAction(MainWindow)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/icons/images/spanner.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionLoad_System_Settings.setIcon(icon6)
        self.actionLoad_System_Settings.setObjectName("actionLoad_System_Settings")
        self.actionTest = QtWidgets.QAction(MainWindow)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/icons/images/test.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionTest.setIcon(icon7)
        self.actionTest.setObjectName("actionTest")
        self.action_Quit = QtWidgets.QAction(MainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/icons/images/door-exit.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Quit.setIcon(icon8)
        self.action_Quit.setObjectName("action_Quit")
        self.actionStart = QtWidgets.QAction(MainWindow)
        self.actionStart.setIcon(icon4)
        self.actionStart.setObjectName("actionStart")
        self.actionSto_p = QtWidgets.QAction(MainWindow)
        self.actionSto_p.setIcon(icon3)
        self.actionSto_p.setObjectName("actionSto_p")
        self.action_Connect_Devices = QtWidgets.QAction(MainWindow)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/icons/images/off.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        icon9.addPixmap(QtGui.QPixmap(":/icons/images/switch.png"), QtGui.QIcon.Normal, QtGui.QIcon.On)
        self.action_Connect_Devices.setIcon(icon9)
        self.action_Connect_Devices.setObjectName("action_Connect_Devices")
        self.action_Disconnect_Devices = QtWidgets.QAction(MainWindow)
        self.action_Disconnect_Devices.setIcon(icon1)
        self.action_Disconnect_Devices.setObjectName("action_Disconnect_Devices")
        self.actionSelect_Save_F_older = QtWidgets.QAction(MainWindow)
        self.actionSelect_Save_F_older.setIcon(icon2)
        self.actionSelect_Save_F_older.setObjectName("actionSelect_Save_F_older")
        self.actionS_ave_Acquisition_Settings = QtWidgets.QAction(MainWindow)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/icons/images/floppy-diskette-with-pen.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionS_ave_Acquisition_Settings.setIcon(icon10)
        self.actionS_ave_Acquisition_Settings.setObjectName("actionS_ave_Acquisition_Settings")
        self.menu_File.addAction(self.action_Load_Acquisition_Settings)
        self.menu_File.addAction(self.actionS_ave_Acquisition_Settings)
        self.menu_File.addAction(self.actionSelect_Save_F_older)
        self.menu_File.addSeparator()
        self.menu_File.addAction(self.action_Quit)
        self.menu_Measurement.addAction(self.actionTest)
        self.menu_Measurement.addAction(self.actionStart)
        self.menu_Measurement.addSeparator()
        self.menu_Measurement.addAction(self.actionSto_p)
        self.menu_System.addAction(self.action_Connect_Devices)
        self.menu_System.addAction(self.action_Disconnect_Devices)
        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menu_System.menuAction())
        self.menubar.addAction(self.menu_Measurement.menuAction())
        self.toolBar.addAction(self.action_Load_Acquisition_Settings)
        self.toolBar.addAction(self.actionS_ave_Acquisition_Settings)
        self.toolBar.addAction(self.actionSelect_Save_F_older)
        self.toolBar.addAction(self.actionTest)
        self.toolBar.addAction(self.actionStart)
        self.toolBar.addAction(self.actionSto_p)
        self.label_Frequency.setBuddy(self.lineEdit_Frequency)
        self.label_IntegrationTime.setBuddy(self.comboBox_IntegrationTime)
        self.label_NumberOfAverages.setBuddy(self.comboBox_NumberOfAverages)
        self.label_Delay.setBuddy(self.spinBox_Delay)
        self.label_OscLevelStart.setBuddy(self.doubleSpinBox_OscLevelStart)
        self.label_OscLevelStep.setBuddy(self.doubleSpinBox_OscLevelStep)
        self.label_OscLevelStop.setBuddy(self.doubleSpinBox_OscLevelStop)
        self.label_NominalBiasStart.setBuddy(self.doubleSpinBox_NominalBiasStart)
        self.label_NominalBiasStep.setBuddy(self.doubleSpinBox_NominalBiasStep)
        self.label_NominalBiasStop.setBuddy(self.doubleSpinBox_NominalBiasStop)
        self.label_FileTag.setBuddy(self.lineEdit_FileTag)
        self.label_Storge.setBuddy(self.plainTextEdit_SaveFolder)

        self.retranslateUi(MainWindow)
        self.comboBox_IntegrationTime.setCurrentIndex(1)
        self.comboBox_NumberOfAverages.setCurrentIndex(3)
        self.action_Quit.triggered.connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.pushButton_Power, self.lineEdit_Frequency)
        MainWindow.setTabOrder(self.lineEdit_Frequency, self.comboBox_IntegrationTime)
        MainWindow.setTabOrder(self.comboBox_IntegrationTime, self.comboBox_NumberOfAverages)
        MainWindow.setTabOrder(self.comboBox_NumberOfAverages, self.spinBox_Delay)
        MainWindow.setTabOrder(self.spinBox_Delay, self.doubleSpinBox_OscLevelStart)
        MainWindow.setTabOrder(self.doubleSpinBox_OscLevelStart, self.doubleSpinBox_OscLevelStep)
        MainWindow.setTabOrder(self.doubleSpinBox_OscLevelStep, self.doubleSpinBox_OscLevelStop)
        MainWindow.setTabOrder(self.doubleSpinBox_OscLevelStop, self.doubleSpinBox_NominalBiasStart)
        MainWindow.setTabOrder(self.doubleSpinBox_NominalBiasStart, self.doubleSpinBox_NominalBiasStep)
        MainWindow.setTabOrder(self.doubleSpinBox_NominalBiasStep, self.doubleSpinBox_NominalBiasStop)
        MainWindow.setTabOrder(self.doubleSpinBox_NominalBiasStop, self.lineEdit_FileTag)
        MainWindow.setTabOrder(self.lineEdit_FileTag, self.pushButton_ChangeDir)
        MainWindow.setTabOrder(self.pushButton_ChangeDir, self.plainTextEdit_SaveFolder)
        MainWindow.setTabOrder(self.plainTextEdit_SaveFolder, self.lineEditProgress)
        MainWindow.setTabOrder(self.lineEditProgress, self.pushButton_StartMeasurement)
        MainWindow.setTabOrder(self.pushButton_StartMeasurement, self.pushButton_AbortMeasurement)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "PyDLCP"))
        MainWindow.setWhatsThis(_translate("MainWindow", "<html><head/><body><p>PyDLCP is an acquisition tool that collects capacitance data sweeping different oscillator levels and nominal biases. The data is stored in hfds files for later processing. The sweeps provide data to be fitted by polynomial function to estimate the drive level as a function of a position variable. </p></body></html>"))
        self.groupBox_Devices.setTitle(_translate("MainWindow", "System"))
        self.label_ImpendanceAnalyzerLampLabel.setText(_translate("MainWindow", "Impedance Analyzer"))
        self.pushButton_Power.setToolTip(_translate("MainWindow", "<html><head/><body><p>Open the connection to the instruments.</p></body></html>"))
        self.pushButton_Power.setText(_translate("MainWindow", "Connect"))
        self.groupBox_AcquisitionGeneral.setTitle(_translate("MainWindow", "General"))
        self.label_Frequency.setText(_translate("MainWindow", "Frequency (Hz):"))
        self.lineEdit_Frequency.setToolTip(_translate("MainWindow", "<html><head/><body><p>The oscillator frequency in Hz.</p></body></html>"))
        self.lineEdit_Frequency.setText(_translate("MainWindow", "1.0E+06"))
        self.label_IntegrationTime.setText(_translate("MainWindow", "Integration Time:"))
        self.comboBox_IntegrationTime.setToolTip(_translate("MainWindow", "<html><head/><body><p>The integration time.</p></body></html>"))
        self.comboBox_IntegrationTime.setItemText(0, _translate("MainWindow", "Short (500 us)"))
        self.comboBox_IntegrationTime.setItemText(1, _translate("MainWindow", "Medium (5 ms)"))
        self.comboBox_IntegrationTime.setItemText(2, _translate("MainWindow", "Long (100 ms)"))
        self.label_NumberOfAverages.setText(_translate("MainWindow", "Number of Averages:"))
        self.comboBox_NumberOfAverages.setToolTip(_translate("MainWindow", "<html><head/><body><p>The number of averages.</p></body></html>"))
        self.comboBox_NumberOfAverages.setItemText(0, _translate("MainWindow", "1"))
        self.comboBox_NumberOfAverages.setItemText(1, _translate("MainWindow", "2"))
        self.comboBox_NumberOfAverages.setItemText(2, _translate("MainWindow", "4"))
        self.comboBox_NumberOfAverages.setItemText(3, _translate("MainWindow", "8"))
        self.comboBox_NumberOfAverages.setItemText(4, _translate("MainWindow", "16"))
        self.comboBox_NumberOfAverages.setItemText(5, _translate("MainWindow", "32"))
        self.comboBox_NumberOfAverages.setItemText(6, _translate("MainWindow", "64"))
        self.comboBox_NumberOfAverages.setItemText(7, _translate("MainWindow", "128"))
        self.comboBox_NumberOfAverages.setItemText(8, _translate("MainWindow", "256"))
        self.label_Delay.setText(_translate("MainWindow", "Delay:"))
        self.spinBox_Delay.setToolTip(_translate("MainWindow", "<html><head/><body><p>Delay time before the acquisition (in milliseconds).</p></body></html>"))
        self.groupBoxAcquisition_Sweep.setTitle(_translate("MainWindow", "DLCP Sweep"))
        self.label_OscLevelStart.setText(_translate("MainWindow", "Osc Level Start:"))
        self.doubleSpinBox_OscLevelStart.setToolTip(_translate("MainWindow", "<html><head/><body><p>The oscillator level start value in mV peak-to-peak.</p></body></html>"))
        self.doubleSpinBox_OscLevelStart.setSuffix(_translate("MainWindow", " (mV p-p)"))
        self.label_OscLevelStep.setText(_translate("MainWindow", "Osc Level Step:"))
        self.doubleSpinBox_OscLevelStep.setToolTip(_translate("MainWindow", "<html><head/><body><p>The oscillator level step value in mV peak-to-peak.</p></body></html>"))
        self.doubleSpinBox_OscLevelStep.setSuffix(_translate("MainWindow", " (mV p-p)"))
        self.label_OscLevelStop.setText(_translate("MainWindow", "Osc Level Stop:"))
        self.doubleSpinBox_OscLevelStop.setToolTip(_translate("MainWindow", "<html><head/><body><p>The oscillator level stop value in mV peak-to-peak.</p></body></html>"))
        self.doubleSpinBox_OscLevelStop.setSuffix(_translate("MainWindow", " (mV p-p)"))
        self.label_NominalBiasStart.setText(_translate("MainWindow", "Nominal Bias Start:"))
        self.doubleSpinBox_NominalBiasStart.setToolTip(_translate("MainWindow", "<html><head/><body><p>The nominal bias start value in V.</p></body></html>"))
        self.doubleSpinBox_NominalBiasStart.setSuffix(_translate("MainWindow", " (V)"))
        self.label_NominalBiasStep.setText(_translate("MainWindow", "Nominal Bias Step:"))
        self.doubleSpinBox_NominalBiasStep.setToolTip(_translate("MainWindow", "<html><head/><body><p>The nominal bias step value in V.</p></body></html>"))
        self.doubleSpinBox_NominalBiasStep.setSuffix(_translate("MainWindow", " (V)"))
        self.label_NominalBiasStop.setText(_translate("MainWindow", "Nominal Bias Stop:"))
        self.doubleSpinBox_NominalBiasStop.setToolTip(_translate("MainWindow", "<html><head/><body><p>The nominal bias stop value in V.</p></body></html>"))
        self.doubleSpinBox_NominalBiasStop.setSuffix(_translate("MainWindow", " (V)"))
        self.groupBox_Plot.setTitle(_translate("MainWindow", "Graph"))
        self.label_SystemStatus.setText(_translate("MainWindow", "Idle"))
        self.label_Progress.setText(_translate("MainWindow", "Point"))
        self.lineEditProgress.setToolTip(_translate("MainWindow", "<html><head/><body><p>The progress of the measurement.</p></body></html>"))
        self.lineEditProgress.setText(_translate("MainWindow", "0/0"))
        self.label_FileTag.setText(_translate("MainWindow", "File Tag:"))
        self.label_Storge.setText(_translate("MainWindow", "Save Folder"))
        self.pushButton_ChangeDir.setToolTip(_translate("MainWindow", "<html><head/><body><p>Select the folder to which the data will be saved.</p></body></html>"))
        self.pushButton_ChangeDir.setText(_translate("MainWindow", "Change Folder"))
        self.pushButton_AbortMeasurement.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:600; color:#ff0000;\">Aborts the current measurement!</span></p></body></html>"))
        self.pushButton_AbortMeasurement.setText(_translate("MainWindow", "Abort"))
        self.pushButton_StartMeasurement.setToolTip(_translate("MainWindow", "<html><head/><body><p>Starts the acquisition.</p></body></html>"))
        self.pushButton_StartMeasurement.setText(_translate("MainWindow", "Start"))
        self.menu_File.setTitle(_translate("MainWindow", "&File"))
        self.menu_Measurement.setTitle(_translate("MainWindow", "&Measurement"))
        self.menu_System.setTitle(_translate("MainWindow", "&System"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.toolBar.setToolTip(_translate("MainWindow", "<html><head/><body><p><br/></p></body></html>"))
        self.action_Load_Acquisition_Settings.setText(_translate("MainWindow", "Lo&ad Acquisition Settings"))
        self.actionLoad_System_Settings.setText(_translate("MainWindow", "Load &System Settings"))
        self.actionLoad_System_Settings.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionTest.setText(_translate("MainWindow", "Run &Test"))
        self.action_Quit.setText(_translate("MainWindow", "&Quit"))
        self.actionStart.setText(_translate("MainWindow", "&Start Measurement"))
        self.actionSto_p.setText(_translate("MainWindow", "Sto&p"))
        self.action_Connect_Devices.setText(_translate("MainWindow", "&Connect Devices"))
        self.action_Disconnect_Devices.setText(_translate("MainWindow", "&Disconnect Devices"))
        self.actionSelect_Save_F_older.setText(_translate("MainWindow", "Select Save &Folder"))
        self.actionS_ave_Acquisition_Settings.setText(_translate("MainWindow", "&Save Acquisition Settings"))

from dlcpgui import dlcp_images_rc
