import os
import sys
import matplotlib as mpl
import configparser
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QDoubleValidator, QRegExpValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import pydlcp.errors as errors

sys.path.insert(0, os.path.abspath('../'))
from dlcpgui.dlcpgui import Ui_MainWindow
from dlcpgui.cv_sweep_dialog import Ui_Dialog
from pydlcp.controller import Controller

mpl.use('Qt5Agg')


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=4.5, height=3.25, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        self.compute_initial_figure()
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass


class DynamicMplCanvas(MplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)

    def compute_initial_figure(self):
        self.axes.plot([0, 1, 2, 3], [1, 2, 0, 4], 'r')

    def update_figure(self):
        # Build a list of 4 random integers between 0 and 10 (both inclusive)
        pass


#        self.axes.cla()
#        self.axes.plot([0, 1, 2, 3], data, 'r')
#        self.draw()


class CVDialogPopUp(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self, *args, obj=None, **kwargs):
        super(CVDialogPopUp, self).__init__(*args, **kwargs)
        self.setupUi(self)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    _configFileName: str = None
    _connected: bool = False
    _dlcpParams = {}
    _folder = os.getcwd()
    _integrationTimeStr = {'ITM1': 0, 'ITM2': 1, 'ITM3': 2}
    _integrationTimeIdx = ['ITM1', 'ITM2', 'ITM3']
    _numberOfAveragesVal2Idx = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4, 32: 5, 64: 6, 128: 7, 256: 8}
    _numberOfAveragesIdx2Val = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    _validFields: bool = False

    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self._controller = Controller()
        self.setupUi(self)

        self.load_state()
        filetag_validator = QRegExpValidator(QtCore.QRegExp(r'^[\w\-. ]+$'))
        self.lineEdit_Frequency.setValidator(QDoubleValidator(100, 1E6, 1))
        self.lineEdit_FileTag.setValidator(filetag_validator)
        self.actionTest.triggered.connect(self.open_test_dialog)
        self.pushButton_ChangeDir.clicked.connect(self.folder_dialog)
        self.pushButton_Power.clicked.connect(self.connect_devices)
        self.pushButton_StartMeasurement.clicked.connect(self.start_measurement)
        self.pushButton_AbortMeasurement.clicked.connect(self.stop_measurement)
        # Disable acquisition buttons until the devices are connected and the params are validated
        self.pushButton_AbortMeasurement.setEnabled(False)
        self.pushButton_StartMeasurement.setEnabled(False)
        self.doubleSpinBox_OscLevelStart.valueChanged.connect(self.validate_fields)
        self.doubleSpinBox_OscLevelStep.valueChanged.connect(self.validate_fields)
        self.doubleSpinBox_OscLevelStop.valueChanged.connect(self.validate_fields)
        self.doubleSpinBox_NominalBiasStart.valueChanged.connect(self.validate_fields)
        self.doubleSpinBox_NominalBiasStep.valueChanged.connect(self.validate_fields)
        self.doubleSpinBox_NominalBiasStop.valueChanged.connect(self.validate_fields)

        self.action_Connect_Devices.triggered.connect(self.connect_devices)
        self.action_Disconnect_Devices.triggered.connect(self.disconnect_devices)
        self.action_Load_Acquisition_Settings.triggered.connect(self.load_acquisition_parameters)
        self.actionS_ave_Acquisition_Settings.triggered.connect(self.save_acquisition_parameters)
        self.actionSelect_Save_F_older.triggered.connect(self.folder_dialog)
        self.actionStart.setEnabled(False)
        self.actionSto_p.setEnabled(False)

        self.sc = DynamicMplCanvas(parent=self.widgetCVPlot, width=5.27, height=3.25, dpi=100)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.addToolBar(NavigationToolbar(self.sc, self.widgetCVPlot))
        self.sc.axes.set_xlabel('Oscillator Level (mV p-p)')
        self.sc.axes.set_ylabel('Capacitance (F)')
        self.statusBar().showMessage('Disconnected')
        self.plainTextEdit_SaveFolder.setPlainText(self._folder)
        self.setFixedSize(self.size())

    def start_measurement(self):
        self._controller.abort = False
        self.validate_fields()

        if self._validFields:
            self.save_acquisition_parameters()
            # Parse the configuration file
            # Load the system configuration file
            config = configparser.ConfigParser()
            config.read(self._configFileName)
            try:
                self._controller.load_test_config(config)
            except errors.ConfigurationError as e:
                error_dialog = QtWidgets.QErrorMessage(self)
                error_dialog.setWindowModality(QtCore.Qt.WindowModal)
                error_dialog.showMessage(e.message)
                self._validFields = False
                self.widgetStatusLamp.setStyleSheet("background-color: #ff0000;\n"
                                                    "margin: 4px;\n"
                                                    "border-radius: 8px;\n"
                                                    "border: 3px solid #333;\n"
                                                    "")
                self.label_SystemStatus.setText('Error')
                self.statusbar.showMessage('Error')
            else:
                self.pushButton_StartMeasurement.setEnabled(False)
                self.pushButton_AbortMeasurement.setEnabled(True)
                self.actionSto_p.setEnabled(True)
                self.widgetStatusLamp.setStyleSheet("background-color: #00ff00;\n"
                                                    "margin: 4px;\n"
                                                    "border-radius: 8px;\n"
                                                    "border: 3px solid #333;\n"
                                                    "")
                self.label_SystemStatus.setText('Running')
                self.statusbar.showMessage('Measuring...')
                try:
                    success = self._controller.start_dlcp()
                except (errors.InstrumentError, errors.ConfigurationError, errors.DLCPSystemConfigError) as e:
                    error_dialog = QtWidgets.QErrorMessage(self)
                    error_dialog.setWindowModality(QtCore.Qt.WindowModal)
                    error_dialog.showMessage(e.message)
                finally:
                    self.pushButton_StartMeasurement.setEnabled(True)
                    self.widgetStatusLamp.setStyleSheet("background-color: #fff;\n"
                                                        "margin: 4px;\n"
                                                        "border-radius: 8px;\n"
                                                        "border: 3px solid #333;\n"
                                                        "")
                    self.label_SystemStatus.setText('Idle')
                    self.statusbar.showMessage('Connected. Idle')

    def stop_measurement(self):
        self._controller.abort = True
        self.pushButton_StartMeasurement.setEnabled(True)
        self.widgetStatusLamp.setStyleSheet("background-color: #fff;\n"
                                            "margin: 4px;\n"
                                            "border-radius: 8px;\n"
                                            "border: 3px solid #333;\n"
                                            "")
        self.label_SystemStatus.setText('Idle')
        self.statusbar.showMessage('Connected. Idle')

    @staticmethod
    def open_test_dialog():
        dlg = CVDialogPopUp()
        dlg.exec()

    def folder_dialog(self):
        folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory", self._folder))
        if folder != "" and folder is not None:
            self._folder = folder
        self.plainTextEdit_SaveFolder.setPlainText(self._folder)

    def disconnect_devices(self):
        try:
            self._controller.disconnect_devices()
        except errors.InstrumentError as e:
            error_dialog = QtWidgets.QErrorMessage(self)
            error_dialog.setWindowModality(QtCore.Qt.WindowModal)
            error_dialog.showMessage(e.message)
        else:
            self.widgetImpedanceAnalyzerLamp.setStyleSheet("background-color: #ff0000;\n"
                                                           "margin: 4px;\n"
                                                           "border-radius: 8px;\n"
                                                           "border: 3px solid #333;\n"
                                                           "")
            _translate = QtCore.QCoreApplication.translate
            self._connected = False
            self.pushButton_Power.setText(_translate("MainWindow", "Connect"))
            self.statusBar().showMessage('Disconnected')
            self.pushButton_StartMeasurement.setEnabled(False)
            self.actionStart.setEnabled(False)
            self.actionSto_p.setEnabled(False)

    def connect_devices(self):
        if not self._connected:
            try:
                self._controller.init_impedance_analyzer()
            except errors.InstrumentError as e:
                error_dialog = QtWidgets.QErrorMessage(self)
                error_dialog.setWindowModality(QtCore.Qt.WindowModal)
                error_dialog.showMessage(e.message)
                self.pushButton_StartMeasurement.setEnabled(False)
                self.statusBar().showMessage('Disconnected')
            else:
                self.widgetImpedanceAnalyzerLamp.setStyleSheet("background-color: #00ff00;\n"
                                                               "margin: 4px;\n"
                                                               "border-radius: 8px;\n"
                                                               "border: 3px solid #333;\n"
                                                               "")
                _translate = QtCore.QCoreApplication.translate
                self._connected = True
                self.pushButton_Power.setText(_translate("MainWindow", "Disconnect"))
                self.statusBar().showMessage('Connected. Idle')
                self.pushButton_StartMeasurement.setEnabled(True)
                self.actionStart.setEnabled(True)
        else:
            self.disconnect_devices()

    def load_acquisition_parameters(self):
        ini_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Acquisition File", self._folder,
                                                         "ini Files (*.ini)")[0]
        if ini_file != "" and ini_file is not None:
            # Parse the configuration file
            # Load the system configuration file
            config = configparser.ConfigParser()
            config.read(ini_file)
            try:
                self._controller.load_test_config(config)
            except errors.ConfigurationError as e:
                error_dialog = QtWidgets.QErrorMessage(self)
                error_dialog.setWindowModality(QtCore.Qt.WindowModal)
                error_dialog.showMessage(e.message)
                self._validFields = False
            else:
                integration_time_idx = self._integrationTimeStr[config.get(section='dlcp', option='integration_time')]
                number_of_averages_idx = self._numberOfAveragesVal2Idx[config.getint(section='dlcp',
                                                                                     option='number_of_averages')]
                self.lineEdit_Frequency.setText(config.get(section='dlcp', option='frequency'))
                self.comboBox_IntegrationTime.setCurrentIndex(integration_time_idx)
                self.comboBox_NumberOfAverages.setCurrentIndex(number_of_averages_idx)
                self.doubleSpinBox_OscLevelStart.setValue(config.getfloat(section='dlcp', option='osc_level_start'))
                self.doubleSpinBox_OscLevelStep.setValue(config.getfloat(section='dlcp', option='osc_level_step'))
                self.doubleSpinBox_OscLevelStop.setValue(config.getfloat(section='dlcp', option='osc_level_stop'))
                self.doubleSpinBox_NominalBiasStart.setValue(
                    config.getfloat(section='dlcp', option='nominal_bias_start'))
                self.doubleSpinBox_NominalBiasStep.setValue(config.getfloat(section='dlcp', option='nominal_bias_step'))
                self.doubleSpinBox_NominalBiasStop.setValue(config.getfloat(section='dlcp', option='nominal_bias_stop'))
                self.spinBox_Delay.setValue(config.getint(section='dlcp', option='delay'))
                self.lineEdit_FileTag.setText(config.get(section='general', option='file_tag'))
                self.plainTextEdit_SaveFolder.setPlainText(config.get(section='general', option='base_path'))
                self._folder = config.get(section='general', option='base_path')
                self._validFields = True

    def save_acquisition_parameters(self):
        config_filename = os.path.join(self._folder, self.lineEdit_FileTag.text() + '.ini')
        config_filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Configuration', config_filename,
                                                                'ini (*.ini)')[0]

        if config_filename != "" and config_filename is not None:
            self.validate_fields()
            if self._validFields:
                config = configparser.ConfigParser()
                config['general'] = {
                    'base_path': self.plainTextEdit_SaveFolder.toPlainText(),
                    'file_tag': self.lineEdit_FileTag.text(),
                }
                config['dlcp'] = {
                    'frequency': float(self.lineEdit_Frequency.text()),
                    'integration_time': self._integrationTimeIdx[self.comboBox_IntegrationTime.currentIndex()],
                    'number_of_averages': self._numberOfAveragesIdx2Val[self.comboBox_NumberOfAverages.currentIndex()],
                    'delay': self.spinBox_Delay.value(),
                    'osc_level_start': self.doubleSpinBox_OscLevelStart.value(),
                    'osc_level_step': self.doubleSpinBox_OscLevelStep.value(),
                    'osc_level_stop': self.doubleSpinBox_OscLevelStop.value(),
                    'nominal_bias_start': self.doubleSpinBox_NominalBiasStart.value(),
                    'nominal_bias_step': self.doubleSpinBox_NominalBiasStep.value(),
                    'nominal_bias_stop': self.doubleSpinBox_NominalBiasStop.value(),
                }
                with open(config_filename, 'w') as config_file:
                    config.write(config_file)
                self._configFileName = config_filename

    def save_state(self):
        state_filename = os.path.join(os.getcwd(), 'app_state.ini')
        try:
            self.validate_fields()
        except Exception as e:
            print('Error saving the application state: Invalid parameters')
        finally:
            config = configparser.ConfigParser()
            config['state'] = {'current_folder': self.plainTextEdit_SaveFolder.toPlainText()}
            config['general'] = {
                'base_path': self.plainTextEdit_SaveFolder.toPlainText(),
                'file_tag': self.lineEdit_FileTag.text(),
            }
            config['dlcp'] = {
                'frequency': float(self.lineEdit_Frequency.text()),
                'integration_time': self._integrationTimeIdx[self.comboBox_IntegrationTime.currentIndex()],
                'number_of_averages': self._numberOfAveragesIdx2Val[self.comboBox_NumberOfAverages.currentIndex()],
                'delay': self.spinBox_Delay.value(),
                'osc_level_start': self.doubleSpinBox_OscLevelStart.value(),
                'osc_level_step': self.doubleSpinBox_OscLevelStep.value(),
                'osc_level_stop': self.doubleSpinBox_OscLevelStop.value(),
                'nominal_bias_start': self.doubleSpinBox_NominalBiasStart.value(),
                'nominal_bias_step': self.doubleSpinBox_NominalBiasStep.value(),
                'nominal_bias_stop': self.doubleSpinBox_NominalBiasStop.value(),
            }
            with open(state_filename, 'w', encoding='utf-8') as config_file:
                config.write(config_file)

    def load_state(self):
        state_filename = os.path.join(os.getcwd(), 'app_state.ini')
        config = configparser.ConfigParser()
        try:
            config.read(state_filename)
            integration_time_idx = self._integrationTimeStr[config.get(section='dlcp', option='integration_time')]
            number_of_averages_idx = self._numberOfAveragesVal2Idx[config.getint(section='dlcp',
                                                                                 option='number_of_averages')]
            self.lineEdit_Frequency.setText(config.get(section='dlcp', option='frequency'))
            self.comboBox_IntegrationTime.setCurrentIndex(integration_time_idx)
            self.comboBox_NumberOfAverages.setCurrentIndex(number_of_averages_idx)
            self.doubleSpinBox_OscLevelStart.setValue(config.getfloat(section='dlcp', option='osc_level_start'))
            self.doubleSpinBox_OscLevelStep.setValue(config.getfloat(section='dlcp', option='osc_level_step'))
            self.doubleSpinBox_OscLevelStop.setValue(config.getfloat(section='dlcp', option='osc_level_stop'))
            self.doubleSpinBox_NominalBiasStart.setValue(config.getfloat(section='dlcp', option='nominal_bias_start'))
            self.doubleSpinBox_NominalBiasStep.setValue(config.getfloat(section='dlcp', option='nominal_bias_step'))
            self.doubleSpinBox_NominalBiasStop.setValue(config.getfloat(section='dlcp', option='nominal_bias_stop'))
            self.spinBox_Delay.setValue(config.getint(section='dlcp', option='delay'))
            self.lineEdit_FileTag.setText(config.get(section='general', option='file_tag'))
            self.plainTextEdit_SaveFolder.setPlainText(config.get(section='general', option='base_path'))
            current_folder = config.get(section='state', option='current_folder')
            if current_folder != "" and current_folder is not None:
                self._folder = current_folder
            self.widgetStatusLamp.setStyleSheet("background-color: #fff;\n"
                                                "margin: 4px;\n"
                                                "border-radius: 8px;\n"
                                                "border: 3px solid #333;\n"
                                                "")
            self.label_SystemStatus.setText('Idle')
            self.statusbar.showMessage('Connected. Idle')
        except (configparser.NoSectionError, configparser.NoOptionError, configparser.ParsingError,
                configparser.Error) as e:
            print(e.message)

    def validate_fields(self):
        osc_level_delta = self.doubleSpinBox_OscLevelStop.value() - self.doubleSpinBox_OscLevelStart.value()
        nb_delta = self.doubleSpinBox_NominalBiasStop.value() - self.doubleSpinBox_NominalBiasStart.value()

        if self.doubleSpinBox_OscLevelStep.value() > abs(osc_level_delta):
            msg = "Invalid value for the oscillator level step."
            error_dialog = QtWidgets.QErrorMessage(self)
            error_dialog.setWindowModality(QtCore.Qt.WindowModal)
            error_dialog.showMessage(msg)
            self.field_error(self.doubleSpinBox_OscLevelStep)
            self._validFields = False
        elif self.doubleSpinBox_NominalBiasStep.value() > abs(nb_delta):
            msg = "Invalid value for the nominal bias step."
            error_dialog = QtWidgets.QErrorMessage(self)
            error_dialog.setWindowModality(QtCore.Qt.WindowModal)
            error_dialog.showMessage(msg)
            self.field_error(self.doubleSpinBox_NominalBiasStep)
            self._validFields = False
        else:
            self.clear_field_errors()
            self._validFields = True

    def field_error(self, field: QtWidgets.QWidget):
        field.setStyleSheet('border: 1px solid #ff0000;')
        field.setFocus()

    def clear_field_errors(self):
        self.doubleSpinBox_OscLevelStep.setStyleSheet("")
        self.doubleSpinBox_NominalBiasStep.setStyleSheet("")

    def __del__(self):
        self.save_state()
