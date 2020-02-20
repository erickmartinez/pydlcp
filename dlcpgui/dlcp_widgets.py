import os
import sys
import matplotlib as mpl
from PyQt5 import QtWidgets
from PyQt5.QtGui import QDoubleValidator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
sys.path.insert(0, os.path.abspath('../'))
from dlcpgui.dlcpgui import Ui_MainWindow
from dlcpgui.cv_sweep_dialog import Ui_Dialog as CVDialog

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


class CVDialogPopUp(QtWidgets.QDialog, CVDialog):
    def __init__(self, *args, obj=None, **kwargs):
        super(CVDialogPopUp, self).__init__(*args, **kwargs)
        self.setupUi(self)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    _folder = os.getcwd()

    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.lineEditFrequency.setValidator(QDoubleValidator(100, 1E6, 1))
        self.actionTest.triggered.connect(self.open_test_dialog)
        self.pushButton_ChangeDir.clicked.connect(self.folder_dialog)

        self.sc = DynamicMplCanvas(parent=self.widgetCVPlot, width=5.27, height=3.25, dpi=100)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.addToolBar(NavigationToolbar(self.sc, self.widgetCVPlot))
        self.sc.axes.set_xlabel('Oscillator Level (mV p-p)')
        self.sc.axes.set_ylabel('Capacitance (F)')
        self.statusBar().showMessage('Disconnected')
        self.plainTextEdit_SaveFolder.setPlainText(self._folder)
        self.setFixedSize(self.size())

    @staticmethod
    def open_test_dialog():
        dlg = CVDialogPopUp()
        dlg.exec()

    def folder_dialog(self):
        folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
        if folder != "" and folder is not None:
            self._folder = folder
        self.plainTextEdit_SaveFolder.setPlainText(self._folder)