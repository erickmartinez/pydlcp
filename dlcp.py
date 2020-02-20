import os
import sys
sys.path.insert(0, os.path.abspath('../'))
from PyQt5 import QtWidgets
from dlcpgui.dlcp_widgets import DynamicMplCanvas, CVDialogPopUp, MainWindow

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
