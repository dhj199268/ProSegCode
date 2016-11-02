from PyQt4 import QtGui, QtCore
from src.main.UI.MainWindow import Ui_MainWindow
import logging.config


class Window(QtGui.QMainWindow, Ui_MainWindow):
    logger = logging.getLogger("Window")

    def __init__(self):
        super(Window, self).__init__()
        self.logger.info("init Window Class.")
        self.setupUi(self)

        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(self.width(), self.height())

        # singal slot register
        self.singalHander()

        # init Widgets

    def singalHander(self):
        # singal process
        self.logger.info("singal process,register singal slot.")
        QtCore.QObject.connect(self.OpenImge, QtCore.SIGNAL('clicked()'), self.showImge)

    def showImge(self):
        self.logger.info("show image")
        QtGui.QMessageBox.information(self, u"sdf", u"sdf")


if __name__ == '__main__':
    import sys, os

    # load logging
    con_path = r"H:\ProSegCode\src\main\logging.ini"
    if os.path.isfile(con_path):
        logging.config.fileConfig(con_path)

    # setup app
    app = QtGui.QApplication(sys.argv)
    ui = Window()
    ui.show()
    sys.exit(app.exec_())
