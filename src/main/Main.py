# encoding:utf-8
from PyQt4 import QtGui, QtCore
from UI.MainWindow import Ui_MainWindow
import logging
import sys, os
import pyqtgraph as pg
import numpy as np
from Prostate.Until.Tool import loadMat

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

tmpname = "temp_data"


class Window(QtGui.QMainWindow, Ui_MainWindow):
    # logger = logging.getLogger("Window")

    def __init__(self):
        super(Window, self).__init__()
        logging.info("init Window Class.")

        # init father ui widget
        self.setupUi(self)

        # init Widgets
        ##更改视图窗口
        self.segView = pg.GraphicsView(self.centralwidget)
        self.segView.setGeometry(QtCore.QRect(20, 20, 281, 261))
        self.segView.setObjectName(_fromUtf8("segView"))
        self.segVB = pg.ViewBox()
        self.segVB.setAspectLocked()
        self.segView.setCentralItem(self.segVB)

        self.imgView = pg.GraphicsView(self.centralwidget)
        self.imgView.setGeometry(QtCore.QRect(180, 330, 281, 261))
        self.imgView.setObjectName(_fromUtf8("imgView"))
        self.imgVB = pg.ViewBox()
        self.imgVB.setAspectLocked()
        self.imgView.setCentralItem(self.imgVB)

        self.groundImgVeiw = pg.GraphicsView(self.centralwidget)
        self.groundImgVeiw.setGeometry(QtCore.QRect(340, 20, 281, 261))
        self.groundImgVeiw.setObjectName(_fromUtf8("groundImgVeiw"))
        self.groundImgVB = pg.ViewBox()
        self.groundImgVB.setAspectLocked()
        self.groundImgVeiw.setCentralItem(self.groundImgVB)

        ## 防止窗口最大化，和变动
        self.setWindowFlags(QtCore.Qt.WindowMinimizeButtonHint)
        self.setFixedSize(self.width(), self.height())

        # singal slot register
        self.singalHander()

    def singalHander(self):
        # singal process
        logging.info("singal process,register singal slot.")
        QtCore.QObject.connect(self.LoadTrainDataButton, QtCore.SIGNAL('clicked()'), self.openTreatImg)
        QtCore.QObject.connect(self.menuFile, QtCore.SIGNAL('clicked()'), self.openTreatImg)

    def openTreatImg(self):
        self._openImg(self.imgVB)

    def openGroundTruthImg(self):
        self._openImg(self.groundImgVB)

    def openResultImg(self):
        self._openImg(self.segVB)

    def _openImg(self, viewbox):
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file'))
        if filename:
            logging.info("open img file path:{}".format(filename))
            img3D = loadMat(filename, tmpname)
            img2D = get2DSlice(img3D, 13)
            showImge(viewbox, img2D)
            return img3D


def showImge(viewbox, img):
    if isinstance(viewbox, pg.ViewBox):
        showImg = pg.ImageItem(img)
        viewbox.addItem(showImg)
        viewbox.autoRange()
    else:
        raise TypeError("viewbox arg type is error")


def get2DSlice(img, slice):
    if img.ndim == 3 and isinstance(img, np.ndarray):
        return img[:, :, slice]
    else:
        raise TypeError("img type is error")


if __name__ == '__main__':

    # load logging
    import logging.config

    con_path = r"H:\ProSegCode\src\main\logging.ini"
    if os.path.isfile(con_path):
        logging.config.fileConfig(con_path)

    # setup app
    logging.info("start app")
    app = QtGui.QApplication(sys.argv)
    ui = Window()
    ui.show()
    logging.info("end app")
    sys.exit(app.exec_())
