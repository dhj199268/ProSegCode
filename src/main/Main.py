# encoding:utf-8
from PyQt4 import QtGui, QtCore
from UI.MainWindow import Ui_MainWindow
import logging
import sys, os
import pyqtgraph as pg
import numpy as np
from Prostate.Until.Tool import loadMat, saveMat

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

tmpname = "temp_data"


class Window(QtGui.QMainWindow, Ui_MainWindow):
    groundImg = None
    segImg = None
    treatsImg = None
    model = None

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

        ## 禁止使用
        self.SliceSlider.setEnabled(False)
        self.SliceSlider.setSingleStep(1)

        ## 设置edit
        self.xEdit.setEnabled(False)
        self.yEdit.setEnabled(False)
        self.zEdit.setEnabled(False)
        self.iterEdit.setEnabled(False)
        self.treenumEdit.setEnabled(False)
        self.depthEdit.setEnabled(False)
        self.sliceEdit.setEnabled(False)

        ## 设置 参数 slider
        self.IterSlider.setRange(1, 6)
        self.TreeSlider.setRange(100, 10000)
        self.depthSlider.setRange(10, 30)

        self.IterSlider.setSingleStep(1)
        self.TreeSlider.setSingleStep(1)
        self.depthSlider.setSingleStep(1)

        self.iterEdit.setText("1")
        self.treenumEdit.setText("100")
        self.depthEdit.setText("10")
        self.sliceEdit.setText("1")
        # singal slot register
        self.singalRegister()

    def singalRegister(self):
        # singal process
        logging.info("singal process,register singal slot.")
        QtCore.QObject.connect(self.LoadTrainDataButton, QtCore.SIGNAL('clicked()'), self.openTreatImg)
        QtCore.QObject.connect(self.actionOpen, QtCore.SIGNAL('triggered()'), self.openTreatImg)
        QtCore.QObject.connect(self.actionLoad_GroundImg, QtCore.SIGNAL('triggered()'), self.openGroundTruthImg)
        QtCore.QObject.connect(self.SliceSlider, QtCore.SIGNAL('valueChanged(int)'), self,
                               QtCore.SLOT('sliceShow(int)'))

    @QtCore.pyqtSlot(int)
    def sliceShow(self, slice):
        logging.info("show slice img")
        logging.debug("show img slice :".format(slice))
        ## show image
        if self.groundImg is not None:
            logging.debug("ground img is exist")
            img = get2DSlice(self.groundImg, slice)
            showImge(self.groundImgVB, img)

        if self.segImg is not None:
            logging.debug("seg img is exist")
            img = get2DSlice(self.segImg, slice)
            showImge(self.segVB, img)

        if self.treatsImg is not None:
            logging.debug("treat img is exist")
            img = get2DSlice(self.treatsImg, slice)
            showImge(self.imgVB, img)

    @QtCore.pyqtSlot()
    def openTreatImg(self):
        try:
            self.treatsImg = self._openImg(self.imgVB)
            self._infoAndEnable(self.treatsImg)
        except ValueError:
            QtGui.QMessageBox.critical(self, "Critical", _fromUtf8("请导入正确类型"))
            logging.error("fail load treats img.")

    @QtCore.pyqtSlot()
    def openGroundTruthImg(self):
        try:
            self.groundImg = self._openImg(self.groundImgVB)
            self._infoAndEnable(self.groundImg)
        except ValueError:
            QtGui.QMessageBox.critical(self, "Critical", _fromUtf8("请导入正确类型"))
            logging.error("fail load ground truth img.")

    @QtCore.pyqtSlot()
    def openSegImg(self):
        try:
            self.segImg = self._openImg(self.segVB)
            self._infoAndEnable(self.segImg)
        except ValueError:
            QtGui.QMessageBox.critical(self, "Critical", _fromUtf8("请导入正确类型"))
            logging.error("fail load set img.")

    def _infoAndEnable(self, img):
        if img is not None:
            self.setWidgetEnabled(True)
            self._setImgInfo(img)

    def setWidgetEnabled(self, bool):
        self.SliceSlider.setEnabled(bool)
        self.TreeSlider.setEnabled(bool)
        self.depthSlider.setEnabled(bool)
        self.IterSlider.setEnabled(bool)
        self.BootstrapButton.setEnabled(bool)

    def _openImg(self, viewbox):
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file'))
        slice = self.SliceSlider.value()
        if filename:
            logging.info("open img file path:{}".format(filename))
            img3D = loadMat(filename, tmpname)
            img2D = get2DSlice(img3D, slice)
            showImge(viewbox, img2D)
            return img3D

    def _setImgInfo(self, img):
        if isinstance(img, np.ndarray) and img.ndim == 3:
            dim = img.shape
            self.xEdit.setText(str(dim[0]))
            self.yEdit.setText(str(dim[1]))
            self.zEdit.setText(str(dim[2]))
            self.SliceSlider.setRange(0, dim[2] - 1)
        else:
            raise TypeError("viewbox arg type is error")


def showImge(viewbox, img):
    if isinstance(viewbox, pg.ViewBox):
        showImg = pg.ImageItem(img)
        viewbox.addItem(showImg)
        viewbox.autoRange()
    else:
        raise TypeError("viewbox arg type is error")


def saveImge(img, fliename):
    if isinstance(img, np.ndarray):
        saveMat(fliename, img, tmpname)
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
    sys.exit(app.exec_())
