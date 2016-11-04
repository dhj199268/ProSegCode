# encoding:utf-8
from PyQt4 import QtGui, QtCore
from UI.MainWindow import Ui_MainWindow
import logging
import sys, os
import pyqtgraph as pg
import numpy as np
from Prostate.Until.Tool import loadMat, saveMat
from Prostate.System import LocalSystem
from Prostate.Core import TrainData

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

tmpname = "temp_data"
suffix = "_pro.mat"


class Window(QtGui.QMainWindow, Ui_MainWindow):
    groundImg = None
    segImg = None
    treatsImg = None
    model = None
    trainDataPath = None
    system = LocalSystem()

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

        ## 设置edit
        self.xEdit.setEnabled(False)
        self.yEdit.setEnabled(False)
        self.zEdit.setEnabled(False)
        self.iterEdit.setEnabled(False)
        self.treenumEdit.setEnabled(False)
        self.depthEdit.setEnabled(False)
        self.sliceEdit.setEnabled(False)
        self.sampleEdit.setEnabled(False)

        ## 设置 参数 slider
        self.setWidgetEnabled(False)

        self.IterSlider.setRange(1, 6)
        self.TreeSlider.setRange(100, 10000)
        self.depthSlider.setRange(10, 30)
        self.sampleSlider.setRange(1000, 10000)

        self.SliceSlider.setSingleStep(1)
        self.IterSlider.setSingleStep(1)
        self.TreeSlider.setSingleStep(1)
        self.depthSlider.setSingleStep(1)
        self.sampleSlider.setSingleStep(1)

        self.iterEdit.setText("1")
        self.treenumEdit.setText("100")
        self.depthEdit.setText("10")
        self.sliceEdit.setText("1")
        self.sampleEdit.setText("1000")

        # singal slot register
        self.singalRegister()

    def singalRegister(self):
        # singal process
        logging.info("singal process,register singal slot.")

        QtCore.QObject.connect(self.SegmentButton, QtCore.SIGNAL('clicked()'), self.segment)
        QtCore.QObject.connect(self.TrainingButton, QtCore.SIGNAL('clicked()'), self.trainning)
        QtCore.QObject.connect(self.LoadTrainDataButton, QtCore.SIGNAL('clicked()'), self.getTrainDataPaths)

        QtCore.QObject.connect(self.actionTreatment_Img, QtCore.SIGNAL('triggered()'), self.openTreatImg)
        QtCore.QObject.connect(self.actionGround_Img, QtCore.SIGNAL('triggered()'), self.openGroundTruthImg)
        QtCore.QObject.connect(self.actionModel, QtCore.SIGNAL('triggered()'), self.loadModel)
        QtCore.QObject.connect(self.actionSave_Model, QtCore.SIGNAL('triggered()'), self.saveModel)
        QtCore.QObject.connect(self.actionResult, QtCore.SIGNAL('triggered()'), self.saveResult)

        QtCore.QObject.connect(self.SliceSlider, QtCore.SIGNAL('valueChanged(int)'), self,
                               QtCore.SLOT('sliceShow(int)'))
        QtCore.QObject.connect(self.SliceSlider, QtCore.SIGNAL('valueChanged(int)'), self,
                               QtCore.SLOT('sliceEditShow(int)'))
        QtCore.QObject.connect(self.IterSlider, QtCore.SIGNAL('valueChanged(int)'), self,
                               QtCore.SLOT('iterEditShow(int)'))
        QtCore.QObject.connect(self.TreeSlider, QtCore.SIGNAL('valueChanged(int)'), self,
                               QtCore.SLOT('treeNumEditShow(int)'))
        QtCore.QObject.connect(self.depthSlider, QtCore.SIGNAL('valueChanged(int)'), self,
                               QtCore.SLOT('depthEditShow(int)'))
        QtCore.QObject.connect(self.sampleSlider, QtCore.SIGNAL('valueChanged(int)'), self,
                               QtCore.SLOT('sampleEditShow(int)'))

    @QtCore.pyqtSlot()
    def segment(self):
        logging.info("segment img")

        if self.model is not None and self.treatsImg is not None:
            segData = TrainData((self.treatsImg, None))
            self.segImg = self.system.setModels(self.model).predict(segData)

            slice = self.SliceSlider.value()

            if self.segImg is not None:
                img2D = get2DSlice(self.segImg, slice)
                showImge(self.segVB, img2D)
        else:
            QtGui.QMessageBox.critical(self, "Critical", _fromUtf8("没有训练图像或可用的模型"))
        logging.info("segment over")

    @QtCore.pyqtSlot()
    def trainning(self):
        logging.info("train model")
        iterNum, treeNum, depth, sample, boostrap = self.getParams()

        logging.debug(
            "iter:{}  tree num:{} depth:{} sample:{} boostrap:{}".format(iterNum, treeNum, depth, sample,
                                                                         boostrap))
        trainDatas = list()
        for path in self.trainDataPath:
            img = loadMat(path[0], tmpname)
            groundImg = loadMat(path[1], tmpname)
            trainDatas.append(TrainData((img, groundImg), sample))
        try:
            self.system.setIter(iterNum).setMaxDepth(depth).setTreeNum(treeNum)
            self.system.train(trainDatas)
            self.model = self.system.getModels()
        except Exception, e:
            logging.error(e.message)

        logging.info("train over")

    @QtCore.pyqtSlot()
    def loadModel(self):
        filename = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file'))

        if os.path.isfile(filename) and ".pkl" in filename:
            self.model = LocalSystem.loadModel(filename)
        else:
            QtGui.QMessageBox.critical(self, "Critical", _fromUtf8("model 文件无法解析"))

    @QtCore.pyqtSlot()
    def saveModel(self):
        logging.info("save model")

        if self.model is not None:
            filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Open file'))
            LocalSystem.saveModel(self.model, filename)
        else:
            QtGui.QMessageBox.critical(self, "Critical", _fromUtf8("没有模型可以保存"))

    @QtCore.pyqtSlot()
    def saveResult(self):
        logging.info("save Img")
        if self.segImg is not None:
            filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Open file'))
            saveMat(filename, self.segImg, tmpname)
        else:
            QtGui.QMessageBox.critical(self, "Critical", _fromUtf8("没有结果可以保存"))

    @QtCore.pyqtSlot(int)
    def sampleEditShow(self, num):
        self.sampleEdit.setText(str(num))

    @QtCore.pyqtSlot(int)
    def sliceEditShow(self, num):
        self.sliceEdit.setText(str(num + 1))

    @QtCore.pyqtSlot(int)
    def iterEditShow(self, num):
        self.iterEdit.setText(str(num))

    @QtCore.pyqtSlot(int)
    def treeNumEditShow(self, num):
        self.treenumEdit.setText(str(num))

    @QtCore.pyqtSlot(int)
    def depthEditShow(self, num):
        self.depthEdit.setText(str(num))

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
        self.sampleSlider.setEnabled(bool)

    @QtCore.pyqtSlot()
    def getTrainDataPaths(self):
        logging.info("load train data paths")
        try:
            father, trainDataSuffix = self._getTrainDataPaths()
            logging.debug("dir path :{} suffixs:{}".format(father, trainDataSuffix))
        except IOError, ie:
            QtGui.QMessageBox.critical(self, "Critical", ie.message)
            logging.error(ie.message)
            return None

        if father is not None and trainDataSuffix is not None:
            self.trainDataInfoEdit.clear()
            self.trainDataInfoEdit.append("dir :{}".format(father))

            self.trainDataPath = list()
            for suffix in trainDataSuffix:
                train = os.path.join(father, suffix[0])
                truth = os.path.join(father, suffix[1])
                self.trainDataPath.append((train, truth))
                self.trainDataInfoEdit.append("Img :{}      Ground Truth Img:{}".format(suffix[0], suffix[1]))

    def _getTrainDataPaths(self):

        ## filter "_pro"
        def suffixFilter(path):
            if "_pro" not in path:
                return path

        files = QtGui.QFileDialog.getOpenFileNames(self, 'Open file')

        ## filter path include "_pro"
        files = filter(suffixFilter, files)
        trainDataPath = list()
        father = None
        for f in files:
            f = str(f)
            if os.path.isfile(f) and "mat" in f:

                treatsImg = os.path.basename(f)
                father = os.path.dirname(f)
                groundImg = treatsImg.split(".")[0] + suffix

                if os.path.isfile(os.path.join(father, groundImg)):
                    trainDataPath.append((treatsImg, groundImg))
                else:
                    raise IOError(_fromUtf8("Img：{} 无法解析".format(groundImg)))
            else:
                raise IOError(_fromUtf8("Img：{} 无法解析".format(f)))

        return father, trainDataPath

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

    def getParams(self):
        iterNum = self.IterSlider.value()
        treeNum = self.TreeSlider.value()
        depth = self.depthSlider.value()
        boostrap = self.BootstrapButton.isChecked()
        sample = self.sampleSlider.value()
        return iterNum, treeNum, depth, sample, boostrap


def showImge(viewbox, img):
    if isinstance(viewbox, pg.ViewBox):
        showImg = pg.ImageItem(img)
        viewbox.addItem(showImg)
        viewbox.autoRange()
    else:
        raise TypeError("viewbox arg type is error")


# def saveImge(img, fliename):
#     if isinstance(img, np.ndarray):
#         saveMat(fliename, img, tmpname)
#     else:
#         raise TypeError("viewbox arg type is error")


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
