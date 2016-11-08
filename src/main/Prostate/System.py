# encoding:utf-8

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib
import logging

T = 0.8


class LocalSystem():
    def __init__(self, niter=3, ntree=500, maxdepth=15, bootstrap=True):
        self.__logger = logging
        self.__iter = niter
        self.__ntree = ntree
        self.__maxdepth = maxdepth
        self.__bootstrap = bootstrap
        self.__models = list()

    def isBootstrap(self, bootstrap):
        self.__bootstrap = True
        return self

    def setIter(self, niter):
        self.__iter = niter
        return self

    def setTreeNum(self, ntree):
        self.__ntree = ntree
        return self

    def setMaxDepth(self, maxdepth):
        self.__maxdepth = maxdepth
        return self

    def setModels(self, models):
        self.__models = models
        return self

    def getModels(self):
        return self.__models

    def _initModel(self):
        self.__logger.info("Init model,the num of model:" + str(self.__iter))

        if self.__iter > 0:
            for i in xrange(0, self.__iter):
                self.__models.append(RandomForestClassifier(n_estimators=self.__ntree, max_depth=self.__maxdepth,
                                                            bootstrap=self.__bootstrap, n_jobs=3))
        else:
            raise Exception("iter must mort than zero")

    def __calFeature(self, tradtas):
        for data in tradtas:
            data.start()
        for data in tradtas:
            data.join()

    def train(self, trdatas):
        # self.__logger.info("Train model" )
        self._initModel()

        # cal train data feature ans sample
        self.__calFeature(trdatas)
        self.__logger.info("Init feature")
        self.__logger.info("Train model")
        # for model in self.__models:
        for i in xrange(0, len(self.__models)):
            self.__logger.info("the time of iter train:" + str(i + 1))
            model = self.__models[i]
            self._train(trdatas, model)
            if i == len(self.__models) - 1:
                break
            self._predict(trdatas, model)

    def _train(self, trdata_list, model):
        # get train data
        traindata, label = trdata_list[0].getTrainData()
        for i in xrange(1, len(trdata_list)):
            train_tmp, labe_tmp = trdata_list[0].getTrainData()
            traindata = np.row_stack((traindata, train_tmp))
            label = np.concatenate((label, labe_tmp))

        self.__logger.debug("Train data:" + str(traindata))
        self.__logger.debug("Label data:" + str(label))

        model.fit(traindata, label)

    def _predict(self, trdata_list, model):
        for traindata in trdata_list:
            predict = traindata.getPredict()
            result = model.predict_proba(predict)
            traindata.updateSecFeature(result[:, 1])

    def predict(self, data):
        self.__logger.info("Predict data")
        data.start()
        data.join()
        for model in self.__models:
            tmp = data.getPredict()
            pro_map = model.predict_proba(tmp)
            if model == self.__models[len(self.__models) - 1]:
                break
            data.updateSecFeature(pro_map[:, 1])
        nsize = data.getImgSize()
        self.__logger.debug("Img size:" + str(nsize))
        self.__logger.debug("pro map size:" + str(pro_map.shape))
        result = pro_map[:, 1].reshape(nsize[0], nsize[1], nsize[2])
        result[result > T] = 1
        result[result < T] = 0
        return result

    def saveModel(self, filename):
        self.__logger.info("Save model:" + filename)
        LocalSystem.saveModel(self.__models, filename)

    def loadModel(self, filename):
        self.__logger.info("Load model:" + filename)
        self.__models = LocalSystem.loadModel(filename)

    @staticmethod
    def saveModel(models, filename):
        joblib.dump(models, filename, compress=3)

    @staticmethod
    def loadModel(filename):
        return joblib.load(filename)


def test():
    con_path = "logging.ini"
    import logging.config
    logging.config.fileConfig(con_path)
    from Core import TrainData
    modelfile = r"H:\ProSegCode\model.pkl"
    thread_list = list()
    img_file = u"H:\DHJ\课题二\seg_data\pat1\img1.mat"
    ground_file = u"H:\DHJ\课题二\seg_data\pat1\img1_pro.mat"
    traindata = TrainData((img_file, ground_file), samnple=1000)
    thread_list.append(traindata)

    # img_file = u"H:\DHJ\课题二\seg_data\pat1\img2.mat"
    # ground_file = u"H:\DHJ\课题二\seg_data\pat1\img2_pro.mat"
    # traindata = TrainData((img_file, ground_file), samnple=13000)
    # thread_list.append(traindata)
    #
    # img_file = u"H:\DHJ\课题二\seg_data\pat1\img3.mat"
    # ground_file = u"H:\DHJ\课题二\seg_data\pat1\img3_pro.mat"
    # traindata = TrainData((img_file, ground_file), samnple=13000)
    # thread_list.append(traindata)

    system = LocalSystem()
    system.train(thread_list)
    # system.saveModel(modelfile)
    # system.loadModel(modelfile)
    img_file = u"H:\DHJ\课题二\seg_data\pat1\img3.mat"
    ground_file = u"H:\DHJ\课题二\seg_data\pat1\img3_pro.mat"
    traindata = TrainData((img_file, ground_file))
    # traindata.start()
    # traindata.join()

    result = system.predict(traindata)
    from Until import Tool
    print"show img"
    Tool.imShow3D(result)


if __name__ == '__main__':
    test()
