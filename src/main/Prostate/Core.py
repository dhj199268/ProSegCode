# encoding:utf-8
# import multiprocessing
import os
import numpy as np
from Until import Tool
import CalFeature as cf


class TrainData():
    context_haar = None
    sec_context = None
    pos_position = None
    neg_position = None
    ground_file = None
    fun_context = None
    fun_haar = None
    nsmaple = None
    img_file = None
    img_size = None

    def __init__(self, file_tuple, samnple=8000, context_fun=cf.calContextFeature, haar_fun=cf.calHaarFeature):
        self.img_file = file_tuple[0]
        self.ground_file = file_tuple[1]
        self.nsmaple = samnple
        self.fun_context = context_fun
        self.fun_haar = haar_fun
        assert os.path.exists(self.img_file), "img file no exists"
        assert os.path.exists(self.ground_file), "img file no exists"

    def setSample(self, sample=8000):
        self.nsmaple = sample

    def setHaarFun(self, fun=cf.calHaarFeature):
        self.fun_haar = fun

    def setContextFun(self, fun=cf.calHaarFeature):
        self.fun_context = fun


    def __initFeature(self):
        # load data
        str = "temp_data"
        img = Tool.loadMat(self.img_file, str)
        ground_img = Tool.loadMat(self.ground_file, str)
        self.img_size = img.shape

        # sample data,get pos and neg position
        self.__sample(img, ground_img)
        self.pos_position = Tool.sub2Index(self.img_size, self.pos_position)
        self.neg_position = Tool.sub2Index(self.img_size, self.neg_position)

        # cal feature
        all_pos = np.where(img)
        haar = self.fun_haar(img, all_pos)
        context = self.fun_context(img, all_pos)
        self.context_haar = np.column_stack((context, haar))
        self.sec_context = np.ones(context.shape, dtype=context.dtype) * 0.5
        del all_pos
        del haar
        del context

    def getTrainData(self):
        '''

        :return:train and label
        '''
        # get pos and neg feature
        pos_con_haar = self.context_haar[self.pos_position]
        pos_sec_context = self.sec_context[self.pos_position]
        neg_con_haar = self.context_haar[self.neg_position]
        neg_sec_context = self.sec_context[self.neg_position]
        pos_data = np.column_stack((pos_sec_context, pos_con_haar))
        neg_data = np.column_stack((neg_sec_context, neg_con_haar))

        # reduce memory
        del pos_con_haar
        del pos_sec_context
        del neg_con_haar
        del neg_sec_context

        # merge train and label
        train = np.row_stack((pos_data, neg_data))
        pos_label = np.ones(pos_data.shape[0], np.uint8)
        neg_label = np.zeros(neg_data.shape[0], np.uint8)
        label = np.vstack((pos_label, neg_label))

        return train, label

    def __sample(self, img, ground_img):
        n_pos = self.nsmaple / 2
        n_neg = self.nsmaple / 2

        self.pos_position = np.where(ground_img > 0)
        self.neg_position = np.where(ground_img == 0)
        self.pos_position = Tool.sampleTuple(self.pos_position, n_pos)
        self.neg_position = Tool.sampleTuple(self.neg_position, n_neg)

    def getPredict(self):
        return np.column_stack((self.sec_context, self.context_haar))

    def updateSecFeature(self, pro_img):
        pro_img = pro_img.reshape(self.img_size)
        all_pos = np.where(pro_img)
        self.sec_context=self.fun_context(pro_img,all_pos)

    def run(self):
        self.__initFeature()


if __name__ == '__main__':
    img_file = u"H:\DHJ\课题二\seg_data\pat1\img1.mat"
    ground_file = u"H:\DHJ\课题二\seg_data\pat1\img1_pro.mat"
    traindata = TrainData((img_file, ground_file))
    traindata.run()
    train,label=traindata.getTrainData()
    import sys
    print "traindata memory :",sys.getsizeof(traindata)
    print "train data"
