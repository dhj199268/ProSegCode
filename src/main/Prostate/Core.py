# encoding:utf-8
# import multiprocessing
import os
import numpy as np
from Until import Tool
import CalFeature as cf
import threading
import logging


class TrainData(threading.Thread):
    __logger = logging.getLogger("TrainData")
    __context_haar = None
    __sec_context = None
    __pos_position = None
    __neg_position = None
    __ground_file = None
    __fun_context = None
    __fun_haar = None
    __nsmaple = None
    __img_file = None
    __img_size = None

    def __init__(self, file_tuple, samnple=8000, context_fun=cf.calContextFeature, haar_fun=cf.calHaarFeature):
        threading.Thread.__init__(self)
        self.__img_file = file_tuple[0]
        self.__ground_file = file_tuple[1]
        self.__nsmaple = samnple
        self.__fun_context = context_fun
        self.__fun_haar = haar_fun
        assert os.path.exists(self.__img_file), "img file no exists"
        assert os.path.exists(self.__ground_file), "ground img file no exists"

    def setSample(self, sample):
        self.__nsmaple = sample
        return self

    def setHaarFun(self, fun=cf.calHaarFeature):
        self.__fun_haar = fun
        return self

    def setContextFun(self, fun=cf.calContextFeature):
        self.__fun_context = fun
        return self

    def getTrainData(self):
        '''
        :return:train and label
        '''
        self.__logger.info("Get train data")
        # get pos and neg feature
        # pos_con_haar = self.__context_haar[self.__pos_position]
        # pos___sec_context = self.__sec_context[self.__pos_position]
        # neg_con_haar = self.__context_haar[self.__neg_position]
        # neg___sec_context = self.__sec_context[self.__neg_position]
        # pos_data = np.column_stack((pos___sec_context, pos_con_haar))
        # neg_data = np.column_stack((neg___sec_context, neg_con_haar))
        #
        # # reduce memory
        # del pos_con_haar
        # del pos___sec_context
        # del neg_con_haar
        # del neg___sec_context
        data = self.getPredict()
        pos_data = data[self.__pos_position]
        neg_data = data[self.__neg_position]
        del data
        # merge train and label
        train = np.row_stack((pos_data, neg_data))
        pos_label = np.ones(pos_data.shape[0], np.uint8)
        neg_label = np.zeros(neg_data.shape[0], np.uint8)
        label = np.concatenate((pos_label, neg_label))
        self.__logger.debug(
            "Train data  min :{}   max: {}   shape:{}".format(train.min(), train.max(), str(train.shape)))
        self.__logger.debug("Label min :{}   max:{}   shape:{}:".format(label.min(), label.max(), str(label.shape)))

        del pos_data
        del neg_data

        return train, label

    def __sample(self, img, ground_img):
        self.__logger.info("Sample pos and neg position")
        n_pos = self.__nsmaple * 0.6
        n_neg = self.__nsmaple * 0.4

        self.__pos_position = np.where(ground_img > 0)
        self.__neg_position = np.where(ground_img == 0)
        self.__pos_position = Tool.sampleTuple(self.__pos_position, n_pos)
        self.__neg_position = Tool.sampleTuple(self.__neg_position, n_neg)
        self.__pos_position = Tool.sub2Index(self.__img_size, self.__pos_position)
        self.__neg_position = Tool.sub2Index(self.__img_size, self.__neg_position)

        self.__logger.debug("Pos  position max:" + str(self.__pos_position.max()))
        self.__logger.debug("Pos  position shape:" + str(self.__pos_position.shape))
        self.__logger.debug("Neg  position max:" + str(self.__neg_position.max()))
        self.__logger.debug("Neg  position shape:" + str(self.__neg_position.shape))

    def getPredict(self):
        return np.column_stack((self.__sec_context, self.__context_haar))

    def updateSecFeature(self, pro_map):
        self.__logger.info("Update second feature data")
        size = self.__img_size
        pro_map = pro_map.reshape(size[0], size[1], size[2])

        # Tool.imShow3D(pro_map)

        all_pos = np.where(pro_map >= 0)
        self.__sec_context = self.__fun_context(pro_map, all_pos)
        self.__logger.debug(
            "Second feature min :{}   max: {}   shape:{}:".format(self.__sec_context.min(), self.__sec_context.max(),
                                                                  self.__sec_context.shape))

    def getImgSize(self):
        return self.__img_size

    def run(self):
        # load data
        self.__logger.info("Load img and groundtruth img")
        self.__logger.debug("Img file path:" + self.__img_file)
        self.__logger.debug("Groundtruth file path:" + self.__ground_file)
        str_name = "temp_data"
        img = Tool.loadMat(self.__img_file, str_name)
        ground_img = Tool.loadMat(self.__ground_file, str_name)
        self.__img_size = img.shape

        # sample data,get pos and neg position
        self.__sample(img, ground_img)

        # cal feature
        # cal haar feature
        self.__logger.info("Cal featrue")
        all_pos = np.where(img >= 0)
        haar = self.__fun_haar(img, all_pos)
        self.__logger.debug("haar feature min :{}   max: {}   shape:{}:".format(haar.min(), haar.max(), haar.shape))

        # cal context feature
        context = self.__fun_context(img, all_pos)
        self.__logger.debug(
            "context feature min :{}   max: {}   shape:{}:".format(context.min(), context.max(), context.shape))
        self.__context_haar = np.column_stack((context, haar))
        self.__sec_context = np.ones(context.shape, dtype=context.dtype) * 0.5

        del all_pos
        del haar
        del context

        # def getReuslt(self):
        #     size = self.__img_size
        #     return self.__sec_context.reshape(size[0], size[1], size[2])


if __name__ == '__main__':
    con_path = "conf.ini"
    thread_list = list()
    img_file = u"H:\DHJ\课题二\seg_data\pat1\img1.mat"
    ground_file = u"H:\DHJ\课题二\seg_data\pat1\img1_pro.mat"
    traindata = TrainData((img_file, ground_file))
    thread_list.append(traindata)
    # img_file = u"H:\DHJ\课题二\seg_data\pat1\img2.mat"
    # ground_file = u"H:\DHJ\课题二\seg_data\pat1\img2_pro.mat"
    # traindata = TrainData((img_file, ground_file))
    # thread_list.append(traindata)
    for thread in thread_list:
        thread.start()
    for thread in thread_list:
        thread.join()
    train, label = traindata.getTrainData()
    import sys

    print "traindata memory :", sys.getsizeof(traindata)
    print "train data"
