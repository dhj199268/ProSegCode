# encoding:utf-8
import scipy.io as sio
import numpy as np


def sub2Index(size, sub_tup):
    assert len(size) == 3, "size is error"
    index_mat = np.arange(size[0] * size[1] * size[2])
    index_mat = index_mat.reshape(size)
    return index_mat[sub_tup]


def imShow3D(img, slice=15, axis=3):
    '''
    :param img:numpy array ,a 3D CT picture
    :param slice: the slice of img
    :param axis: axis about 3D picture ,can not more than num 3
    :return:void
    '''
    nsize = img.ndim
    assert axis <= nsize, "axis more than size of img"

    # get sclie data
    if axis == 1:
        imgslice = img[slice, :, :]
    elif axis == 2:
        imgslice = img[:, slice, :]
    elif axis == 3:
        imgslice = img[:, :, slice]

    # show img sclice
    import pylab as pl
    import matplotlib.pyplot as plt
    pl.imshow(imgslice, cmap=plt.cm.gray)
    plt.axis('off')
    pl.show()

def loadMat(file_name, mat_name):
    '''
    :param filename:matlab mat data flie path
    :param matname: name in mat data
    :return: a type of numpy array data
    '''
    data = sio.loadmat(file_name)
    return data[mat_name]


def saveMat(file_name, mat_data, mat_name):
    sio.savemat(file_name, {mat_name: mat_data})


def sampleTuple(tuple_data, nsample, repeat=False):
    '''
    :param tuple_data:tuple data,len(data)=3
    :param nsample: num of sample
    :param repeat: the sample can be repeat
    :return: tuple len(tuple)=3
    '''
    assert len(tuple_data) == 3
    import numpy as np
    # vstack data
    data = np.vstack(tuple_data)
    size = data.shape[1]
    if repeat == False:
        assert nsample <= size, "the num of sample too much"
    random_list = np.random.randint(0, size, nsample)
    return tuple(data[:, random_list])


if __name__ == '__main__':

    str = "temp_data"
    filename = u"H:\DHJ\课题二\seg_data\pat1\img1.mat"
    print "test loadMat funciton"
    img = loadMat(filename, str)
    size=img.shape
    # print"test imShow3D funciton"
    imShow3D(img, axis=3, slice=2)
    # print "test smapleTuple"
    # import numpy as np
    #
    # sub = np.where(img==0)
    # #sub = sampleTuple(sub, 1)
    # print sub
    # print img[sub]
    # # test sub2index
    # index = sub2Index(img.shape, sub)
    # img_other=img.reshape(img.size)
    # data=img_other[index]
    # print data.max()
    # print data.min()
