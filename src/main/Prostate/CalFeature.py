# encoding:utf-8
import numpy as np

def calContextFeature(img, sample_pos):
    r = np.array([[4, 5, 6, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 80]])
    w = np.array([list(xrange(0, 8))])
    w = w.T
    w = np.pi * 0.25 * w

    alph = np.floor(np.cos(w) * r + 0.5)
    beta = np.floor(np.sin(w) * r + 0.5)
    alph = alph.reshape(alph.size, 1, order='F').astype(np.int)
    beta = beta.reshape(beta.size, 1, order='F').astype(np.int)
    pos = np.column_stack((alph, beta))
    pos = np.row_stack(([[0, 0]], pos))

    img_shape = img.shape
    max_r = r.max()
    x_fill = np.zeros((max_r, img_shape[1], img_shape[2]))
    y_fill = np.zeros((max_r * 2 + img_shape[0], max_r, img_shape[2]))

    # img_tmp = np.row_stack((x_fill, img, x_fill))
    # img_tmp = np.column_stack((y_fill, img_tmp, y_fill))
    img_tmp = np.concatenate((x_fill, img, x_fill), axis=0)
    img_tmp = np.concatenate((y_fill, img_tmp, y_fill), axis=1)

    for i in xrange(0, len(pos)):
        x = pos[i, 0]
        y = pos[i, 1]
        extract_pos = (x + sample_pos[0] + max_r, y + sample_pos[1] + max_r, sample_pos[2])
        # data=data.reshape(len(pos),0)
        if i == 0:
            feature = img_tmp[extract_pos]
        else:
            feature = np.column_stack((feature, img_tmp[extract_pos]))
    # featur=map(lambda x:(x + sample_pos[0] + max_r, y + sample_pos[1] + max_r, sample_pos[2]),pos)
    return feature


def calHaarFeature(img, sample_pos):
    haar_pattern = list()
    scale_haar = [8, 10, 9, 4, 5, 3]
    # cal pattern
    for i in xrange(1, scale_haar[0] + 1):
        tmp = np.array([[1, 1, 0], [-1, -(i + 1), -1], [0, 0, 1], [-1, 0, -1], [0, (i + 1), 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-(i + 1), -1, -1], [0, 0, 1], [0, -1, -1], [(i + 1), 0, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-(i + 1), -1, -1], [(i + 1), 0, 1], [-1, -(i + 1), -1], [0, (i + 1), 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-1, -1, -1], [0, 0, 1], [-(i + 1), -1, -1], [-i, 0, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-1, -1, -1], [0, 0, 1], [i, -1, -1], [(i + 1), 0, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-1, -1, -1], [0, 0, 1], [-1, -(i + 1), -1], [0, -i, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-1, -1, -1], [0, 0, 1], [-1, i, -1], [0, i + 1, 1]])
        haar_pattern.append(tmp)
    for i in xrange(1, scale_haar[1] + 1):
        tmp = np.array([[1, 0, 0], [-i, -i, -1], [i, i, 1]])
        haar_pattern.append(tmp)
        tmp = np.array(
            [[2, 2, 0], [-i, -i, -1], [0, 0, 1], [0, 0, -1], [i, i, 1], [0, -i, -1], [i, 0, 1], [-i, 0, -1],
             [0, i, 1]])
        haar_pattern.append(tmp)
    for i in xrange(1, scale_haar[2] + 1):
        tmp = np.array([[1, 1, 0], [-i, -i, -1],
                        [i, i, 1], [-10, -10, -1], [10, 10, 1]])
        haar_pattern.append(tmp)
    for i in xrange(1, scale_haar[3] + 1):
        tmp = np.array([[1, 1, 0], [-2 * i, -i, -1], [2 * i, i, 1], [-10, -5, -1], [10, 5, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-i, -2 * i, -1], [i, 2 * i, 1], [-5, - 10, - 1], [5, 10, 1]])
        haar_pattern.append(tmp)
    for i in xrange(1, scale_haar[4] + 1):
        tmp = np.array([[1, 1, 0], [-i, -2 * i, -1], [i, 2 * i, 1], [-5, -10, -1], [5, 10, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-i, - 2 * i, -1], [0, 2 * i, 1], [0, -2 * i, -1], [i, 2 * i, 1]])
        haar_pattern.append(tmp)
    for i in xrange(1, scale_haar[5] + 1):
        tmp = np.array(
            [[1, 2, 0], [-2 * i, -i, -1], [2 * i, i, 1], [-2 * i, -3 * i, -1], [2 * i, -i, 1], [-2 * i, i, -1],
             [2 * i, 3 * i, 1]])
        haar_pattern.append(tmp)
        tmp = np.array(
            [[1, 2, 0], [-i, -2 * i, -1], [i, 2 * i, 1], [-3 * i, -2 * i, -1], [-i, 2 * i, 1], [i, -2 * i, -1],
             [3 * i, 2 * i, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-2 * i, -2 * i, -1], [-i, 2 * i, 1], [i, - 2 * i, - 1], [2 * i, 2 * i, 1]])
        haar_pattern.append(tmp)
        tmp = np.array([[1, 1, 0], [-2 * i, -2 * i, -1], [2 * i, -i, 1], [-2 * i, i, -1], [2 * i, 2 * i, 1]])
        haar_pattern.append(tmp)

    # init params
    num_pix_patch = 21 * 21
    num_haar_features = len(haar_pattern)
    size_img = img.shape
    # cal cumsum picture
    cum_img = np.cumsum(img, axis=0)
    cum_img = np.cumsum(cum_img, axis=1)
    cum_img = np.cumsum(cum_img, axis=2)
    # cum_img = cum_img.astype(np.float)

    # limit range about postition,inner funciton
    def limitRange(data):
        data[data < 0] = 0
        for i in xrange(0, len(size_img)):
            # print i
            tmp = data[i, :]
            tmp[tmp > size_img[i] - 1] = size_img[i] - 1
        return tuple(data)

    # cal haar value from cum img,inner funciton
    def getValue(tup):
        assert len(tup) == 8, "input len is error"
        # print np.max(tup[0][0])
        # print np.max(tup[0][1])
        # print np.max(tup[0][2])
        # tmp=tup[0]
        tmp = cum_img[tup[0]]
        for i in xrange(1, len(tup) / 2 + 1):
            tmp = tmp + cum_img[tup[i]]
        for i in xrange(len(tup) / 2 + 1, len(tup)):
            tmp = tmp - cum_img[tup[i]]
        tmp = tmp.astype(np.double)
        return tmp

    assert sample_pos != None, "not sample pos"
    # cal haar feature
    for k in xrange(0, num_haar_features):
        # get postive dot
        pattern = haar_pattern[k]
        for j in xrange(1, pattern[0, 0] + 1):
            left_top_high = np.array([sample_pos[0] + pattern[j * 2 - 1, 0], sample_pos[1] + pattern[j * 2, 1],
                                      sample_pos[2] + pattern[j * 2 - 1, 2]])
            left_top_high = limitRange(left_top_high)

            left_top_low = np.array([sample_pos[0] + pattern[j * 2 - 1, 0], sample_pos[1] + pattern[j * 2, 1],
                                     sample_pos[2] + pattern[j * 2, 2]])
            left_top_low = limitRange(left_top_low)

            right_bottom_high = np.array([sample_pos[0] + pattern[j * 2, 0], sample_pos[1] + pattern[j * 2 - 1, 1],
                                          sample_pos[2] + pattern[j * 2 - 1, 2]])
            right_bottom_high = limitRange(right_bottom_high)

            right_bottom_low = np.array([sample_pos[0] + pattern[j * 2, 0], sample_pos[1] + pattern[j * 2 - 1, 1],
                                         sample_pos[2] + pattern[j * 2 - 1, 2]])
            right_bottom_low = limitRange(right_bottom_low)

            left_bottom_high = np.array([sample_pos[0] + pattern[j * 2 - 1, 0], sample_pos[1] + pattern[j * 2 - 1, 1],
                                         sample_pos[2] + pattern[j * 2 - 1, 2]])
            left_bottom_high = limitRange(left_bottom_high)

            left_bottom_low = np.array([sample_pos[0] + pattern[j * 2 - 1, 0], sample_pos[1] + pattern[j * 2 - 1, 1],
                                        sample_pos[2] + pattern[j * 2, 2]])
            left_bottom_low = limitRange(left_bottom_low)

            right_top_high = np.array([sample_pos[0] + pattern[j * 2, 0], sample_pos[1] + pattern[j * 2, 1],
                                       sample_pos[2] + pattern[j * 2 - 1, 2]])
            right_top_high = limitRange(right_top_high)

            right_top_low = np.array([sample_pos[0] + pattern[j * 2, 0], sample_pos[1] + pattern[j * 2, 1],
                                      sample_pos[2] + pattern[j * 2, 2]])
            right_top_low = limitRange(right_top_low)

            tup = (right_top_low, left_bottom_low, right_bottom_high, left_top_high, left_top_low, right_bottom_low,
                   right_top_high, left_bottom_high)
            if j == 1:
                haar_positive = getValue(tup)
            else:
                haar_positive = haar_positive + getValue(tup)
                # haar_positive = haar_positive.astype(np.int32)
        for j in xrange(1, pattern[0, 1] + 1):
            t = pattern[0, 0] * 2
            left_top_high = np.array([sample_pos[0] + pattern[j * 2 - 1 + t, 0], sample_pos[1] + pattern[j * 2 + t, 1],
                                      sample_pos[2] + pattern[j * 2 - 1 + t, 2]])
            left_top_high = limitRange(left_top_high)

            left_top_low = np.array([sample_pos[0] + pattern[j * 2 - 1 + t, 0], sample_pos[1] + pattern[j * 2 + t, 1],
                                     sample_pos[2] + pattern[j * 2 + t, 2]])
            left_top_low = limitRange(left_top_low)

            right_bottom_high = np.array(
                [sample_pos[0] + pattern[j * 2 + t, 0], sample_pos[1] + pattern[j * 2 - 1 + t, 1],
                 sample_pos[2] + pattern[j * 2 - 1 + t, 2]])
            right_bottom_high = limitRange(right_bottom_high)

            right_bottom_low = np.array(
                [sample_pos[0] + pattern[j * 2 + t, 0], sample_pos[1] + pattern[j * 2 - 1 + t, 1],
                 sample_pos[2] + pattern[j * 2 - 1 + t, 2]])
            right_bottom_low = limitRange(right_bottom_low)

            left_bottom_high = np.array(
                [sample_pos[0] + pattern[j * 2 - 1 + t, 0], sample_pos[1] + pattern[j * 2 - 1 + t, 1],
                 sample_pos[2] + pattern[j * 2 - 1 + t, 2]])
            left_bottom_high = limitRange(left_bottom_high)

            left_bottom_low = np.array(
                [sample_pos[0] + pattern[j * 2 - 1 + t, 0], sample_pos[1] + pattern[j * 2 - 1 + t, 1],
                 sample_pos[2] + pattern[j * 2 + t, 2]])
            left_bottom_low = limitRange(left_bottom_low)

            right_top_high = np.array([sample_pos[0] + pattern[j * 2 + t, 0], sample_pos[1] + pattern[j * 2 + t, 1],
                                       sample_pos[2] + pattern[j * 2 - 1 + t, 2]])
            right_top_high = limitRange(right_top_high)

            right_top_low = np.array([sample_pos[0] + pattern[j * 2 + t, 0], sample_pos[1] + pattern[j * 2 + t, 1],
                                      sample_pos[2] + pattern[j * 2 + t, 2]])
            right_top_low = limitRange(right_top_low)

            tup = (right_top_low, left_bottom_low, right_bottom_high, left_top_high, left_top_low, right_bottom_low,
                   right_top_high, left_bottom_high)
            if j == 1:
                haar_negative = getValue(tup)
            else:
                haar_negative = haar_negative + getValue(tup)
                # haar_negative = haar_negative.astype(np.int32)

        # tmp = haar_positive - haar_negative
        # tmp = tmp.astype(np.double)
        if k == 0:
            haar_feature = (haar_positive - haar_negative) / num_pix_patch
        else:
            feature_tmp = (haar_positive - haar_negative) / num_pix_patch
            haar_feature = np.column_stack((haar_feature, feature_tmp))

    return haar_feature


if __name__ == '__main__':

    str = "temp_data"
    filename = u"H:\DHJ\课题二\seg_data\pat1\img1.mat"
    print "test calContextFeature funciton"
    from Until import Tool

    img = Tool.loadMat(filename, str)
    #test cunm sum picture
    # data = np.cumsum(img, axis=0)
    # data = np.cumsum(data, axis=1)
    # data = np.cumsum(data, axis=2)
    # Tool.imShow3D(data, slice=1, axis=2)

    #test haar feature and context feature
    sample = np.where(img>1700)
    # sample = np.where(img>1700)
    from time import clock
    import  sys
    start = clock()
    data = calContextFeature(img, sample)
    stop = clock()
    print "context feature memory size:",sys.getsizeof(data)
    print "cal all pos context feature cost:", (stop - start)
    # Tool.imShow3D(data, 15)
    start = clock()
    data = calHaarFeature(img, sample)
    stop = clock()
    print "haar feature memory size:",sys.getsizeof(data)
    print "cal all pos haar feature cost:", (stop - start)
