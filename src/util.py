import cv2 as cv
import numpy as np
import random
import os
try:
    import gesture
except Exception:
    from . import gesture

file_path = os.path.dirname(os.path.realpath(__file__))
rs_path = os.path.join(file_path, '../resource')


def genTrainSets(img_path, num):
    img = cv.imread(os.path.join(rs_path, img_path))
    img_dir = os.path.dirname(img_path)
    for i in range(num):
        angle = random.randrange(-90, 90)
        h, w = img.shape[:2]
        ma = cv.getRotationMatrix2D((w / 2, h / 2), angle, 0.8)
        res = cv.warpAffine(img, ma, (w, h))
        cv.imwrite(os.path.join(rs_path, img_dir + '/hand' + str(i) + '.png'), res)


def SvmTraining():
    tolabel = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
    segment = gesture.HandSegment()
    svm = gesture.GestureSVM()
    feature_data = {}
    traindir = os.path.join(rs_path, 'train')
    for dirname in os.listdir(traindir):
        imgdir = os.path.join(traindir, dirname)
        if os.path.isdir(imgdir):
            feature_data[tolabel[dirname]] = []
            for file in os.listdir(imgdir):
                segment.image = cv.imread(os.path.join(imgdir, file))
                segment.getContour(flag=1)
                segment.getfourier()
                if segment.feature_info.shape[0] != 31:
                    continue
                feature_data[tolabel[dirname]].append(segment.feature_info)
            if feature_data[tolabel[dirname]] == []:
                feature_data.pop(tolabel[dirname])
    svm.train(feature_data)
    print('Complete training')
    svm.save()


if __name__ == '__main__':
    # for i in range(10):
    #     genTrainSets('train/{}/{}.png'.format(i, i), 30)
    SvmTraining()

    # img = cv.imread(os.path.join(rs_path, 'train/5/hand0.png'))
    # segment = gesture.HandSegment()
    # segment.image = img
    # segment.getContour(flag=1)
    # segment.getfourier()
    # svm = gesture.GestureSVM()
    # svm.load()
    # # print(segment.feature_info)
    # res = svm.predict(segment.feature_info)
    # print(res)
