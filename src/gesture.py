import cv2 as cv
import numpy as np
from enum import Enum
import json
import time
import os


class CntStyle(Enum):
    cnt = 'cnt'
    rect = 'rect'
    aprect = 'aprect'


class HandSegment():
    def __init__(self, img=np.zeros((100, 100), dtype=np.uint8)):
        """初始化变量"""
        # init args
        # self.low = np.array([0, 48, 50])
        # self.high = np.array([20, 170, 255])
        self.low = np.array([10, 125, 90])
        self.high = np.array([255, 175, 135])
        self.frame_max = 15
        self.data_width = 32

        # init img variable
        self.image = img
        self.image_without_bg = None
        self.image_hand_mask = None
        self.image_draw_cnt = None
        self.image_draw_res = None

        self.bg = None
        self.contour = None
        self.feature_info = None
        self.predict_res = None

    def bgSubtraction(self, count, weight=0.7):
        """去背景"""
        blur = cv.GaussianBlur(self.image, (5, 5), 0)
        gray = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        if count >= self.frame_max:
            diff = cv.absdiff(np.uint8(self.bg), gray)
            mask = cv.threshold(diff, 25, 255, cv.THRESH_BINARY)[1]
            res = cv.bitwise_and(self.image, self.image, mask=mask)
        else:
            if self.bg is None:
                self.bg = np.zeros_like(gray, dtype=np.float32)
            else:
                cv.accumulateWeighted(gray, self.bg, weight)
            res = None

        self.image_without_bg = res
        return res

    def getMaskHSV(self):
        """通过HSV色彩空间分离背景，得到手部mask"""
        img = self.image if self.image_without_bg is None else self.image_without_bg
        img = cv.cuda_GpuMat(img)
        img = cv.cuda.bilateralFilter(img, 20, 50, 50)
        blur = img.download()
        # blur = cv.GaussianBlur(img, (5, 5), 90)
        img_hsv = cv.cvtColor(blur, cv.COLOR_BGR2YCrCb)
        mask = cv.inRange(img_hsv, self.low, self.high)
        mask = cv.medianBlur(mask, 7)
        self.image_hand_mask = cv.erode(mask, (5, 5), iterations=3)
        self.image_hand_mask = cv.morphologyEx(self.image_hand_mask, cv.MORPH_CLOSE, (3, 3), iterations=3)
        # self.image_hand_mask = cv.dilate(self.image_hand_mask, (5, 5), iterations=1)

        return self.image_hand_mask

    def getContour(self, flag=None):
        """获得手部轮廓"""
        if flag is not None:
            self.image_hand_mask = self.image
        if self.image_hand_mask is None:
            return None
        canny = cv.Canny(self.image_hand_mask, 25, 200)
        contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contours = list(contours)
        if contours:
            contours.sort(key=lambda x: cv.arcLength(x, False), reverse=True)
            # if len(contours) >= 2:
            #     self.contour = np.concatenate(contours[0:2])
            # else:
            self.contour = contours[0]
        else:
            self.contour = None
        return self.contour

    def drawContour(self, style, color=(37, 177, 247), thick=2):
        """绘制手部轮廓"""
        img = self.image.copy()
        if self.contour is not None:
            if style == CntStyle.cnt:
                cnt = self.contour
                cv.drawContours(img, cnt, -1, color, thick)
            elif style == CntStyle.rect:
                rect = cv.boundingRect(self.contour)
                cv.rectangle(img, rect, color, thick)
            elif style == CntStyle.aprect:
                poly = cv.approxPolyDP(self.contour, 32, True)
                rect = cv.boundingRect(poly)
                cv.rectangle(img, rect, color, thick)
                # cv.polylines(img, [rect], True, color, thick)
        self.image_draw_cnt = img
        return self.image_draw_cnt

    def drawResult(self, font=cv.FONT_HERSHEY_SIMPLEX, scale=1, color=(37, 177, 247), thick=2):
        """绘制手部轮廓"""
        img = self.image.copy()
        if self.contour is not None:
            poly = cv.approxPolyDP(self.contour, 32, True)
            rect = cv.boundingRect(poly)
            if self.predict_res is not None:
                text = "result:" + self.predict_res
                text_w, text_h = cv.getTextSize(text, font, scale, thick)[0]
                pos = (rect[0], rect[1] - text_h)
                text_ws, text_hs = cv.getTextSize(text, font, scale - 0.3, thick)[0]
                cv.rectangle(img, pos, (pos[0] + text_w, pos[1] + text_h), color, -1)
                cv.putText(img, text, (rect[0] + (text_w - text_ws) // 2, rect[1] - (text_h - text_hs + thick) // 2),
                           font, scale - 0.3, (53, 57, 63), thick)
                cv.rectangle(img, pos, (pos[0] + text_w, pos[1] + text_h), color, thick)
            cv.rectangle(img, rect, color, thick)

        self.image_draw_res = img
        return self.image_draw_res

    def getfourier(self):
        """获得傅立叶描述子"""
        if self.contour is None:
            self.feature_info = None
            return None
        cnt = self.contour[:, 0, :]
        cnt_complex = np.empty(cnt.shape[:-1], dtype=complex)
        cnt_complex.real = cnt[:, 0]
        cnt_complex.imag = cnt[:, 1]
        f = np.fft.fft(cnt_complex)

        # 设置特征数据宽度
        fshift = np.fft.fftshift(f)
        if len(fshift) < self.data_width:
            self.feature_info = None
            return None
        cindex = len(fshift) // 2
        left, right = cindex - (self.data_width // 2), cindex + (self.data_width // 2)
        frr = np.fft.ifftshift(fshift[left:right])
        frr = np.abs(frr)

        res = []
        for index in range(1, len(frr)):
            res.append(frr[index] / frr[0])

        self.feature_info = np.array(res, dtype=np.float32)
        return self.feature_info


class GestureSVM():
    def __init__(self):
        self.svm = cv.ml.SVM_create()
        file_pwd = os.path.dirname(os.path.realpath(__file__))
        self.save_path = os.path.join(file_pwd, '../resource/svm_data')
        self.date = time.strftime('%Y%m%d%H%M%S')
        self.setParams()
        self.trainset = {}

    def setParams(self):
        """设置SVM参数"""
        self.svm.setType(cv.ml.SVM_NU_SVC)
        self.svm.setKernel(cv.ml.SVM_RBF)
        # self.svm.setGamma(0.1e-5)
        self.svm.setNu(0.2)
        # self.svm.setC(8)
        # self.svm.setTermCriteria((cv.TermCriteria_MAX_ITER, 1000, 0.001))

    def train(self, trainset):
        """训练"""
        datalist, labellist = [], []
        for label, data in trainset.items():
            labellist += [[label] for count in range(len(data))]
            datalist += data
        datas = np.stack(datalist)
        datas = np.float32(datas)
        labels = np.array(labellist)
        res = self.svm.train(datas, cv.ml.ROW_SAMPLE, labels)
        return res

    def predict(self, data):
        """预测"""
        label = self.svm.predict(np.array((data, )))[1]
        return label

    def saveTrainset(self):
        with open(os.path.join(self.save_path, 'trainset.json'), 'w') as fp:
            json.dump(self.trainset, fp)

    def addData(self, label, data):
        if self.trainset.get(label) is None:
            self.trainset[label] = []
        self.trainset[label].append(data)

    def loadTrainset(self):
        filename = 'trainset.json'
        with open(os.path.join(self.save_path, filename), 'r') as fp:
            self.trainset = json.load(fp)

    def loadModel(self):
        """加载模型"""
        self.svm = cv.ml.SVM_load(os.path.join(self.save_path, 'data.xml'))

    def saveModel(self):
        """保存SVM模型"""
        self.svm.save(os.path.join(self.save_path, 'data.xml'))
