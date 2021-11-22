import cv2 as cv
import numpy as np
try:
    import gesture
except Exception:
    from . import gesture
# from matplotlib import pyplot as plt


# 滤波去噪
def blur(img, size, sigma):
    return cv.GaussianBlur(img, (size, size), sigma)


def avgColor(img):
    avg = [cv.equalizeHist(channel) for channel in cv.split(img)]
    avgGray = sum(avg) / 3

    imgAvg = img.copy()
    h, w, c = img.shape
    print(w, h, c)
    for i in range(c):
        for x in range(w):
            for y in range(h):
                imgAvg[i][x][y] = avgGray
    return imgAvg


def equalize(img):
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    channels = cv.split(img_ycrcb)
    channels[0] = cv.equalizeHist(channels[0])
    img_avg = cv.merge(channels)
    res = cv.cvtColor(img_avg, cv.COLOR_YCrCb2BGR)

    return res


def getSkin(img):
    # imgAvg = equalize(img)

    # imgBlur = blur(imgAvg, 3, 0)

    skincrcbHist = np.zeros((256, 256), np.uint8)
    cv.ellipse(skincrcbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)

    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    y, cr, cb = cv.split(img_ycrcb)
    w, h = img.shape[:2]
    mask = np.zeros((w, h), np.uint8)

    for x in range(w):
        for y in range(h):
            if skincrcbHist[cr[x][y], cb[x][y]] > 0:
                mask[x][y] = 255

    res = cv.bitwise_and(img, img, mask=mask)
    return res


def getSkin_otsu(img):
    imgAvg = equalize(img)
    img_ycrcb = cv.cvtColor(imgAvg, cv.COLOR_BGR2YCrCb)
    cr = cv.split(img_ycrcb)[1]
    cr = blur(cr, 5, 0)
    ret, mask = cv.threshold(cr, 25, 255, cv.THRESH_OTSU)

    res = cv.bitwise_and(img, img, mask=mask)
    return res


def getSkin_hsv(img, low=np.array([0, 48, 50]), high=np.array([20, 170, 255])):
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask = cv.inRange(img_hsv, low, high)
    mask = handProcess(mask, 3, 3)
    mask = blur(mask, 5, 90)

    res = cv.bitwise_and(img, img, mask=mask)
    return res


def getSkin_ycbcr(img, low=np.array([0, 114, 79]), high=np.array([255, 123, 108])):
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    mask = cv.inRange(img_ycrcb, low, high)
    mask = handProcess(mask, 3, 3)
    mask = blur(mask, 5, 90)

    res = cv.bitwise_and(img, img, mask=mask)
    return res


def handProcess(img, ksize, iterations):
    kernel = np.ones((ksize, ksize), np.uint8)
    img_erode = cv.dilate(img, kernel, iterations=iterations)
    res = cv.erode(img_erode, kernel, iterations=iterations)
    return res


def threshold(img):
    imgblur = blur(img, 15, 0)
    gray = cv.cvtColor(imgblur, cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(gray, 30, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    return mask


def getMask(img, low=np.array([10, 130, 90]), high=np.array([255, 175, 125])):
    img_ycrcb = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    mask = cv.inRange(img_ycrcb, low, high)
    mask = blur(mask, 5, 90)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, (3, 3), iterations=3)

    return mask


def drawContours(img):
    imgCanny = cv.Canny(img, 100, 100)
    contours, hierarchy = cv.findContours(imgCanny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    res = np.zeros_like(img)
    for cnt in contours:
        length = cv.arcLength(cnt, False)
        area = cv.contourArea(cnt, False)
        if length > 20 and area > 50:
            cv.drawContours(res, cnt, -1, (255, 255, 255), 2)

    return res


def fourier(cnt):
    f = np.fft.fft2(cnt)
    des = np.abs(f)
    res = []
    for term in range(1, len(des)):
        res.append(term / des[0])
    return res


def svmInit():
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)


def kmeans(img, k, channel_num):
    img1 = np.float32(img.reshape((-1, channel_num)))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1)
    flags = cv.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv.kmeans(img1, k, None, criteria, 10, flags)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    res = res.reshape((img.shape))

    return res


def updateData(x):
    pass


if __name__ == '__main__':
    img = cv.imread('resource/train/0.png')
    cv.imshow('img', img)

    low = np.array([0, 48, 50])
    high = np.array([20, 170, 255])

    imgMask = getMask(img)
    cv.imshow('imgSkin', imgMask)

    # mask = cv.morphologyEx(imgMask, cv.MORPH_OPEN, (5, 5), iterations=3)
    # cv.imshow('imgMask', mask)

    imgSkin = getSkin_hsv(img, low, high)
    cv.namedWindow('imgb')
    cv.createTrackbar("bar_y", 'imgb', 0, 255, updateData)
    cv.createTrackbar("bar_cb", 'imgb', 0, 255, updateData)
    cv.createTrackbar("bar_cr", 'imgb', 0, 255, updateData)
    cv.createTrackbar("bar_ym", 'imgb', 0, 255, updateData)
    cv.createTrackbar("bar_cbm", 'imgb', 0, 255, updateData)
    cv.createTrackbar("bar_crm", 'imgb', 0, 255, updateData)

    # imgSkin = cv.bitwise_and(img, img, mask=mask)

    # img_contour = drawContours(imgSkin, img)
    # cv.imshow('Result', img_contour)
    while True:
        y = cv.getTrackbarPos("bar_y", 'imgb')
        cb = cv.getTrackbarPos("bar_cb", 'imgb')
        cr = cv.getTrackbarPos("bar_cr", 'imgb')
        ym = cv.getTrackbarPos("bar_ym", 'imgb')
        cbm = cv.getTrackbarPos("bar_cbm", 'imgb')
        crm = cv.getTrackbarPos("bar_crm", 'imgb')
        low = np.array([y, cb, cr])
        high = np.array([ym, cbm, crm])
        hand = gesture.HandSegment(img)
        hand.low = low
        hand.high = high
        imgSkin = hand.getMaskHSV()
        cv.imshow('imgb', imgSkin)
        key = cv.waitKey(1)
        if key & 0xff == ord('q'):
            break
        elif key & 0xff == ord('s'):
            print("ok")
            cv.imwrite('resource/train/0/0.png', imgSkin)
