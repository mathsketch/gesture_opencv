from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QRect, QSize, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QImage, QFont
import cv2 as cv
from enum import Enum
import time
try:
    import gesture
except Exception:
    from . import gesture


class ContentType(Enum):
    raw = 'raw'
    mid = 'mid'
    res = 'res'
    pre = 'predict'


class VideoCapture(QThread):
    change_pixmap_signal = pyqtSignal(dict)
    change_display_freeze = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._run = True
        self.content = {}

        self._initOption()
        self._initCap()

    def _initOption(self):
        self.fps = 30
        self.dps = 30
        self.width = 640
        self.height = 480
        self.content_width = 300
        self.content_height = 300
        self.display_freeze = False
        self.background_subtraction = True
        self.action_add_data = False
        self.label_name = ''

    def _initCap(self):
        self.cap = cv.VideoCapture(0)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

    def imgCrop(self, img):
        left = (self.width - self.content_width) // 2
        right = self.width - left
        top = (self.height - self.content_height) // 2
        bottom = self.height - top
        return img[top: bottom, left: right]

    def run(self):
        previous_time = time.time() * 1000
        self.hand = gesture.HandSegment()
        self.svm = gesture.GestureSVM()
        self.svm.loadModel()
        frame_count = 0
        bg_avg_count = 0
        while self._run:
            ret, img = self.cap.read()
            last_time = time.time() * 1000
            if ret and last_time - previous_time > 1000 / self.fps:
                previous_time = last_time
                img = self.imgCrop(img)
                img = cv.flip(img, 1)
                if not self.display_freeze:
                    self.hand.image = img
                else:
                    self.hand.getfourier()
                    if self.action_add_data:
                        if self.hand.feature_info is not None:
                            try:
                                label = int(self.label_name)
                            except Exception:
                                label = -1
                            self.svm.addData(label, self.hand.feature_info.tolist())
                        print({x: len(y) for x, y in self.svm.trainset.items()})
                        self.action_add_data = False
                        self.display_freeze = False
                        self.change_display_freeze.emit(False)
                    continue

                if self.background_subtraction:
                    ret = self.hand.bgSubtraction(bg_avg_count)
                    if ret is None:
                        bg_avg_count += 1
                else:
                    self.hand.bg = None
                    self.hand.image_without_bg = None
                    bg_avg_count = 0

                self.content[ContentType.raw] = self.hand.image if self.hand.image_without_bg is None else self.hand.image_without_bg
                self.content[ContentType.mid] = self.hand.getMaskHSV()
                self.hand.getContour()
                self.content[ContentType.res] = self.hand.drawContour(gesture.CntStyle.cnt)

                # 手势预测
                frame_count += 1
                if frame_count > self.fps / self.dps:
                    frame_count = 0
                    self.hand.getfourier()
                    if self.hand.feature_info is not None and len(self.hand.feature_info) == self.hand.data_width - 1:
                        res = self.svm.predict(self.hand.feature_info)
                        print(res)

                self.change_pixmap_signal.emit(self.content)
        self.cap.release()

    @pyqtSlot()
    def saveData(self):
        self.svm.saveTrainset()

    @pyqtSlot()
    def saveSVM(self):
        self.svm.saveModel()

    @pyqtSlot()
    def trainSVM(self):
        self.svm.train(self.svm.trainset)
        print("Complete training")

    def isOpened(self):
        return self.cap.isOpened()

    def stop(self):
        self._run = False
        self.wait()


class CvWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setOption()
        self.display_content = None

    def _setOption(self):
        self.factor = 1
        self.scale_factor = 1
        self.text_height = 20
        self.type_name = None
        self.hidden = False

    def initPainter(self, qp):
        font = QFont('FiraCode', 10)
        qp.setFont(font)

    @pyqtSlot(bool)
    def toggleDisplay(self, option):
        self.setHidden(not option)

    @pyqtSlot(dict)
    def labelUpdate(self, content):
        self.display_content = content[self.type_name] if self.type_name is not None else None
        self.update()

    def convertImage(self, img, size):
        rgb_image = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap(convert_to_Qt_format)
        return pixmap.scaled(size, Qt.KeepAspectRatio)

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        name, img = self.type_name.value, self.display_content
        if img is not None:
            height, width = img.shape[:2]
            factor = min(self.width() / width, (self.height() - self.text_height) / height)
            img_size = QSize(int(width * factor * self.scale_factor),
                             int(height * factor * self.scale_factor))
            img_pos = QPoint((self.width() - img_size.width()) // 2,
                             (self.height() - self.text_height - img_size.height()) // 2)
            qp.drawPixmap(img_pos, self.convertImage(img, img_size))
            rect = QRect(img_pos.x(), img_pos.y() + img_size.height(), img_size.width(), self.text_height)
            qp.drawText(rect, Qt.AlignCenter, name)
        qp.end()
        # w = x - self.spacing
        # self.setMinimumWidth(w if w > 0 else 0)
        # self.setMinimumHeight(w if w > 0 else 0)
