from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5 import uic
import numpy as np
import sys
import os
try:
    from cvwidget import VideoCapture, ContentType
except Exception:
    from .cvwidget import VideoCapture, ContentType


class MainWindow(QWidget):
    changeIndex = pyqtSignal(int)
    changeOption = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.multibled = False
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(CURRENT_DIR)
        uic.loadUi(os.path.join(CURRENT_DIR, "../ui/display.ui"), self)
        self.initVC()
        self.setUI()

    def setUI(self):
        self.cv_raw.type_name = ContentType.raw
        self.cv_intermediate.type_name = ContentType.mid
        self.cv_product.type_name = ContentType.res
        self.cap.change_pixmap_signal.connect(self.cv_raw.labelUpdate)
        self.cap.change_pixmap_signal.connect(self.cv_intermediate.labelUpdate)
        self.cap.change_pixmap_signal.connect(self.cv_product.labelUpdate)
        self.btn_savedata.released.connect(self.cap.saveData)
        self.btn_savesvm.released.connect(self.cap.saveSVM)
        self.btn_train.released.connect(self.cap.trainSVM)
        self.cap.change_display_freeze.connect(self.btn_adddata.setEnabled)

    def initVC(self):
        self.cap = VideoCapture()
        self.cap.content_width = 500
        self.cap.content_height = 300
        if self.cap.isOpened():
            self.cap.start()

    @pyqtSlot(bool)
    def toggleBackground(self, option):
        self.cap.background_subtraction = option

    @pyqtSlot()
    def setDataAction(self):
        self.cap.action_add_data = True

    @pyqtSlot()
    def setLabelName(self):
        self.cap.label_name = self.label_line.text()

    @pyqtSlot()
    def displayFreeze(self):
        self.cap.display_freeze = False if self.cap.display_freeze else True
        self.btn_adddata.setEnabled(self.cap.display_freeze)

    @pyqtSlot()
    def updateValue(self):
        low = np.array([int(self.hmin.text()), int(self.smin.text()), int(self.vmin.text())])
        high = np.array([int(self.hmax.text()), int(self.smax.text()), int(self.vmax.text())])
        self.cap.hand.low = low
        self.cap.hand.high = high

    def closeEvent(self, event):
        self.cap.stop()
        event.accept()


def runWindow():
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    runWindow()
