# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'LoginANdRegister.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
import os
import sys
from utils.record import RecordAudio

from infer_recognition import register, recognition, infer, load_audio_db, args, person_name
import os
import numpy as np
import numpy
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
from facial_recognition import face
class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()
        self.Worker2 = TrainWorker()


        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.CaptureFinished.connect(self.training)
        self.Worker2.CaptureFinishedWorker.connect(self.finishedTraining)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()
        sys.exit(0)
    def training(self, Name):
        print(Name)
        self.Worker2.start()
    def finishedTraining(self,msg):
        self.Worker2.stop()



class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    CaptureFinished=pyqtSignal(str)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        data_path = "./haarcascade_frontalface_alt.xml"
        classfier = cv2.CascadeClassifier(data_path)
        path_name=r"F:\PycharmProject\researchProject\Yue\facial_recognition\face_img\User"
        # os.mkdir("./face_img/User")
        color = (0, 255, 0)
        num=0
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # ???????????????????????????????????????
                # ???????????????1.2???2?????????????????????????????????????????????????????????
                faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if len(faceRects) > 0 and num<10:  # ??????0??????????????????
                    for faceRect in faceRects:  # ???????????????????????????
                        x, y, w, h = faceRect

                        # ???????????????????????????
                        img_name = '%s/%d.jpg ' % (path_name, num)
                        image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                        cv2.imwrite(img_name, image)
                        num += 1
                        # if num > 10:  # ????????????????????????????????????????????????
                        #     break

                        # ???????????????
                        cv2.rectangle(Image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                if num==10:
                    self.CaptureFinished.emit("Yes")
                    num=100
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                # cv2.imshow("Test", frame)
                c = cv2.waitKey(10)
                if c & 0xFF == ord('q'):
                    break
    def stop(self):
        self.ThreadActive = False
        self.quit()

class TrainWorker(QThread):
    CaptureFinishedWorker = pyqtSignal(str)


    def run(self):
        self.ThreadActive = True
        face.training_process()
        self.CaptureFinishedWorker.emit("Finished")

    def stop(self):
        self.ThreadActive = False
        self.quit()

class TestWorker(QThread):
    CaptureFinishedWorker = pyqtSignal(str)

    def __init__(self, myvar, parent=None):
        QThread.__init__(self, parent)
        self.myvar = myvar
    def run(self):
        self.ThreadActive = True
        pre=face.new_test_process(self.myvar)
        self.CaptureFinishedWorker.emit(str(pre))

    def stop(self):
        self.ThreadActive = False
        self.quit()

class MainWindow2(QWidget):
    def __init__(self):
        super(MainWindow2, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)

        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)
        self.NextBTN = QPushButton("Next")
        self.NextBTN.clicked.connect(self.Next)
        self.VBL.addWidget(self.NextBTN)
        self.Worker2 = Worker2()
        # self.TestWorker=TestWorker()

        self.Worker2.start()
        self.Worker2.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker2.CaptureFinished.connect(self.training)


        self.setLayout(self.VBL)
    def Next(self):
        self.close()
        self.audio=AudioWindow()
        self.audio.show()

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker2.stop()
        sys.exit(0)
    def training(self,image):
        self.TestWorker=TestWorker(image)

        self.TestWorker.CaptureFinishedWorker.connect(self.finishedTraining)
        self.TestWorker.start()

    def finishedTraining(self,msg):
        self.TestWorker.stop()
        print(msg)
class AudioTest(QWidget):

    def __init__(self):
        super(AudioTest, self).__init__()
        self.register_cnt = 1
        self.compare_audio_path1=r"F:\PycharmProject\researchProject\Yue\SpeakerDatabase\temp1.wav"
        self.compare_audio_path2 = r"F:\PycharmProject\researchProject\Yue\SpeakerDatabase\temp.wav"

        self.VBL = QVBoxLayout()

        self.Record = QPushButton("Record")
        self.Compare=QPushButton("Compare")
        self.Record.clicked.connect(self.register_record_pubutton_clicked)
        self.Compare.clicked.connect(self.compare_pubutton_clicked)
        self.VBL.addWidget(self.Record)
        self.VBL.addWidget(self.Compare)

        self.setLayout(self.VBL)
        self.record_audio = RecordAudio()

    def register_record_pubutton_clicked(self):
        self.register_record_or_not = 1

        # if self.register_cnt == 0:
        #     self.register_cnt = 1
        #     self.set_register_recorder_mode(self.register_cnt)
        # elif self.register_cnt == 1:
        # self.set_register_recorder_mode(self.register_cnt)
        self.register_audio_path = self.record_audio.record(wait=False)
        # self.register_cnt = 0
        # self.set_register_recorder_mode(self.register_cnt)

    def compare_pubutton_clicked(self):
        if os.path.exists(self.compare_audio_path1) and os.path.exists(self.compare_audio_path2):
            feature1 = infer(self.compare_audio_path1)[0]
            feature2 = infer(self.compare_audio_path2)[0]
            dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
            if dist > args.threshold:
               print("?????????????????????, ?????????:{:.3f}".format(dist))
            else:
               print("????????????????????????, ?????????:{:.3f}".format(dist))
        else:
            print('??????????????????')


class AudioWindow(QWidget):
    def __init__(self):
        super(AudioWindow, self).__init__()
        self.register_cnt = 1

        self.VBL = QVBoxLayout()

        self.Record = QPushButton("Record")
        self.Record.clicked.connect(self.register_record_pubutton_clicked)
        self.VBL.addWidget(self.Record)

        self.setLayout(self.VBL)
        self.record_audio = RecordAudio()

    def register_record_pubutton_clicked(self):
        self.register_record_or_not = 1

        # if self.register_cnt == 0:
        #     self.register_cnt = 1
        #     self.set_register_recorder_mode(self.register_cnt)
        # elif self.register_cnt == 1:
            # self.set_register_recorder_mode(self.register_cnt)

        self.register_audio_path = self.record_audio.record(wait=False)
            # self.register_cnt = 0
            # self.set_register_recorder_mode(self.register_cnt)


class Worker2(QThread):
    ImageUpdate = pyqtSignal(QImage)
    CaptureFinished=pyqtSignal(numpy.ndarray)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)

        data_path = "./haarcascade_frontalface_alt.xml"
        classfier = cv2.CascadeClassifier(data_path)
        path_name=r"F:\PycharmProject\researchProject\Yue\facial_recognition\face_img\User"
        # os.mkdir("./face_img/User")
        color = (0, 255, 0)
        num=0
        capuFinished=False
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # ???????????????????????????????????????
                # ???????????????1.2???2?????????????????????????????????????????????????????????
                faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if len(faceRects) > 0 and num<10:  # ??????0??????????????????
                    for faceRect in faceRects:  # ???????????????????????????
                        x, y, w, h = faceRect

                        # ???????????????????????????
                        # img_name = '%s/%d.jpg ' % (path_name, num)
                        image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                        if not capuFinished:
                            self.CaptureFinished.emit(image)
                            print("Image Passed")
                            capuFinished=True
                        # print(type(image))
                        # cv2.imwrite(img_name, image)
                        # num += 1
                        # if num > 10:  # ????????????????????????????????????????????????
                        #     break

                        # ???????????????
                        # cv2.rectangle(Image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                        # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)

                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                # cv2.imshow("Test", frame)
                c = cv2.waitKey(10)
                if c & 0xFF == ord('q'):
                    break
    def stop(self):
        self.ThreadActive = False
        self.quit()
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        self.Form=Form
        Form.setObjectName("Form")
        Form.resize(1125, 732)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(380, 260, 321, 101))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.changewindow)
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(380, 390, 321, 101))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.login)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "Login"))
        self.pushButton_2.setText(_translate("Form", "Register"))

    def changewindow(self):
        self.camWindows=MainWindow()
        self.camWindows.show()
        self.Form.close()
    def login(self):
        self.loginWindows=MainWindow2()
        self.loginWindows.show()
        self.Form.close()




if __name__ == "__main__":

    # face.training_process()
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui=AudioTest()
    ui.show()
    # ui = Ui_Form()
    # ui.setupUi(Form)
    #
    # Form.show()
    sys.exit(app.exec_())
