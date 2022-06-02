import os
import sys

import tensorflow.python.keras.backend
sys.path.append(r"F:\PycharmProject\researchProject\Yue\ECPCA")
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

from PyQt5 import QtCore, QtGui, QtWidgets

import main_dlg
FaceDB_path="./User/FaceDB"
VoiceDB_path="./User/VoiceDB"
IrisDB_path="./User/IrisDB"
user_path=""
user_name=""
login_user_path=""
login_user_name=""

class FileDialog(QDialog):
    def __init__(self):
        super(FileDialog,self).__init__()
        self.initUI()
    def initUI(self):
        self.setWindowTitle("Result")
        self.resize(80,80)
        self.msg=QLabel("hello")
        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.msg)
        self.setLayout(self.mainLayout)
        self.btn=QPushButton("OK")
        self.btn.clicked.connect(self.closeDia)
        self.mainLayout.addWidget(self.btn)

    def setMessage(self,msg):
        self.msg.setText(msg)
    def closeDia(self):
        self.close()

class LoginAndRegister(object):
    def setupUi(self, Form):
        self.Form=Form
        Form.setObjectName("Form")
        Form.resize(1024, 768)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(358, 260, 300, 100))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.login_User)
        self.pushButton_2 = QtWidgets.QPushButton(Form)
        self.pushButton_2.setGeometry(QtCore.QRect(358, 390, 300, 100))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_2.clicked.connect(self.registerUser)



        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Biotic Authentication System"))
        self.pushButton.setText(_translate("Form", "Login"))
        self.pushButton_2.setText(_translate("Form", "Register"))

        # self.pushButton_3.setText(_translate("Form", "Test"))

    def registerUser(self):
        self.UserName = QtWidgets.QWidget()
        self.ui = UserName()
        self.ui.setupUi(self.UserName)
        self.Form.close()
        self.UserName.show()
    def login_User(self):
        self.login= QtWidgets.QWidget()
        self.login_ui=UserName_login()
        self.login_ui.setupUi(self.login)
        self.Form.close()
        self.login.show()

class UserName_login(object):
    def setupUi(self, Form):
        self.Form = Form
        Form.setObjectName("Form")
        Form.resize(663, 258)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(260, 150, 131, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.LoginUser)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(250, 120, 171, 20))
        self.label.setText("")
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(230, 80, 201, 41))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "Next"))
        self.textEdit.setHtml(_translate("Form",
                                         "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                         "p, li { white-space: pre-wrap; }\n"
                                         "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Please Enter Your Username</span></p>\n"
                                         "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p></body></html>"))

    def LoginUser(self):

        self.userName = self.textEdit.toPlainText()
        global login_user_name
        login_user_name = self.userName.strip("\n")

        self.userPath = os.getcwd() + "\\User\\" + self.userName.strip("\n")
        global login_user_path
        login_user_path = self.userPath

        user_list = os.listdir("./User")

        if os.path.exists(FaceDB_path+"\\"+login_user_name):
            print("Exist")
            self.label.setText("Welcome"+self.userName)
            import time
            time.sleep(1)
            self.testImage=MainWindow2()
            self.Form.close()
            self.testImage.show()

        else:
            self.label.setText("Invalid User")

class UserName(object):
    def setupUi(self, Form):
        self.Form=Form
        Form.setObjectName("Form")
        Form.resize(663, 258)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(260, 150, 131, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.RegisterUser)
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(250, 120, 171, 20))
        self.label.setText("")
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(Form)
        self.textEdit.setGeometry(QtCore.QRect(230, 80, 201, 41))
        self.textEdit.setObjectName("textEdit")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pushButton.setText(_translate("Form", "Next"))
        self.textEdit.setHtml(_translate("Form",
                                         "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                         "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                         "p, li { white-space: pre-wrap; }\n"
                                         "</style></head><body style=\" font-family:\'SimSun\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
                                         "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:10pt;\">Please Enter Your Username</span></p>\n"
                                         "<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:10pt;\"><br /></p></body></html>"))
    def RegisterUser(self):
        if os.path.exists("./User"):
            pass
        else:
            os.mkdir("./User")
            os.mkdir("./User/FaceDB")
            os.mkdir("./User/VoiceDB")
            os.mkdir("./User/IrisDB")

        self.userName=self.textEdit.toPlainText()
        global user_name
        user_name=self.userName.strip("\n")
        if os.path.exists(FaceDB_path+"\\"+user_name):
            print("Exist")
            self.label.setText("User already Exists")
        else:
            os.mkdir(FaceDB_path+"\\"+user_name)
            os.mkdir(VoiceDB_path+"\\"+user_name)
            os.mkdir(IrisDB_path+"\\"+user_name)
            # os.mkdir(self.userPath)
            self.FaceRegister=FaceRegister()
            self.Form.close()
            self.FaceRegister.show()

class MainWindow2(QWidget):
    def __init__(self):
        super(MainWindow2, self).__init__()

        self.VBL = QVBoxLayout()
        self.resize(640,480)
        self.FeedLabel = QLabel()
        self.VBL.addWidget(self.FeedLabel)
        self.MessageLabel=QLabel()
        self.VBL.addWidget(self.MessageLabel)
        self.MessageLabel.setText("Waiting For Camera..")
        self.MessageLabel.setAlignment(Qt.AlignHCenter)


        self.CancelBTN = QPushButton("Cancel")
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)
        # self.NextBTN = QPushButton("Next")
        # # self.NextBTN.clicked.connect(self.Next)
        # self.VBL.addWidget(self.NextBTN)
        self.Worker2 = Worker2()

        self.Worker2.start()
        self.Worker2.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker2.CaptureFinished.connect(self.training)


        self.setLayout(self.VBL)
    # def Next(self):
    #     self.close()
    #     self.audio=AudioWindow()
    #     self.audio.show()

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
        d=FileDialog()
        if str(msg)==str(login_user_name):
            d.setMessage("Facial authentication Passed")
            d.exec_()
            QThread.sleep(1)
            self.close()
            self.audioTest=AudioTest()
            self.audioTest.show()
            self.Worker2.stop()
        else:
            d.setMessage("Facial authentication Failed")
            d.exec_()
            self.close()

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

                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
                # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
                faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if len(faceRects) > 0 and num<10:  # 大于0则检测到人脸
                    for faceRect in faceRects:  # 单独框出每一张人脸
                        x, y, w, h = faceRect

                        image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                        if not capuFinished:
                            self.CaptureFinished.emit(image)
                            print("Image Passed")
                            capuFinished=True


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

class FaceRegister(QWidget):
    def __init__(self):
        super(FaceRegister, self).__init__()

        self.VBL = QVBoxLayout()

        self.FeedLabel = QLabel()
        self.FaceDetectLabel=QLabel()

        self.FaceDetectLabel.setText("Waiting For the Camera...")
        self.FaceDetectLabel.setAlignment(Qt.AlignHCenter)
        self.resize(640,480)
        self.VBL.addWidget(self.FeedLabel)
        self.FeedLabel.setGeometry(QtCore.QRect(0, 0, 640, 480))
        self.VBL.addWidget(self.FaceDetectLabel)
        self.CancelBTN = QPushButton("Next")
        self.CancelBTN.setEnabled(False)
        self.CancelBTN.clicked.connect(self.CancelFeed)
        self.VBL.addWidget(self.CancelBTN)

        self.Worker1 = Worker1()
        self.Worker2 = TrainWorker()


        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.CaptureFinished.connect(self.training)
        self.Worker1.FaceFound.connect(self.changeMsg)
        self.Worker2.CaptureFinishedWorker.connect(self.finishedTraining)
        self.setLayout(self.VBL)

    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def CancelFeed(self):
        self.Worker1.stop()
        self.AudioRegister=AudioWindow()
        self.close()
        self.AudioRegister.show()
    def training(self, Name):
        print(Name)
        self.Worker2.start()
    def changeMsg(self,msg):
        self.FaceDetectLabel.setText(msg)
    def finishedTraining(self,msg):
        self.Worker2.stop()
        self.FaceDetectLabel.setText("Capturing Finished, Click Next to Continue")
        self.CancelBTN.setEnabled(True)

class AudioTest(QWidget):

    def __init__(self):
        super(AudioTest, self).__init__()
        self.register_cnt = 1
        print(os.getcwd())

        # login_user_path = r"F:\PycharmProject\researchProject\Yue\User\NewTry"
        self.compare_audio_path1=login_user_path+"\\audio\\register.wav"
        self.compare_audio_path2=login_user_path+"\\audio\\login.wav"
        self.resize(400,240)
        self.VBL = QVBoxLayout()
        self.Message=QLabel()
        self.Message.setText("Click Record to Compare")
        self.Message.setAlignment(Qt.AlignHCenter)
        self.VBL.addWidget(self.Message)


        self.Record = QPushButton("Record")
        # self.Compare=QPushButton("Compare")
        # self.Compare.setEnabled(False)

        self.Record.clicked.connect(self.register_record_pubutton_clicked)
        # self.Compare.clicked.connect(self.compare_pubutton_clicked)
        self.VBL.addWidget(self.Record)
        # self.VBL.addWidget(self.Compare)
        # self.RecordTestWorker=AudioTestWorker()

        self.setLayout(self.VBL)
        self.record_audio = RecordAudio()

    def register_record_pubutton_clicked(self):
        self.register_record_or_not = 1
        output_path=VoiceDB_path+"\\"+login_user_name+"\\login.wav"
        s="1 "+"register.wav "+"login.wav"
        with open(VoiceDB_path+"\\"+login_user_name+"\\list.txt", 'w') as f:
            f.write(s)
        # print(login_user_path+"\\audio")

        self.record_worker=AudioRecordWorker(eval_path=output_path)
        self.record_worker.WorkingState.connect(self.changeLabel)
        self.record_worker.start()
        #
        # self.register_audio_path = self.record_audio.record(output_path=output_path, wait=False)

    def compare(self):

        self.RecordTestWorker = AudioTestWorker(eval_path=VoiceDB_path+"\\"+login_user_name,eval_list=VoiceDB_path+"\\"+login_user_name+"\\list.txt")

        self.RecordTestWorker.ReturnScoreWorker.connect(self.work_finished)

        self.RecordTestWorker.start()

    def changeLabel(self,msg):
        self.Message.setText(msg)
        if msg=="Voice Saved, Press Next to continue":
            self.compare()

    @QtCore.pyqtSlot(str)
    def work_finished(self,msg):
        d = FileDialog()
        print(msg)
        self.RecordTestWorker.stop()
        if float(msg)>0:
            d.setMessage("Speaker Verification Passed")
            d.exec_()
            # self.Message.setText("Voice Authentication Passed")
            self.Iris=main_dlg.QMainDialog(login_user_name)
            self.close()
            self.Iris.show()
        else:
            d.setMessage("Speaker Verification Failed")
            d.exec_()
            self.close()


    def compare_pubutton_clicked(self):
        import torch
        torch.cuda.empty_cache()
        if os.path.exists(self.compare_audio_path1) and os.path.exists(self.compare_audio_path2):
            print("yes")
            feature1 = infer(self.compare_audio_path1)[0]
            feature2 = infer(self.compare_audio_path2)[0]
            dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
            if dist > args.threshold:
               print("二者是同一个人, 相似度:{:.3f}".format(dist))
            else:
               print("二者不是同一个人, 相似度:{:.3f}".format(dist))
        else:
            print('未识别到音频')
        self.Iris=main_dlg.QMainDialog()
        self.close()
        self.Iris.show()

class AudioTestWorker(QThread):
    ReturnScoreWorker = pyqtSignal(str)

    def __init__(self, eval_path,eval_list, parent=None):
        QThread.__init__(self, parent)
        self.eval_path = eval_path
        self.eval_list = eval_list

    def run(self):
        self.ThreadActive = True
        from ECPCA.trainECAPAModel import ECAPAModelEval
        score=ECAPAModelEval(eval_list=self.eval_list,eval_path=self.eval_path)
        self.ReturnScoreWorker.emit(str(score))

    def stop(self):
        self.ThreadActive = False
        self.quit()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)
    CaptureFinished=pyqtSignal(str)
    FaceFound=pyqtSignal(str)
    def run(self):
        self.ThreadActive = True
        Capture = cv2.VideoCapture(0)
        data_path = "./haarcascade_frontalface_alt.xml"
        classfier = cv2.CascadeClassifier(data_path)

        path_name=FaceDB_path+"\\"+user_name
        print(path_name)
        color = (0, 255, 0)
        num=0
        CAMERA=False
        while self.ThreadActive:
            ret, frame = Capture.read()
            if ret:
                Image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 将当前桢图像转换成灰度图像
                # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
                faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
                if len(faceRects)==0 and CAMERA:
                    self.FaceFound.emit("No Face Detected! Please Face to The Camera")
                if len(faceRects)>1 and CAMERA:
                    self.FaceFound.emit("Multiple Faces Detected Please Hold")
                if len(faceRects) ==1 and num<10:  # 大于0则检测到人脸
                    for faceRect in faceRects:  # 单独框出每一张人脸
                        x, y, w, h = faceRect
                        self.FaceFound.emit("Capturing...")
                        # 将当前帧保存为图片
                        img_name = '%s/%d.jpg ' % (path_name, num)
                        image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                        cv2.imwrite(img_name, image)
                        num += 1
                        # if num > 10:  # 如果超过指定最大保存数量退出循环
                        #     break

                        # 画出矩形框
                        cv2.rectangle(Image, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)

                        # 显示当前捕捉到了多少人脸图片了，这样站在那里被拍摄时心里有个数，不用两眼一抹黑傻等着
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # cv2.putText(frame, 'num:%d' % (num), (x + 30, y + 30), font, 1, (255, 0, 255), 4)
                if num==10:
                    self.CaptureFinished.emit("Yes")
                    CAMERA=False
                    num=100
                FlippedImage = cv2.flip(Image, 1)
                ConvertToQtFormat = QImage(FlippedImage.data, FlippedImage.shape[1], FlippedImage.shape[0], QImage.Format_RGB888)
                Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.ImageUpdate.emit(Pic)
                if num==0:
                    CAMERA=True



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
        face.training_process(FaceDB_path)
        self.CaptureFinishedWorker.emit("Finished")

    def stop(self):
        self.ThreadActive = False
        self.quit()

class AudioWindow(QWidget):
    def __init__(self):
        super(AudioWindow, self).__init__()
        self.register_cnt = 1

        self.VBL = QVBoxLayout()
        self.resize(400,240)
        self.Notation=QLabel()
        self.Notation.setAlignment(Qt.AlignHCenter)
        self.Notation.setText("Please push the record button")
        self.Notation.setFont(QFont('Times', 16))
        self.VBL.addWidget(self.Notation)
        self.Record = QPushButton("Record")
        self.Record.clicked.connect(self.register_record_pubutton_clicked)
        self.NextBtn=QPushButton("Next")
        self.NextBtn.setEnabled(False)
        self.NextBtn.clicked.connect(self.goIris)
        self.VBL.addWidget(self.Record)
        self.VBL.addWidget(self.NextBtn)
        self.setWindowTitle("Voice Register")
        self.setLayout(self.VBL)
        self.record_audio = RecordAudio()

    def register_record_pubutton_clicked(self):
        # self.Notation.setText("Start Recording, Please Keep Talking For 5 seconds")
        output_path=VoiceDB_path+"\\"+user_name

        output_path+="\\register.wav"
        # import time
        # for i in range(3):
        #     self.Notation.setText("Recording will begin in :" +str(3-int(i)))
        #     time.sleep(1) self.Notation.setText("Start Recording, Please Keep Talking For 3 seconds")
        self.record_worker=AudioRecordWorker(eval_path=output_path)
        self.record_worker.WorkingState.connect(self.changeLabel)
        self.record_worker.start()
        # self.register_audio_path = self.record_audio.record(output_path=output_path,wait=False)
        # self.Notation.setText("Voice Saved")

    def goIris(self):
        self.Qwi=QtWidgets.QWidget()
        self.iris_ui=CopyIris()
        self.iris_ui.setupUi(self.Qwi)
        self.close()
        self.Qwi.show()
    def changeLabel(self,msg):
        self.Notation.setText(msg)
        if msg=="Voice Saved, Press Next to continue":
            self.NextBtn.setEnabled(True)

class AudioRecordWorker(QThread):
    WorkingState = pyqtSignal(str)

    def __init__(self, eval_path, parent=None):
        QThread.__init__(self, parent)
        self.eval_path = eval_path
        self.record_audio = RecordAudio()

    def run(self):
        self.ThreadActive = True
        self.WorkingState.emit("Start Recording, Please Keep Talking For 3 seconds")
        self.register_audio_path = self.record_audio.record(output_path=self.eval_path, wait=False)
        self.WorkingState.emit("Voice Saved, Press Next to continue")

    def stop(self):
        self.ThreadActive = False
        self.quit()
            # self.register_cnt = 0
            # self.set_register_recorder_mode(self.register_cnt)

class CopyIris(object):
    def setupUi(self, Form):
        self.Form=Form
        Form.setObjectName("Form")
        Form.resize(400, 240)

        self.Notation=QtWidgets.QLabel(Form)
        self.Notation.setGeometry(QtCore.QRect(0, 90, 400, 30))
        self.Notation.setAlignment(Qt.AlignHCenter)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(0, 180, 400, 30))
        self.pushButton2 = QtWidgets.QPushButton(Form)
        self.pushButton2.setGeometry(QtCore.QRect(0, 210, 400, 30))

        self.pushButton.setObjectName("pushButton")
        self.pushButton2.setObjectName("pushButton2")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Copy Iris"))
        self.Notation.setText("Copy Iris Image For Demonstration")
        self.Notation.setFont(QFont('Times', 11))
        # self.Notation.setGeometry(QtCore.QRect(0, 0, 400, 30))
        # self.Notation.setAlignment(Qt.AlignHCenter)
        self.pushButton.setText(_translate("Form", "Browse File"))
        self.pushButton2.setText(_translate("Form", "Login"))
        self.pushButton.clicked.connect(self.pushButton_handler)
        self.pushButton2.clicked.connect(self.login)
        self.pushButton2.setEnabled(False)
    def pushButton_handler(self):
        print("Button pressed")
        self.open_dialog_box()

    def open_dialog_box(self):
        folderpath = QtWidgets.QFileDialog.getExistingDirectory(self.Form, 'Select Folder')
        print(folderpath)
        import os
        import shutil

        source_folder = folderpath+"\\"
        # os.mkdir(user_path+"\\Iris")

        destination_folder = IrisDB_path+"\\"+user_name+"\\"
        print(destination_folder)

        # fetch all files
        try:
            for file_name in os.listdir(source_folder):
                # construct full file path
                print(file_name)
                source = source_folder + file_name
                destination = destination_folder + file_name
                # copy only files
                if os.path.isfile(source):
                    shutil.copy(source, destination)
                    print('copied', file_name)
            length=len(os.listdir(source_folder))
            self.Notation.setText("  "+str(length)+" File Copied! Registration Finished")
            self.Notation.setGeometry(QtCore.QRect(0, 90, 400, 30))
            self.Notation.setAlignment(Qt.AlignHCenter)
            # self.Notation.adjustSize()
            self.pushButton2.setEnabled(True)
        except:
            self.Notation.setText("No dir Selected")

            # import time
            # time.sleep(5)
            # sys.exit(0)

    def login(self):
        self.login = QtWidgets.QWidget()
        self.login_ui = UserName_login()
        self.login_ui.setupUi(self.login)
        self.Form.close()
        self.login.show()



def except_hook(cls, exception, traceback):
    sys.__excepthook__(cls, exception, traceback)

if __name__ == "__main__":


    import sys
    app = QtWidgets.QApplication(sys.argv)

    # Form=AudioWindow()
    # Form.show()
    Form = QtWidgets.QWidget()
    ui = LoginAndRegister()
    ui.setupUi(Form)
    sys.excepthook = except_hook
    Form.show()
    sys.exit(app.exec_())
