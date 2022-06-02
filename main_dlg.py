# -*- coding: utf-8 -*-
# @Author  : 飞鸟
# @Time    : 2021/7/4 9:12
# @project :
# @File    : main_dlg.py
# @note    : main_dlg
# --------------------------------
import os.path
import sys

import cv2 as cv
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from dlg.draw_img import DrawPic
from main_process import MainProcess
from ui.ui_IrisDlg import Ui_IrisDlg  # 导入UI类
from MainGUI import FileDialog

class QMainDialog(QDialog, Ui_IrisDlg):  # 修改点（UI类）
    signal_show_result = pyqtSignal(dict)  # 显示列表信号槽

    def __init__(self,user_name, parent=None):
        super().__init__(parent)
        self.user_name=user_name
        self.ui = Ui_IrisDlg()  # 修改点（UI类）
        self.ui.setupUi(self)  # 构造UI
        self.show_img = DrawPic(self.ui.img_view)
        self.file_path = None
        self.img_org = None
        self.main_pro = None
        self.signal_show_result.connect(self.show_result)

    # 加载模型
    @pyqtSlot()
    def on_btn_load_clicked(self):
        print('加载模型')
        is_resnet = self.ui.radioBtn_resnet.isChecked()
        if is_resnet:
            self.main_pro = MainProcess(self.signal_show_result)
            self.main_pro.load_model('resnet')
        else:
            self.main_pro = MainProcess(self.signal_show_result)
            self.main_pro.load_model('transformer')
        # 加载成功

    # 导入图片
    @pyqtSlot()
    def on_btn_import_clicked(self):
        if self.main_pro is None:
            return

            # 打开图片
        curDir = QDir.currentPath()
        open_dir = os.path.join(curDir, 'test_img')
        self.file_path, fit = QFileDialog.getOpenFileName(self, "打开文件", open_dir, "图片文件(*.jpg);;图片文件(*.png)")
        print(self.file_path, fit)
        if self.file_path == "":
            return

        try:
            base_name = os.path.basename(self.file_path)
            self.ui.label_file_name.setText('File Name: ' + base_name)
            # 读取图片
            self.img_org = cv.imdecode(np.fromfile(self.file_path, dtype=np.uint8), -1)

            self.show_img.show(self.img_org)
            second_path = "./User/IrisDB/"+self.user_name
            iris_list=os.listdir(second_path)
            second_image=second_path+"/"+iris_list[0]


            # self.main_pro.start_predict(self.file_path)
            #second_path=r"F:\PycharmProject\researchProject\IrisRecognition-master\IrisRecognition-master\CASIA1\68\068_1_2.jpg"
            self.main_pro.start_predict_(self.file_path,second_image)



        except:
            print('Open Error! Try again!')

        print('处理图片')

    def show_result(self, data):
        print(data)
        code = data['code']
        if code == 200:
            d=FileDialog()
            class_name = data['class']
            score = data['score']
            result_text = 'Result: ' + str(class_name) + ", " + str(score)
            if float(score)>0.85:
                d.setMessage("You Finished the Verification Process, Welcome")
                d.exec_()
                self.ui.label_result.setText("Iris Authentication Passed")
            else:
                d.setMessage("Iris Authentication Failed")
                d.exec_()

                self.ui.label_result.setText("Iris Authentication Failed")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_form = QMainDialog()
    main_form.show()

    sys.exit(app.exec_())
