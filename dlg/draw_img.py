# -*- coding: utf-8 -*-
# @Author  : 飞鸟
# @Time    : 2021/7/3 10:00 
# @project : 
# @File    : draw_img.py
# @note    : draw_img
# --------------------------------
import PyQt5.QtCore as QtCore
import PyQt5.QtGui as QtGui
import numpy as np
from PyQt5.QtWidgets import *


class DrawPic(QtCore.QObject):

    def __init__(self, view):
        super(DrawPic, self).__init__()
        self.pic_view = view
        self.zoom_scale = 1
        self.init_size = 600
        # 基础图片显示
        img_base = np.zeros((self.init_size, self.init_size, 3), dtype=np.uint8)
        img_qt = self.cv2QPix(img_base)
        self.img_item = QGraphicsPixmapItem(img_qt)  # 创建像素图元

        self.scene = QGraphicsScene()  # 创建场景
        self.scene.addItem(self.img_item)

        self.pic_view.setScene(self.scene)  # 将场景添加至视图

        self.pic_view.setTransform(QtGui.QTransform())
        self.pic_view.scale(self.zoom_scale, self.zoom_scale)

    def get_scale(self, img):
        print(img.shape)
        if len(img.shape) == 3:
            height, width, _ = img.shape
        else:
            height, width = img.shape

        height_scale = self.init_size / height
        width_scale = self.init_size / width
        print(height, width, height_scale, width_scale)

        last_scale = 1
        if height_scale > width_scale:
            last_scale = width_scale
        else:
            last_scale = height_scale

        return last_scale

    # 显示图片
    def show(self, img):
        if img is None:
            return
        self.scene.removeItem(self.img_item)

        pix_img = self.cv2QPix(img)
        self.img_item = QGraphicsPixmapItem(pix_img)  # 创建像素图元
        self.scene.addItem(self.img_item)  # 添加像素元

        last_scale = self.get_scale(img)
        print(last_scale)
        self.zoom_scale = last_scale
        self.pic_view.setTransform(QtGui.QTransform())
        self.img_item.setScale(self.zoom_scale)

        self.scene.update()

    @staticmethod
    def cv2QPix(img):
        img_shape = img.shape

        if len(img_shape) == 2:
            # cv 图片转换成 qt图片
            qt_img = QtGui.QImage(img.data,  # 数据源
                                  img.shape[1],  # 宽度
                                  img.shape[0],  # 高度
                                  img.shape[1],  # 行字节数
                                  QtGui.QImage.Format_Grayscale8)
            return QtGui.QPixmap.fromImage(qt_img)
        elif len(img_shape) == 3:
            # cv 图片转换成 qt图片
            qt_img = QtGui.QImage(img.data,  # 数据源
                                  img.shape[1],  # 宽度
                                  img.shape[0],  # 高度
                                  img.shape[1] * 3,  # 行字节数
                                  QtGui.QImage.Format_BGR888)
            return QtGui.QPixmap.fromImage(qt_img)
