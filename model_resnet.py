# -*- coding: utf-8 -*-
# @Author  : 胡振兴
# @Time    : 2021/7/26 17:29
# @project : 虹膜识别
# version： 
# @File    : model_resnet.py
# @note    : 
# --------------------------------
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from resnet import resnet18, resnet34, resnet50, resnet101, resnet152


class ModelResnet(object):
    def __init__(self):
        self.model = None
        self.device = None
        self.img_size = (256, 256)
        # self.load_model("resnet_50", 14, self.img_size,
        #                 './model_exp/resnet_50-size-256_epoch-195.pth')

    def load_model(self, num_classes, img_size, model_name, test_model):
        if model_name == 'resnet_18':
            self.model = resnet18(num_classes=num_classes, img_size=img_size[0])
        elif model_name == 'resnet_34':
            self.model = resnet34(num_classes=num_classes, img_size=img_size[0])
        elif model_name == 'resnet_50':
            self.model = resnet50(num_classes=num_classes, img_size=img_size[0])
        elif model_name == 'resnet_101':
            self.model = resnet101(num_classes=num_classes, img_size=img_size[0])
        elif model_name == 'resnet_152':
            self.model = resnet152(num_classes=num_classes, img_size=img_size[0])
        else:
            print('error no the struct model : {}'.format(model_name))

        use_cuda = torch.cuda.is_available()

        # self.device = torch.device("cuda:0" if use_cuda else "cpu")///
        self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为前向推断模式

        # 加载测试模型
        chkpt = torch.load(test_model, map_location=self.device)
        self.model.load_state_dict(chkpt)
        print('加载模型成功 : {}'.format(test_model))

    def img_pre(self, img):
        img_ = cv2.resize(img, (self.img_size[1], self.img_size[0]), interpolation=cv2.INTER_CUBIC)

        img_ = img_.astype(np.float32)
        img_ = (img_ - 128.) / 256.

        img_ = img_.transpose(2, 0, 1)
        img_ = torch.from_numpy(img_)
        img_ = img_.unsqueeze_(0)
        return img_

    def predict(self, img_path):
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        img_brg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_brg, cv2.COLOR_BGR2RGB)

        img_pre = self.img_pre(img_rgb)

        img_pre.to(self.device)
        pre_ = self.model(img_pre.to(self.device).float())

        outputs = F.softmax(pre_, dim=1)
        outputs = outputs[0]
        output = outputs.cpu().detach().numpy()
        output = np.array(output)

        max_index = np.argmax(output)

        score_ = output[max_index]
        print(max_index, score_)
        return max_index, score_


if __name__ == '__main__':
    model_resnet = ModelResnet()
    test_model = '../model/resnet34_256.pth'
    model_resnet.load_model(108, (256, 256), 'resnet_34', test_model)
    img_org = cv2.imdecode(np.fromfile('../test_img/068_2_4.jpg', dtype=np.uint8), -1)
    model_resnet.predict(img_org,)
