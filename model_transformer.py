# -*- coding: utf-8 -*-
# @Author  : 胡振兴
# @Time    : 2021/7/26 17:28
# @project : 虹膜识别
# version： 
# @File    : model_transformer.py
# @note    : 
# --------------------------------

import torch
from PIL import Image
from torchvision import transforms

from vit_model import vit_base_patch16_224_in21k as create_model


class ModelTransformer(object):
    def __init__(self):
        self.model = None
        self.device = None

    def load_model(self, num_classes, weight_path):
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.device = torch.device("cpu")

        # create model
        self.model = create_model(num_classes=num_classes, has_logits=False).to(self.device)
        # load model weights

        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()
        print('加载模型成功！', weight_path)

    def predict(self, img_path):
        img = Image.open(img_path).convert("RGB")
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        return predict_cla, predict[predict_cla].numpy()
    def predict_(self, img_path):
        img = Image.open(img_path).convert("RGB")
        data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            # predict class
            output = torch.squeeze(self.model(img.to(self.device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
        #                                              predict[predict_cla].numpy())
        return predict

if __name__ == '__main__':
    model_transformer = ModelTransformer()
    model_type = 'transformer'
    weight_path = "../model/transformer_256.pth"
    model_transformer.load_model(108, weight_path)

    img = Image.open('../test_img/068_2_4.jpg').convert("RGB")

    model_transformer.predict(img)
