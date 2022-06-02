import json
import os
import threading

from model_resnet import ModelResnet
from model_transformer import ModelTransformer


class MainProcess(object):
    def __init__(self, signal_show_result):
        self.signal_show_result = signal_show_result
        self.model_type = 'resnet'
        self.model_resnet = ModelResnet()
        self.model_transformer = ModelTransformer()
        self.class_indict = {}

    def load_model(self, model_type='resnet'):
        if model_type == 'resnet':
            self.model_type = 'resnet'
            test_model = './model/resnet34_256.pth'
            self.model_resnet.load_model(108, (256, 256), 'resnet_34', test_model)
        else:
            self.model_type = 'transformer'
            weight_path = "./model/transformer_256.pth"
            self.model_transformer.load_model(108, weight_path)

        # read class_indict
        json_path = './class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

        json_file = open(json_path, "r")
        self.class_indict = json.load(json_file)

    def start_predict(self, img_path):
        t1 = threading.Thread(target=self.predict, args=(img_path,), daemon=True)
        t1.start()

    def start_predict_(self, img_path1,img_path2):
        t1 = threading.Thread(target=self.predict_, args=(img_path1,img_path2,), daemon=True)
        t1.start()
    def predict(self, img_path):
        if self.model_type == 'resnet':
            max_index, score_ = self.model_resnet.predict(img_path)
            class_name = str(max_index + 1)
        else:
            max_index, score_ = self.model_transformer.predict(img_path)

            class_name = self.class_indict[str(max_index)]

        print_res = "class: {}   prob: {:.3}".format(class_name,
                                                     score_)
        print(print_res)
        result_dict = {'code': 200, 'class': class_name, 'score': score_}
        self.signal_show_result.emit(result_dict)

    def predict_(self, img_path1,img_path2):
        # if self.model_type == 'resnet':
        #     max_index, score_ = self.model_resnet.predict(img_path)
        #     class_name = str(max_index + 1)
        # else:
            # max_index, score_ = self.model_transformer.predict(img_path)
        import numpy as np
        feature1=self.model_transformer.predict_(img_path1)
        feature2=self.model_transformer.predict_(img_path2)
        dist = np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2))
        print(dist)
        # return dist
            # class_name = self.class_indict[str(max_index)]


        #
        # print_res = "class: {}   prob: {:.3}".format(class_name,
        #                                              score_)
        # print(print_res)
        result_dict = {'code': 200, 'class': dist, 'score': dist}
        self.signal_show_result.emit(result_dict)
