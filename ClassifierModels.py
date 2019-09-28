#-*-coding:utf-8-*-

"""
    @file:			ClassifierModels.py
    @autor:			Victor Chen
    @description:
        多大的数据集，选择多大的对应表达能力的网络就行嘛
"""

import torch
import torch.nn as nn
import torch.functional as F 

class AlexNetd(nn.Module):
    """用AlexNet类似结构作KMNIST的分类
    """
    def __init__(self):
        super(AlexNetd,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0, 1),
            nn.ReLU(True),
            nn.Conv2d(96, 192, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0, 1),
            nn.ReLU(True),
            nn.Conv2d(192, 384, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 0, 1),
            nn.ReLU(True)
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size()[0], -1) )



class DeeperNetClassifier(nn.Module):
    """ 4-Conv NN, input: [ , 3, 32, 32], output: 10 classes
    """
    def __init__(self):
        super(DeeperNetClassifier, self).__init__()
    
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 5),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 64, 10),
#            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size()[0], -1))