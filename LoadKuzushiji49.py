#-*-coding:utf-8-*-

"""
    @file:			LoadKuzushiji49.py
    @autor:			Victor Chen
    @description:
        Kuzushiji49 数据集的loader
"""

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import utils

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from PIL import Image
import random

from utils import show_images, train_val_split

root = "/home/victorchen/workspace/Venus/Kuzushiji49/"

class Kuzhushiji49Dataset(Dataset):
    """"""
    def __init__(self, root, transform =  None, train = True, target_transform = None):

        super(Kuzhushiji49Dataset, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.train = train
        target = "train" if self.train else "test" 

        self.data = np.load(root + "k49-{}-imgs.npz".format(target))["arr_0"]
        self.labels = np.load(root + "k49-{}-labels.npz".format(target))["arr_0"]

        # read class labels
        classdf = pd.read_csv(root + "k49_classmap.csv")
        self.classes =  { x:y for x,y in zip(classdf.iloc[:,0] , classdf.iloc[:, 2])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        assert idx >= 0 and idx < self.__len__()
        # load 
        image = self.data[idx, :]
        label = self.labels[idx].astype(np.long)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label 


def default_loader():
    """ default loading Kuzushiji19 dataset using default transform and default validation set size
    Returns:
        train, val, test (three dataloaders)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    datasets = []
    train_dataset = Kuzhushiji49Dataset(root, transform = transform, train=True)
    train_dataset, val_dataset = train_val_split(train_dataset)
    test_dataset = Kuzhushiji49Dataset(root, transform =transform, train=False)
    datasets = [train_dataset, val_dataset, test_dataset]

    dataloaders = []
    for dataset in datasets:
        dataloaders.append(DataLoader(dataset, shuffle=True, num_workers=2, batch_size=32))

    return dataloaders