#-*-coding:utf-8-*-

"""
    @file:			utils.py
    @autor:			Victor Chen
    @description:
        utils 
"""

import torch
from torch.utils.data import random_split

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision import utils as vutils

import numpy as np 
import matplotlib.pyplot as plt

def show_model_structure(model):
    for i, m in enumerate(model.children()):
        print("{} - {}".format(i, m))


def test_model_output(model, input_size):
    """test the output in every child module of model
    """
    assert(isinstance(input_size, (list, tuple)))
    tensor = torch.randn(*input_size)
    for ms in model.children():
        tensor = ms(tensor)
        print("{} \t output: {}".format(ms.__class__.__name__, tensor.size()))


def show_images(tensor):
    """show the image of 4-dimensional tensor [batch_size, C, H, W]
    Return:
        plt
    """
    grid = vutils.make_grid(tensor).detach().cpu().numpy().transpose(1,2,0)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(grid)
    plt.show()


def train_val_split(dataset, val_size = 0.3):
    length = len(dataset)
    train_size = int(length * ( 1- val_size))
    size = [train_size, length - train_size]
    train, val = random_split(dataset, size)
    return train, val