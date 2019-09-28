#-*-coding:utf-8-*-

"""
    @file:			train2.py
    @autor:			Victor Chen
    @description:
        
"""

import matplotlib.pyplot as plt 

import torch
import torch.nn as nn 

from ClassifierModels import DeeperNetClassifier
from utils import show_images, show_model_structure, test_model_output
from LoadKuzushiji49 import default_loader
from ClassifierTrainer import ClassfierTrainer


# load from default loader
train, val, test = default_loader()

# show some exampels or test the right output of the net
num_class = len(test.dataset.classes)
batch, label = next(iter(train))

# show examples
# show_images(batch)
# plt.show()

# load and modify model
model = DeeperNetClassifier()
model.features[0] = nn.Conv2d(1, 16, 5)
model.classifier = nn.Sequential(nn.Linear( 3 * 3* 64, num_class))

# test the size of input after each layer in net
# test_model_output(model.features, batch.size())
# show_model_structure(model)

# trainer and train, test, inference
trainer = ClassfierTrainer()

# train 
# trainer.train(model, train, val)

# test on test set
# trainer.test(model, test)


# inference
# model.load_state_dict(torch.load("models/DeeperNetClassifier_on_Kuzushiji49.pth"))
# labels = trainer.inference(model, batch)
# print("predicted class:")
# for i, l in enumerate(labels):
#     print(test.dataset.classes[l], end=" ")
#     if (i + 1) % 8 == 0:
#         print()
# 
# show_images(batch)
# plt.show()