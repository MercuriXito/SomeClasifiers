# SomeClassifier

[![README-en](https://img.shields.io/badge/README-English-red)](README.md)

记录在不同的数据集上训练不同的分类器。

## Networks

所有模型的结构都可以在`ClassifierModels`里面找到。所有已经训练和测试的模型的**state_dict**都存储在`models`文件夹下。欢迎取用。

+ `DeeperNetClassifier`: 包含四层卷积层的卷积网络，有足够的能力表达一些小数据集的特征，比如在MNIST类数据集、CIFAR10上。

## Dataset

对所有列出的数据集的dataset(`torch.utils.data.Dataset`的实例)，如果在它`torchvision.utils.dataset`中没有给出合适加载的函数，对应的加载方法可以查看不同的`.py`文件。你可以正常使用`torch.utils.data.DataLoader`加载。

+ `FashionMNIST`: [更多细节](https://github.com/zalandoresearch/fashion-mnist)
+ `KMNIST`: [更多细节](https://github.com/rois-codh/kmnist)
+ `Kuzushiji49`: [更多细节](https://github.com/rois-codh/kmnist)
+ `Kuzushiji-Kanji`: [更多细节](https://github.com/rois-codh/kmnist)

## Work

如果没有具体说明，我使用默认的参数训练网络(学习率`learning rate`=0.001)，而且我使用默认20个epochs训练，其他的参数(例如`optimizer`梯度下降方法，`criterion`loss计算函数)在`ClassifierTrainer`中可以看出。

| 模型 | 数据集 | 验证集最好表现 | 测试集表现 | 参数设置 |
| -- | -- | -- | -- | -- |
| `DeeperNetClassifier` | Kuzushiji49 | 95.162632% | 90.803942% | default |
