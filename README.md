# SomeClassifier

[![README-zh](https://img.shields.io/badge/README-%E4%B8%AD%E6%96%87-brightgreen)](README_zh.md)

Track on training different classifiers on different dataset. I think the process would be rather fun.

## Networks

All the structure of models are avaliable in Module `ClassifierModels`. And **state_dict** of every trained and tested model would be stored under folder `models`. It' welcomed to use these models for transfer learning or else.

+ `DeeperNetClassifier`: ConvNet with 4 convolutional layer, it's fairly enough in many small dataset, like MNIST or else.

## Dataset

All the dataset (`torch.utils.data.Dataset`) of dataset would be specified in different `.py` file if it\`s not rather easily loaded by modules in `torchvision.utils.dataset`. And you could loaded with `torch.utils.data.DataLoader`

+ `FashionMNIST`: [for more information](https://github.com/zalandoresearch/fashion-mnist)
+ `KMNIST`: [for more information](https://github.com/rois-codh/kmnist)
+ `Kuzushiji49`: [for more information](https://github.com/rois-codh/kmnist)
+ `Kuzushiji-Kanji`: [for more information](https://github.com/rois-codh/kmnist)

## Work

If not specified, all the network are trained with default parameters like **learning rate = 10e-3** and we train for **20 epochs**, other parameters like `optimizer` or `criterion` are also avaliable in Module `ClassifierTranier`.

| models | dataset | best behave on validation set | behave on test set | settings | preprocessing |
| -- | -- | -- | -- | -- | -- |
| `DeeperNetClassifier` | Kuzushiji49 | 95.162632% | 90.803942% | default | Data augmentation |
| `DeeperNetClassifier` | KMNIST | 97.905585% | 98.323333% | default | Data augmentation |
| `DeeperNetClassifier` | CIFAR10 | 74.104299% | 80.310301% | default | Data augmentation |
| `DenseNet` | CIFAR10 | 79.905063% | 84.149217% | 50 epochs , PlateauReduce | Data augmentation |
| `ResNet101` from torchvision | CIFAR10 | 86.226115% | 77.631158% | 50 epochs, PlateauReduce, | Resize96 Data augmentation |
