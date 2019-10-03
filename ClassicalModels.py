#-*-coding:utf-8-*-

"""
    @file:			ClassicalModels.py
    @autor:			Victor Chen
    @description:
        reimplement of classical state-of-art models with pytorch.
        Since all the models implemented below are originally trained on ImageNet,
        most of them take input image with size '3*224*224'. While trained with 
        images with different size, more modification on model will be specified 
        during training. 
"""

import torch 
import torch.nn as nn 
import torch.functional as F

########################################
# GoogLeNet

class Inception(nn.Module):
    """ 4 paralized structure "Inception" in GoogLeNet, 
    Inception Block does not change images size, but only change the channels.
    """
    def __init__(self, in_channels, out_channels_list):
        super(Inception, self).__init__()

        assert(isinstance(out_channels_list, list))
        assert(len(out_channels_list) == 4) and (len(out_channels_list[1]) == 2) and (len(out_channels_list[2])) == 2

        self.paralize1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_list[0], 1, 1),
            nn.ReLU(inplace=True),
        )

        self.paralize2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_list[1][0], 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_list[1][0], out_channels_list[1][1], 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.paralize3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels_list[2][0], 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels_list[2][0], out_channels_list[2][1], 5, 1, 2),
            nn.ReLU(inplace=True)
        )

        self.paralize4 = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(in_channels, out_channels_list[3], 1, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.paralize1(x)
        x2 = self.paralize2(x)
        x3 = self.paralize3(x)
        x4 = self.paralize4(x)

        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x


# Inception v1
class GoogLeNet(nn.Module):

    def __init__(self, in_channels, output_class):
        super(GoogLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1), # maxpool decrease both height and width by half
            nn.Conv2d(64, 64, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            Inception(192, [64,[96, 128],[16, 32],32]),
            Inception(256, [128, [128, 192], [32, 96], 64]),
            nn.MaxPool2d(3, 2),
            # followed by 5 inception blocks
            Inception(480, [192, [96, 208], [16, 48], 64]),
            Inception(512, [160, [112, 224], [24, 64], 64]),
            Inception(512, [128, [128, 256], [24, 64], 64]),
            Inception(512, [112, [144, 288], [32, 64], 64]),
            Inception(528, [256, [160, 320],[32, 128], 128]),
            nn.MaxPool2d(3, 2, 1),
            # followd by 2 inception blocks
            Inception(832, [256, [160, 320], [32, 128], 128]),
            Inception(832, [384, [192, 384], [48, 128], 128]),
        )   
        self.pooling = nn.AvgPool2d(7, 1) 
        self.classifier = nn.Sequential(
            nn.Linear(1024, output_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return self.classifier(x.view(x.size(0), -1))

##########################
# GoogLeNet的特点: 
# + wider: Inception结构使用多尺寸的卷积和对图像进行卷积，提取不同的图像特征，作为不同的feature map。
# + smaller: Inception结构五层连续使用，看起来很复杂，但是它作用在14*14大小的图片上，所以计算量不
#   会多很多， 看出GoogLeNet的思想是：先用大的卷积核粗提取特征，再在小的特征上尝试很多种特征组合，最后
#   它借鉴了NIN，去掉了AlexNet来的超大的fc层，而是通道维度的平均之间作为特征，很有种特征组合的味道。所以
#   GoogLeNet看起来很复杂，但是它需要的参数量远少于VGG。


########################################
# VGG

class vggblock(nn.Module):
    """VGGblock embeded in VGG models. One vggblock would decrease the width 
    and height of input by half respectively.
    """
    def __init__(self, in_channels, out_channels, num_conv):
        super(vggblock, self).__init__()

        blocks = [
            nn.Conv2d(in_channels, out_channels, 3, 1,1),
            nn.ReLU(inplace=True)
        ]

        for i in range(num_conv - 1):
            blocks.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            blocks.append(nn.ReLU(inplace=True))

        blocks.append(nn.MaxPool2d(2, 2))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class vgg11(nn.Module):
    """vgg11 model with arrangement of vggblock: '1-1-2-2-2'
    In VGG model, the size of pictures is changed by pooling layer (much different from AlexNet)
    """
    def __init__(self, in_channels, out_class):
        super(vgg11, self).__init__()
        self.features = nn.Sequential(
            vggblock(in_channels, 64, 1),
            vggblock(64, 128, 1),
            vggblock(128, 256, 2),
            vggblock(256, 512, 2),
            vggblock(512, 512, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, out_class),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x.view(x.size(0), -1))


########################################
# NIN

class NINblock(nn.Module):
    """NINblock embeded in NIN model, with one normal conv layer, followed by two 1*1 conv layer.
    one NINblock would act like one normal conv layer, since the followed two 1*1 conv layer 
    do not change the size of input.
    
    Q: why we need two 1*1 conv layer followed?
    A: to add more nonlinearity and much more complicated combination of feature maps, as also to decrease the 
    total number of parameters.
    """
    def __init__(self, in_channels, out_channels, kernel_size,\
        stride = 1, padding = 0, dilation = 1, groups = 1, bias = True, padding_mode = "zeros" ):

        super(NINblock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
            padding, dilation, groups, bias, padding_mode),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class NIN(nn.Module):
    """ NIN model.
    """
    def __init__(self, in_channels, out_class):
        super(NIN, self).__init__()
        self.features = nn.Sequential(
            NINblock(in_channels, 96, 11, 4, 2 ),
            nn.MaxPool2d(3, 2),
            NINblock(96, 256, 5, 1, 2 ),
            nn.MaxPool2d(3, 2),
            NINblock(256, 512, 3, 1, 1),
            nn.MaxPool2d(3, 2),
            NINblock(512, out_class, 1, 1)
        )
        self.pool = nn.AvgPool2d(6,1)
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


########################################
# test defined modules

if __name__ == "__main__":
    
    # use tool function in utils to test the structure of models
    import utils as cutils

    for model in [vgg11, NIN, GoogLeNet]:
        net = model(3, 1000)
        print("{} output: {}".format(net.__class__.__name__, net(torch.randn([3, 3, 224, 224])).size()))