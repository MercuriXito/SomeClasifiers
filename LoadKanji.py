#-*-coding:utf-8-*-

"""
    @file:			GetData.py
    @autor:			Victor Chen
    @description:
        数据集"Kuzushiji-Kanji"的三种loader方法，当然最推荐最后一种。。
        "Kuzushiji-Kanji"
"""

import os,sys
from functools import reduce
import PIL.Image as Image

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Dataset

root = '/home/victorchen/workspace/Venus/kkanji2/'

kanji_root = root 


class KanJiLoadCSV(object):
    """ 根据Kanji2数据集中各种文字的数量整理分类，生成一个DataFrame，最后写入一个csv文件，csv文件最大有2GB多，慎用该加载方式
    """
    def __init__(self, root, choose_n = 0):

        self.root = root
        self.choose_n = choose_n

    def load(self):
        num_items = 0
        map_class = {}

        self.data = None
        # load all data

        total_pic_num = 0
        for kanji in os.listdir(self.root):

            if not os.path.isdir(self.root + kanji):
                continue

            # the label is the name of filefolder
            unicode_str = "0x" + kanji[2:]
            unicodes = int(unicode_str, base=16).to_bytes(4, "little").decode("utf-16") # 小端，4个字节
            
            # drop class with less pictures
            num_pic = len(os.listdir(self.root + kanji))
            if not (num_pic > self.choose_n):
                continue

            total_pic_num += num_pic
            # print(num_pic)
            
            # load all pictures
            images = np.zeros([num_pic, 64, 64], dtype=np.int)

            for i, pic in enumerate(os.listdir(self.root + kanji)):
                image = Image.open(self.root + kanji + os.sep + pic)
                images[i,:] = np.array(image)
            
            images = images.reshape(num_pic, -1)
            # generate labels and its mapping
            labels = np.full([num_pic, ], num_items)
            map_class[num_items] = unicodes

            # concat as one file
            tempdata = pd.DataFrame(images)
            tempdata = pd.concat([tempdata, pd.DataFrame(labels, columns=["label"])], axis=1)
            if self.data is None:
                self.data = tempdata
            else:
                self.data = pd.concat([self.data, tempdata], axis = 0, ignore_index = True)

            num_items += 1

            # print(kanji, unicodes)
            # break # break for test

        print("total num:{}".format(total_pic_num))
        self.num_classes = num_items + 1
        self.map_class = map_class

        # auto save as a csv file
        self._save_csv()


    def _save_csv(self, root = None):
        if root is None:
            root = self.root

        # save images
        images_savepath = root + "{}_{}.csv".format(self.__class__.__name__, self.choose_n)
        self.data.to_csv(images_savepath, index = False)

        # save mappings
        mapping_savepath = root + "{}_{}_mapping.csv".format(self.__class__.__name__, self.choose_n)
        label_map = pd.DataFrame([[x,y] for x,y in self.map_class.items()], columns=["class_label","class_name"])
        label_map.astype({"class_label":int, "class_name":str})
        label_map.to_csv(mapping_savepath, index=False)

        print(label_map)

        print(" Save csv data file in {}.\n Save label mapping file in {}".format(images_savepath, mapping_savepath))


class KanjiLoadIndex(object):
    """根据Kanji数据集各类别图片数量选择样本，统计满足条件的类别的 
    classname, folder(类别对应的文件夹), num(图片数量), label为数组中的下标
    该加载方式会比csv的方式快得多。
    """
    def __init__(self, root, choose_n = 0):
        self.root = root 
        self.choose_n = choose_n
        self._load()

    def _load(self):
        num_items = 0
        map_class = {}

        self.label_mapping = []
        self.data_mapping = []
        
        # load all data
        for kanji in os.listdir(self.root):

            if not os.path.isdir(self.root + kanji):
                continue

            # the label is the name of filefolder
            unicode_str = "0x" + kanji[2:]
            unicodes = int(unicode_str, base=16).to_bytes(4, "little").decode("utf-16") # 小端，4个字节
            
            # drop class with less pictures
            num_pic = len(os.listdir(self.root + kanji))
            if not (num_pic > self.choose_n):
                continue

            self.label_mapping.append(unicodes)
            self.data_mapping.append({
                "folder": kanji,
                "num": num_pic
            })

            num_items += 1


class NKanjiDatasetForIndex(Dataset):
    """ KMNIST 的第二份数据集 Kanji2的加载， 需要传入一个KanjiLoadIndex类型的loader，使用该loader
    保存的几个mapping的参数获得样本
    """
    def __init__(self, root, loader, choose_n = 0, transforms = None , target_transforms = None, save=True):
        
        super(NKanjiDatasetForIndex, self).__init__()
        
        assert(isinstance(loader, KanjiLoadIndex))
        
        self.root = root
        self.loader = loader
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.choose_n = choose_n

        # 更新一个各种类所在的num范围
        x = 0
        self.idxrange = [ t for t in self.loader.data_mapping["num"] ]
        for i in range(1, len(self.idxrange)):
            self.idxrange[i] = self.idxrange[i] + self.idxrange[i-1]

        self.length = reduce(lambda x,y: x+y, self.idxrange)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        assert idx >= 0 and idx < self.__len__()

        # find right index
        for fidx,t in enumerate(self.idxrange):
            if(idx < t):
                break
        fidx = fidx - 1
        infidx = idx - self.idxrange[fidx]

        folder = self.loader.data_mapping[fidx]["folder"]
        imagepath = root + folder + os.sep + os.listdir(folder)[infidx]
        
        images = np.array(Image.open(imagepath))
        labels = fidx 

        if self.transforms:
            images = self.transforms(images)
        if self.target_transforms:
            labels = self.target_transforms(labels)

        return images, labels



class KanjiLoadPath(object):
    """ 读取Kanji2文件夹，选择包含的样本数量大于 choose_n 的类别，
    将对应图片的路径和label存储在csv文件里，
    将label和class name存储在另一个csv文件中。
    """
    def __init__(self, root, choose_n = 0):
        self.root = root 
        self.choose_n = choose_n
        self._load()

    def _load(self):
        num_items = 0
        map_class = {}

        self.data = None
        
        # load all data
        for kanji in os.listdir(self.root):

            if not os.path.isdir(self.root + kanji):
                continue

            # the label is the name of filefolder
            unicode_str = "0x" + kanji[2:]
            unicodes = int(unicode_str, base=16).to_bytes(4, "little").decode("utf-16") # 小端，4个字节
            
            # drop class with less pictures
            num_pic = len(os.listdir(self.root + kanji))
            if not (num_pic > self.choose_n):
                continue

            print("{}-{}".format(kanji, unicodes))
            map_class[num_items] = unicodes
            
            # save the filepath and its label and the 
            paths = [kanji + os.sep + path for path in os.listdir(self.root + kanji)]
            labels = np.full([num_pic, ], num_items)
            data = np.array([labels, paths], dtype=np.str).T

            # concat all together
            df = pd.DataFrame(data, columns=["label","image_path"], dtype=str)
            if self.data is None:
                self.data = df
            else:
                self.data = pd.concat([self.data, df])

            num_items += 1

        # label
        self.labeldf = pd.DataFrame([[i, l] for i,l in map_class.items()], columns=["label","name"], dtype=str)
        self._save()

    def _save(self):
        datasavepath = root + "{}_{}.csv".format(self.__class__.__name__, self.choose_n)
        self.data.to_csv(datasavepath, index = False)

        labelsavepath = root + "{}_{}_label.csv".format(self.__class__.__name__, self.choose_n)
        self.labeldf.to_csv(labelsavepath, index=False)


class KanjiDataset(Dataset):
    """ 使用 KanjiLoadPath 生成的对应choose_n的两种文件，加载数据集。
    """
    def __init__(self, root, choose_n = 0, transforms = None , target_transforms = None):
        
        self.root = root
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.choose_n = choose_n

        datapath = self.root + "KanjiLoadPath_{}.csv".format(self.choose_n)
        labelpath = self.root + "KanjiLoadPath_{}_label.csv".format(self.choose_n)

        self.datapath = pd.read_csv(datapath)
        labeldf = pd.read_csv(labelpath)
        self.classes = {labeldf.iloc[i,0]:labeldf.iloc[i,1]  for i in range(len(labeldf))}

    def __len__(self):
        return len(self.datapath)

    def __getitem__(self, idx):

        assert idx >= 0 and idx <= self.__len__()

        images = np.array(Image.open(self.root + self.datapath.iloc[idx,1]))
        labels = self.datapath.iloc[idx,0]

        if self.transforms is not None:
            images = self.transforms(images)

        return images, labels

"""
KanjiLoadPath(root)
"""

c = KanjiDataset(root)

image, l = c[2056]

print(l)
print(c.classes[l])

plt.imshow(image)
plt.show()