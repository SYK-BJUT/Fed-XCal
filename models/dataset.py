# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """

    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):

        # if not T.functional._is_tensor_image(tensor):
        #     raise TypeError('tensor is not a torch image.')

        c, h, w = tensor.size()

        if c != 1:
            raise TypeError('only support graysclae image.')

        # print(tensor.size)

        tensor = tensor.view(c, h * w)
        idx = tensor > 0
        t = tensor[idx]

        # print(t)
        m = t.mean()
        s = t.std()
        t = t.sub_(m).div_(s + 1e-6)
        tensor[idx] = t

        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats=self.outchannels, dim=0)

        return tensor


import numpy as np
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
import cv2


class MyDataset(data.Dataset):
    '''
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    [batch, outchannels, imside, imside]
    '''

    def __init__(self, txt, transforms=None, train=True, imside=128, outchannels=1):

        self.train = train

        self.imside = imside  # 128, 224
        self.chs = outchannels  # 1, 3

        self.text_path = txt

        self.transforms = transforms

        if transforms is None:
            if not train:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)

                ])
            else:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),  # 0.3 0.35
                        T.RandomResizedCrop(size=self.imside, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),  # (0.1, 0.2) (0.05, 0.05)
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, expand=False, center=(0.5 * self.imside, 0.0)),
                            T.RandomRotation(degrees=10, expand=False, center=(0.0, 0.5 * self.imside)),
                        ]),
                    ]),

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)
                ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []

        txt_file = self.text_path

        # 【修改】指定utf-8编码读取，支持中文路径
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_label.append(item[1])

    # 【新增】支持中文路径的图片加载函数
    def _load_image(self, img_path):
        """使用cv2加载图片以支持中文路径"""
        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return Image.fromarray(img)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_label[index]

        idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])

        if self.train == True:
            idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])

        img_path2 = self.images_path[idx2]

        # 【修改】使用新的加载函数替代Image.open()
        data = self._load_image(img_path)
        data = self.transforms(data)

        data2 = self._load_image(img_path2)
        data2 = self.transforms(data2)

        data = [data, data2]

        return data, int(label)

    def __len__(self):
        return len(self.images_path)


class MyDataset2(data.Dataset):
    '''
    Load and process the ROI images with 3 views::
    
    INPUT::
    txt: a text file containing pathes & labels of the input images (主数据集)
    txt_auxiliary: a text file for the third view (辅助数据集)
    transforms: None
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image

    OUTPUT::
    返回3个视图：[data1, data2, data3], label
    - data1: 原始图片（来自主数据集）
    - data2: 同类样本（来自主数据集）
    - data3: 同类样本（来自辅助数据集）
    '''

    def __init__(self, txt, txt_auxiliary, transforms=None, train=True, imside=128, outchannels=1):

        self.train = train

        self.imside = imside
        self.chs = outchannels

        self.text_path = txt
        self.text_path_auxiliary = txt_auxiliary

        self.transforms = transforms

        if transforms is None:
            if not train:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)

                ])
            else:
                self.transforms = T.Compose([

                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                        T.RandomResizedCrop(size=self.imside, scale=(0.8, 1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, expand=False, center=(0.5 * self.imside, 0.0)),
                            T.RandomRotation(degrees=10, expand=False, center=(0.0, 0.5 * self.imside)),
                        ]),
                    ]),

                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)
                ])


        self._read_txt_file()

    def _read_txt_file(self):

        self.images_path = []
        self.images_label = []

        with open(self.text_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                self.images_label.append(item[1])


        self.images_path_auxiliary = []
        self.images_label_auxiliary = []

        with open(self.text_path_auxiliary, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                item = line.strip().split(' ')
                self.images_path_auxiliary.append(item[0])
                self.images_label_auxiliary.append(item[1])

    def _load_image(self, img_path):

        img_array = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        return Image.fromarray(img)

    def __getitem__(self, index):

        img_path = self.images_path[index]
        label = self.images_label[index]


        if self.train:
            idx2 = np.random.choice(np.arange(len(self.images_label))[np.array(self.images_label) == label])
        else:
            idx2 = index
        img_path2 = self.images_path[idx2]


        auxiliary_same_class_indices = np.arange(len(self.images_label_auxiliary))[
            np.array(self.images_label_auxiliary) == label
        ]
        
        if len(auxiliary_same_class_indices) > 0:
            if self.train:
                idx3 = np.random.choice(auxiliary_same_class_indices)
            else:
                idx3 = auxiliary_same_class_indices[0]
            img_path3 = self.images_path_auxiliary[idx3]
        else:

            img_path3 = img_path2

        # 加载并增强3个视图
        data1 = self._load_image(img_path)
        data1 = self.transforms(data1)

        data2 = self._load_image(img_path2)
        data2 = self.transforms(data2)

        data3 = self._load_image(img_path3)
        data3 = self.transforms(data3)

        data = [data1, data2, data3]

        return data, int(label)

    def __len__(self):
        return len(self.images_path)
