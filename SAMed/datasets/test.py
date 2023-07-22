# -*- coding: utf-8 -*-
import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
from icecream import ic
from torchvision.transforms import Normalize, ToTensor



def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, num_classes=10):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split + '.txt')).readlines()
        self.data_dir = base_dir
        self.num_classes = num_classes

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            filename = self.sample_list[idx].strip('\n')
            image_path = os.path.join(self.data_dir, 'images', filename[:-4] + '.jpg')
            label_path = os.path.join(self.data_dir, 'labels', filename[:-4] + '.png')
        else:
            filename = self.sample_list[idx].strip('\n')
            image_path = os.path.join(self.data_dir, 'images', filename[:-4] + '.jpg')
            label_path = os.path.join(self.data_dir, 'labels', filename[:-4] + '.png')

        image = Image.open(image_path).convert('RGB')  # labels是jpg
        
        label = Image.open(label_path).convert('L')  # labels是png
        label = np.asarray(label) / 255
        sample = {'image': image, 'label': label}
        sample['case_name'] = filename

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomGenerator(object):
    def __init__(self, output_size, low_res, num_classes=10):
        self.output_size = output_size
        self.low_res = low_res
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = np.array(image)  # 转换为NumPy数组
        label = np.array(label)  # 转换为NumPy数组

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y, _ = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        
        # 标签数据范围调整
        label = np.clip(label, 0, self.num_classes - 1)

        
        # 图像归一化处理
        to_tensor = ToTensor()
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image = Image.fromarray(image)
        image = normalize(to_tensor(image))
        label = torch.from_numpy(label).float()
        low_res_label = torch.from_numpy(low_res_label).float()

        sample['image'] = image
        sample['label'] = label
        sample['low_res_label'] = low_res_label

        return sample

