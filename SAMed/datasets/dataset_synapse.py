import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
from einops import repeat
# from icecream import ic
from PIL import Image


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


class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            # image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            # ! 3D
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        # image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample


# class Synapse_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, transform=None):
#         self.transform = transform  # using transform in torch!
#         self.split = split
#         self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
#         self.data_dir = base_dir

#     def __len__(self):
#         return len(self.sample_list)

#     def __getitem__(self, idx):
#         if self.split == "train":
#             slice_name = self.sample_list[idx].strip('\n')
#             data_path = os.path.join(self.data_dir, slice_name+'.npz')
#             data = np.load(data_path)
#             image, label = data['image'], data['label']
#         else:
#             vol_name = self.sample_list[idx].strip('\n')
#             filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
#             data = h5py.File(filepath)
#             image, label = data['image'][:], data['label'][:]

#         # Input dim should be consistent
#         # Since the channel dimension of nature image is 3, that of medical image should also be 3

#         sample = {'image': image, 'label': label}
#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.sample_list[idx].strip('\n')
#         return sample

# class Potsdam(Dataset):
#     pass # TODO


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.img_dir = os.path.join(base_dir, 'img_dir', split)
        self.ann_dir = os.path.join(base_dir, 'ann_dir', split)
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        self.masks = list(sorted(os.listdir(self.ann_dir)))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.ann_dir, self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # Normalization to [0, 1]
        img = np.array(img) / 255.0
        # Convert to CHW format
        img = np.transpose(img, (2, 0, 1))

        print(f'img.shape: {img.shape}')
        mask = Image.open(mask_path)


        np.expand_dims(np.array(mask), 0)

        
        print(f'mask.shape: {mask.shape}')

        sample = {'image': np.array(img), 'label': np.array(mask)}

        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.imgs[idx].split('.')[0]  # remove the file extension
        return sample
