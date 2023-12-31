# dataset_synapse.py

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

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        x, y = image.shape[1], image.shape[2]
        image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=1)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)  # Remove the '1,'

        image = image.astype(np.float32)

        # Compute low_res_label
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)  # Remove the '1,'

        return {'image': image, 'label': label, 'low_res_label': low_res_label}


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.img_dir = os.path.join(base_dir, 'img_dir', split)
        self.ann_dir = os.path.join(base_dir, 'ann_dir', split)
        # self.imgs = list(sorted(os.listdir(self.img_dir)))
        # self.masks = list(sorted(os.listdir(self.ann_dir)))
        self.imgs = [f for f in sorted(os.listdir(self.img_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.masks = [f for f in sorted(os.listdir(self.ann_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]

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

        mask = Image.open(mask_path)
        mask = np.array(mask)
        # Ensure the mask has the same size with img
        if mask.shape != img.shape[1:]:
            mask = np.resize(mask, img.shape[1:])

        sample = {'image': img, 'label': mask}

        if self.transform:
            sample = self.transform(sample)

        # Ensure the label is a 2D tensor
        sample['label'] = np.squeeze(sample['label'])

        sample['case_name'] = self.imgs[idx].split('.')[0]  # remove the file extension
        return sample



# class RandomGenerator(object):
#     def __init__(self, output_size, low_res):
#         self.output_size = output_size
#         self.low_res = low_res

#     # def __call__(self, sample):
#     #     image, label = sample['image'], sample['label']

#     #     if random.random() > 0.5:
#     #         image, label = random_rot_flip(image, label)
#     #     elif random.random() > 0.5:
#     #         image, label = random_rotate(image, label)
#     #     x, y, _ = image.shape
#     #     if x != self.output_size[0] or y != self.output_size[1]:
#     #         # image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
#     #         # label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
#     #         # ! 3D
#     #         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=3)
#     #         label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

#     #     label_h, label_w = label.shape
#     #     low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
#     #     image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
#     #     # image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
#     #     label = torch.from_numpy(label.astype(np.float32))
#     #     low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
#     #     sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
#     #     return sample

#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         x, y = image.shape[1], image.shape[2]
#         image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y, 1), order=1)
#         label = zoom(label, (1, self.output_size[0] / x, self.output_size[1] / y), order=0)
#         return {'image': image, 'label': label}



# # class Synapse_dataset(Dataset):
# #     def __init__(self, base_dir, list_dir, split, transform=None):
# #         self.transform = transform  # using transform in torch!
# #         self.split = split
# #         self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
# #         self.data_dir = base_dir

# #     def __len__(self):
# #         return len(self.sample_list)

# #     def __getitem__(self, idx):
# #         if self.split == "train":
# #             slice_name = self.sample_list[idx].strip('\n')
# #             data_path = os.path.join(self.data_dir, slice_name+'.npz')
# #             data = np.load(data_path)
# #             image, label = data['image'], data['label']
# #         else:
# #             vol_name = self.sample_list[idx].strip('\n')
# #             filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
# #             data = h5py.File(filepath)
# #             image, label = data['image'][:], data['label'][:]

# #         # Input dim should be consistent
# #         # Since the channel dimension of nature image is 3, that of medical image should also be 3

# #         sample = {'image': image, 'label': label}
# #         if self.transform:
# #             sample = self.transform(sample)
# #         sample['case_name'] = self.sample_list[idx].strip('\n')
# #         return sample

# # class Potsdam(Dataset):
# #     pass # TODO


# class Synapse_dataset(Dataset):
#     def __init__(self, base_dir, list_dir, split, transform=None):
#         self.transform = transform
#         self.split = split
#         self.img_dir = os.path.join(base_dir, 'img_dir', split)
#         self.ann_dir = os.path.join(base_dir, 'ann_dir', split)
#         self.imgs = list(sorted(os.listdir(self.img_dir)))
#         self.masks = list(sorted(os.listdir(self.ann_dir)))

#     def __len__(self):
#         return len(self.imgs)

#     # def __getitem__(self, idx):
#     #     img_path = os.path.join(self.img_dir, self.imgs[idx])
#     #     mask_path = os.path.join(self.ann_dir, self.masks[idx])
#     #     img = Image.open(img_path).convert("RGB")
#     #     # Normalization to [0, 1]
#     #     img = np.array(img) / 255.0
#     #     # Convert to CHW format
#     #     img = np.transpose(img, (2, 0, 1))

#     #     print(f'img.shape: {img.shape}')
#     #     mask = Image.open(mask_path)

#     #     mask = np.expand_dims(np.array(mask), 0)

#     #     print(f'mask.shape: {mask.shape}')

#     #     sample = {'image': np.array(img), 'label': np.array(mask)}

#     #     if self.transform:
#     #         sample = self.transform(sample)
#     #     sample['case_name'] = self.imgs[idx].split('.')[0]  # remove the file extension
#     #     return sample
    
#     # def __getitem__(self, idx):
#     #     img_path = os.path.join(self.img_dir, self.imgs[idx])
#     #     mask_path = os.path.join(self.ann_dir, self.masks[idx])
#     #     img = Image.open(img_path).convert("RGB")
#     #     # Normalization to [0, 1]
#     #     img = np.array(img) / 255.0
#     #     # Convert to CHW format
#     #     img = np.transpose(img, (2, 0, 1))

#     #     mask = Image.open(mask_path)
#     #     mask = np.array(mask)
#     #     # Convert to CHW format
#     #     mask = np.expand_dims(mask, 0)  # Add an extra dimension for the channel

#     #     sample = {'image': img, 'label': mask}

#     #     if self.transform:
#     #         # Modify the zoom factors to match the dimensions of the image and mask
#     #         zoom_factors = self.transform.zoom_factors
#     #         zoom_factors = zoom_factors + [1] * (img.ndim - len(zoom_factors))
#     #         self.transform.zoom_factors = zoom_factors
#     #         sample = self.transform(sample)
#     #     sample['case_name'] = self.imgs[idx].split('.')[0]  # remove the file extension
#     #     return sample

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.imgs[idx])
#         mask_path = os.path.join(self.ann_dir, self.masks[idx])
#         img = Image.open(img_path).convert("RGB")
#         # Normalization to [0, 1]
#         img = np.array(img) / 255.0
#         # Convert to CHW format
#         img = np.transpose(img, (2, 0, 1))

#         mask = Image.open(mask_path)
#         mask = np.array(mask)

#         # Ensure the mask has the same size with img
#         if mask.shape != img.shape[1:]:
#             mask = np.resize(mask, img.shape[1:])

#         sample = {'image': img, 'label': mask}

#         if self.transform:
#             sample = self.transform(sample)
#         sample['case_name'] = self.imgs[idx].split('.')[0]  # remove the file extension
#         return sample