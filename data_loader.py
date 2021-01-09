from __future__ import print_function, division
import torch
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
import imgaug.augmenters as iaa
import numpy as np


class LvoMTDataLoader(Dataset):
    """Lvo dataset."""

    def __init__(self, csv_file, transform=None, mode=None, augment=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csv_file, index_col=0)
        self.mode = mode
        self.data_frame = df[df['set'] == mode]
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.data_frame.iloc[idx, 6]
        image = np.load(img_dir)
        name = self.data_frame.iloc[idx, 0]
        level = self.data_frame.iloc[idx, 3]
        width = self.data_frame.iloc[idx, 4]
        label = (level, width)

        if self.augment and self.mode == 'train':
            image = self.augmentation(image)

        if self.transform:
            image = self.transform(image)

        return name, image, label

    def augmentation(self, img):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            # iaa.Flipud(0.5),  # vertical flips
            iaa.Crop(32),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(
                0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 512), per_channel=0.5),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)  # apply augmenters in random order

        img = np.expand_dims(img, axis=0)

        return seq(images=img)[0]


class LvoDataLoader(Dataset):
    """Lvo dataset."""

    def __init__(self, csv_file, task=None, transform=None, mode=None, augment=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csv_file, index_col=0)
        self.mode = mode
        self.data_frame = df[df['set'] == mode]
        self.transform = transform
        self.augment = augment
        self.task = task

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.data_frame.iloc[idx, 6]
        image = np.load(img_dir)
        name = self.data_frame.iloc[idx, 0]
        level = self.data_frame.iloc[idx, 3]
        width = self.data_frame.iloc[idx, 4]
        if self.task == 'level':
            label = level
        else:
            label = width

        if self.augment and self.mode == 'train':
            image = self.augmentation(image)

        if self.transform:
            image = self.transform(image)

        return name, image, label

    def augmentation(self, img):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Flipud(0.5),  # vertical flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            # iaa.Sometimes(
            #     0.5,
            #     iaa.GaussianBlur(sigma=(0, 2))
            # ),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            # iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-25, 25),
                shear=(-8, 8)
            )
        ], random_order=True)  # apply augmenters in random order

        img = np.expand_dims(img, axis=0)

        return seq(images=img)[0]

