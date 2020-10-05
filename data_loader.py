from __future__ import print_function, division
import torch
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class LvoDataLoader(Dataset):
    """Lvo dataset."""

    def __init__(self, csv_file, transform=None, mode=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        df = pd.read_csv(csv_file)
        self.data_frame = df[df['set'] == mode]
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir = self.data_frame.iloc[idx, 5]
        image = nib.load(img_dir).get_fdata()[:, :, 120:160]
        label = self.data_frame.iloc[idx, 9]

        if self.transform:
            image = self.transform(image)

        return image, label
