import pandas as pd
import nibabel as nib
from PIL import Image
import numpy as np


def main():
    df = pd.read_csv('dataset.csv')
    for index, row in df.iterrows():
        org_img = row['original']
        adj_img = row['adjusted']

        cta_hu_width = row['window_width_manual']
        cta_hu_level = row['window_level_manual']
        cta_hu_min = cta_hu_level - cta_hu_width
        cta_hu_max = cta_hu_level + cta_hu_width

        ## load original nifti file
        nifti = nib.load(org_img)
        org_img_data = nifti.get_fdata()
        ## window HU
        org_img_data[org_img_data < cta_hu_min] = cta_hu_min
        org_img_data[org_img_data > cta_hu_max] = cta_hu_max
        ## normalize to [0,1]
        org_img_data = (org_img_data - np.min(org_img_data))
        org_img_data = org_img_data / np.max(org_img_data)
        new_img = nib.Nifti1Image(org_img_data, nifti.affine, nifti.header)
        new_img = new_img.get_fdata()

        ## load adjusted nifti
        nifti = nib.load(adj_img)
        adj_img_data = nifti.get_fdata()
        print()
        assert (new_img == adj_img_data).all(), "Input and ouput don't match!" + str(index)


if __name__ == '__main__':
    main()
