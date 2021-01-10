import tqdm
import pandas as pd
import os
from qa_scripts.mip import make_mips
import nibabel as nib
import numpy as np

mip_rewindow = True
df = pd.read_csv('csv/mip_test.csv')

## Make mips using code that uses SimpleITK's maximum projection function
for idx, row in df.iterrows():
    img_path = row['cta_windowed']
    out_path = row['auto_windowed_mip']
    if os.path.exists(img_path) and (not img_path=='') and (not os.path.exists(out_path)):
        make_mips(img_path, os.path.dirname(out_path),120,40,2)
        if mip_rewindow:
            ## reset window to width 1 level 0.5 and re-normalize
            nifti = nib.load(row['auto_windowed_mip'])
            img_data = nifti.get_fdata()
            img_data[img_data < 0] = 0
            img_data[img_data > 1] = 1
            img_data = (img_data - np.min(img_data))
            img_data = img_data/np.max(img_data)
            new_img = nib.Nifti1Image(img_data, nifti.affine, nifti.header)
            nib.save(new_img, row['auto_windowed_mip'])