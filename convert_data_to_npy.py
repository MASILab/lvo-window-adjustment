import numpy as np
import pandas as pd
import nibabel as nib
import os

raw = pd.read_csv('csv/resplit_dataset.csv', index_col=0)

npy_dirs = []
total = len(raw)
counter = 1
for index, row in raw.iterrows():

    img_dir = row['cta']
    npy_data = np.uint8(nib.load(img_dir).get_fdata()[:, :, 120:160])

    fname = os.path.basename(img_dir)
    npy_name = fname.split('.nii')[0] + '.npy'
    npy_dir = os.path.join('data/npy', npy_name)
    np.save(npy_dir, npy_data)

    npy_dirs.append(npy_dir)
    print(f'Converting {counter}/{total}')
    counter = counter + 1

raw['cta_npy'] = npy_dirs
raw.to_csv('csv/resplit_dataset.csv', index=False)

print()