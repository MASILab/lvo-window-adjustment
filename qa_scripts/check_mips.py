import nibabel as nib
import pandas as pd
import numpy as np

df = pd.read_csv('csv/mip_test.csv')
tracker = 0
count = 0
for idx, row in df.iterrows():
    tracker += 1
    sneha = row['mip_windowed']
    bryan = row['auto_windowed_mip']

    sneha = nib.load(sneha)
    sneha = sneha.get_fdata()
    bryan = nib.load(bryan)
    bryan = bryan.get_fdata()

    if not (sneha==bryan).all():
        diff = sneha - bryan
        print(diff.min(), diff.max())
        count += 1
    else:
        print(tracker)

print(count)
