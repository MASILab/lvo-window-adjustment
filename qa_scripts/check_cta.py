import nibabel as nib
import pandas as pd
import numpy as np

df = pd.read_csv('csv/auto_windowed_result.csv')
tracker = 0
count = 0
for idx, row in df.iterrows():
    tracker += 1
    sneha = row['cta_windowed']
    bryan = row['auto_windowed_cta']

    sneha = nib.load(sneha).get_fdata()
    bryan = nib.load(bryan).get_fdata()

    if not (sneha == bryan).all():
        print(row['cta_windowed'])
        count += 1
    else:
        print(tracker)

print(count)
