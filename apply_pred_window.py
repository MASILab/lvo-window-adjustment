import pandas as pd
import nibabel as nib
import numpy as np
import os
import copy

prepare_csv = True
if prepare_csv:
    data = 'csv/resplit_dataset.csv'
    pred = 'results/resnet_mt_2fc_reg_chunks/reduced_test.csv'

    df_meta = pd.read_csv(data)
    df_meta = df_meta[df_meta['set'] == 'test']
    df_meta = df_meta.loc[:, ['subj', 'cta', 'cta_windowed']]

    df_pred = pd.read_csv(pred, index_col=0)
    df_merged = df_pred.merge(df_meta, left_on='subj', right_on='subj')
    df_merged.to_csv('csv/pre_apply.csv')

df = pd.read_csv('csv/pre_apply.csv')

auto_windowed_dir = []
for idx, row in df.iterrows():
    raw_ct_dir = row['cta']
    out_f_dir = os.path.join('data/predicted', 'auto_' + os.path.basename(raw_ct_dir))
    if not os.path.exists(out_f_dir):
        nifti = nib.load(raw_ct_dir)
        raw_ct = nifti.get_fdata()

        pred_level = int(row['prediction_level'])
        pred_width = int(row['prediction_width'])
        cta_hu_min = pred_level - pred_width
        cta_hu_max = pred_level + pred_width

        ## window HU
        raw_ct[raw_ct < cta_hu_min] = cta_hu_min
        raw_ct[raw_ct > cta_hu_max] = cta_hu_max
        ## normalize to [0,1]
        adj_data = (raw_ct - np.min(raw_ct))
        adj_data = adj_data / np.max(adj_data)
        ## save changed image as nifti file
        new_img = nib.Nifti1Image(adj_data, nifti.affine, nifti.header)
        nib.save(new_img, out_f_dir)
    else:
        print('File already exists...')
    print('Windowing {} / 60 ...'.format(idx + 1))
    auto_windowed_dir.append(out_f_dir)

df_out = copy.deepcopy(df)
df_out['auto_windowed_cta'] = auto_windowed_dir
df_out.to_csv('csv/auto_windowed_result.csv')

