import pandas as pd
import numpy as np
import os
import copy

data = 'csv/resplit_dataset.csv'

df = pd.read_csv(data)

df_new = df.loc[:, 'subj':'cta_npy']

all_chunks = []
for idx, row in df_new.iterrows():
    np_volume = row['cta_npy']
    volume = np.load(np_volume)
    chunks = []
    for i in range(volume.shape[2]):
        chunk = volume[:, :, i]
        volume_name = os.path.basename(np_volume).split('.npy')[0]
        chunk_name = volume_name + '_' + str(i) + '.npy'
        chunk_dir = os.path.join('data/npy_chunks', chunk_name)
        np.save(chunk_dir, chunk)
        chunk_row = copy.deepcopy(row)
        chunk_row['cta_npy'] = chunk_dir
        chunks.append(chunk_row)

    all_chunks += chunks
    print('Finish processing volume {}'.format(idx + 1))

chunks_df_cols = df_new.columns
chunk_df = pd.DataFrame(columns=chunks_df_cols, data=all_chunks)
chunk_df.to_csv('csv/resplit_dataset_in_chunks.csv')