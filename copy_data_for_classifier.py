import pandas as pd
import os
import shutil
import copy

df_dest = pd.read_csv('/nfs/masi/remedilw/stuff_for_bryan/paths_for_bryan.csv', index_col=0)
df_local = pd.read_csv('csv/windowed_mip_result.csv', index_col=0)

df_dest = df_dest.merge(df_local.loc[:, ['subj', 'auto_windowed_mip']], left_on='subj', right_on='subj')

df_merge = copy.deepcopy(df_dest)
df_merge['auto_window_file_path'] = df_dest['auto_windowed_mip']
df_merge = df_merge.drop(columns=['auto_windowed_mip'])

dest_dirs = []
for idx, row in df_merge.iterrows():
    root_dir = '/nfs/masi/luy8/mips'
    file_name = os.path.basename(row['auto_window_file_path'])
    dest_dir = os.path.join(root_dir, file_name)
    shutil.copy(row['auto_window_file_path'], dest_dir)
    dest_dirs.append(dest_dir)

df_merge['auto_window_file_path'] = dest_dirs

df_merge.to_csv('/nfs/masi/luy8/auto_mips.csv')

