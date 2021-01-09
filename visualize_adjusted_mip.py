import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

prepare_csv = True
if prepare_csv:
    manual = 'csv/dataset.csv'
    auto = 'csv/auto_windowed_mip_result.csv'

    df_manual = pd.read_csv(manual, index_col=0)
    df_manual = df_manual[df_manual['set'] == 'test']
    df_manual = df_manual.loc[:, ['subj', 'mip', 'mip_windowed']]

    df_auto = pd.read_csv(auto, index_col=0)
    df_merged = df_auto.merge(df_manual, left_on='subj', right_on='subj')
    df_merged.to_csv('csv/windowed_mip_result.csv')

df = pd.read_csv('csv/windowed_mip_result.csv', index_col=0)

sub_df = df[0:4]

fig = plt.figure(figsize=(9, 12))
gs = fig.add_gridspec(4, 3, hspace=0.1, wspace=0)
axs = gs.subplots()
for idx, row in sub_df.iterrows():
    raw = row['mip']
    manual = row['mip_windowed']
    auto = row['auto_windowed_mip']

    raw_mip = nib.load(raw).get_fdata()
    manual_mip = nib.load(manual).get_fdata()
    auto_mip = nib.load(auto).get_fdata()

    axs[idx, 0].set_ylabel(row['subj'], rotation=0, labelpad=30)
    axs[idx, 0].imshow(raw_mip[:, :, 0].T, cmap='gray')
    axs[idx, 1].imshow(manual_mip[:, :, 0].T, cmap='gray')
    axs[idx, 2].imshow(auto_mip[:, :, 0].T, cmap='gray')

axs[0, 0].set_title('raw')
axs[0, 1].set_title('manual')
axs[0, 2].set_title('auto')
for ax in fig.get_axes():
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

plt.show()