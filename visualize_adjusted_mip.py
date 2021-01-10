import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('csv/windowed_mip_result.csv', index_col=0)

sub_df = df[0:4]
sub_df = sub_df.reset_index()

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

plt.savefig('demo/demo.png')