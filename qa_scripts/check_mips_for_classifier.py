import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('/nfs/masi/luy8/auto_mips.csv', index_col=0)

sub_df = df[0:4]

fig = plt.figure(figsize=(9, 12))
gs = fig.add_gridspec(4, 3, hspace=0.1, wspace=0)
axs = gs.subplots()
for idx, row in sub_df.iterrows():
    manual = row['file_path']
    auto = row['auto_window_file_path']

    manual_mip = nib.load(manual).get_fdata()
    auto_mip = nib.load(auto).get_fdata()

    axs[idx, 0].set_ylabel(row['subj'], rotation=0, labelpad=30)
    axs[idx, 0].imshow(manual_mip[:, :, 0].T, cmap='gray')
    axs[idx, 1].imshow(auto_mip[:, :, 0].T, cmap='gray')


axs[0, 0].set_title('manual')
axs[0, 1].set_title('auto')
for ax in fig.get_axes():
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

plt.show()