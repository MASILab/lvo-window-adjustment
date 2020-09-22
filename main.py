# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image

def main():
    sample_3d = '/nfs/masi/lingams/lvo/mip_d2s120t40_windowmanual_ss_betctregmask_cta_registered/train/0/' \
               'w100l100_ss_betctregmask_flirt_headonly_DICOM_AX_1MM_CTA_F_0.6_20181205081229_10_mip_d2s120t40.nii.gz'
    sample_mip = '/nfs/masi/lingams/lvo/ss_betctregmask_cta_registered/train/0/ss_betctregmask_flirt_headmanual_DICOM_AX_1MM_CTA_F_0.6_20181215171251_16.nii.gz'
    nifti = nib.load(sample_3d)
    img_data = nifti.get_fdata()
    print(img_data.shape)

    img_data *= 255
    Image.fromarray(img_data[:, :, 0].T).show()


if __name__ == '__main__':
    main()


