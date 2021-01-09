import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
import copy
import nibabel as nib


def main():
    mip_rewindow = True
    cta = 'csv/auto_windowed_result.csv'
    df = pd.read_csv(cta, index_col=0)
    mip_dir = 'data/mip'
    mip_dir_hist = []
    for idx, row in df.iterrows():
        auto_cta = row['auto_windowed_cta']
        mip = make_mips(idx, auto_cta, mip_dir, 120, 40, 2)
        mip_dir_hist.append(mip)

        if mip_rewindow:
            ## reset window to width 1 level 0.5 and re-normalize
            nifti = nib.load(mip)
            img_data = nifti.get_fdata()
            img_data[img_data < 0] = 0
            img_data[img_data > 1] = 1
            img_data = (img_data - np.min(img_data))
            img_data = img_data / np.max(img_data)
            new_img = nib.Nifti1Image(img_data, nifti.affine, nifti.header)
            nib.save(new_img, mip)

    df_out = copy.deepcopy(df)
    df_out['auto_windowed_mip'] = mip_dir_hist
    df_out.to_csv('csv/auto_windowed_mip_result.csv')

    # combine with original csv
    manual = 'csv/dataset.csv'
    auto = 'csv/auto_windowed_mip_result.csv'

    df_manual = pd.read_csv(manual, index_col=0)
    df_manual = df_manual[df_manual['set'] == 'test']
    df_manual = df_manual.loc[:, ['subj', 'mip', 'mip_windowed']]

    df_auto = pd.read_csv(auto, index_col=0)
    df_merged = df_auto.merge(df_manual, left_on='subj', right_on='subj')
    df_merged.to_csv('csv/windowed_mip_result.csv')


def make_mips(idx, image_path, output_dir, startslice, thickness, dim):
    image = sitk.ReadImage(image_path)
    image = image[:,:,startslice:startslice+thickness]
    image_size = image.GetSize()

    basename = os.path.basename(image_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    proj_basename = basename.replace('.nii.gz', '_mip_{}.nii.gz'.format('d'+str(dim)+'s'+str(startslice)+'t'+str(thickness)))
    out_name = os.path.join(output_dir, proj_basename)

    if not os.path.exists(out_name):
        projection = sitk.MaximumProjection(image, dim)

        if image_size[dim] % 2:  # odd number
            voxel = [0, 0, 0]
            voxel[dim] = (image_size[dim] - 1) / 2
            origin = image.TransformIndexToPhysicalPoint(voxel)
        else:  # even
            voxel1 = np.array([0, 0, 0], int)
            voxel2 = np.array([0, 0, 0], int)
            voxel1[dim] = image_size[dim] / 2 - 1
            voxel2[dim] = image_size[dim] / 2
            point1 = np.array(image.TransformIndexToPhysicalPoint(voxel1.tolist()))
            point2 = np.array(image.TransformIndexToPhysicalPoint(voxel2.tolist()))
            origin = np.mean(np.vstack((point1, point2)), 0)
        projection.SetOrigin(origin)
        projection.SetDirection(image.GetDirection())

        sitk.WriteImage(projection, out_name)

    else:
        print(out_name, 'already exists')

    print('Generating MIPs {} / 60 ...'.format(idx + 1))

    return out_name


if __name__ == '__main__':
    main()
