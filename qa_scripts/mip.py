#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Function to make maximum intensity projection (MIP) using SimpleITK, Adapted by Sneha Lingam from:
https://gist.github.com/fepegar/a8814ff9695c5acd8dda5cf414ad64ee#file-mip-py
Created on Fri Nov 10 16:51:47 2017
@author: fernando
"""

import os
import numpy as np
import SimpleITK as sitk

def make_mips(image_path, output_dir,startslice, thickness, dim):
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

    return out_name