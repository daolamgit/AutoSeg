import os
import h5py
import glob
import pydicom as dicom
import numpy as np

from skimage.transform import resize
from skimage import measure

from AutoSeg.Dicom.image_constants import *

import matplotlib.pyplot as plt

import nibabel as nib

def convert2original(Mask, size_origin, PixelSpacing):
    '''
    convert from NN output to: z,224, 224 -> Z, 512,512 -> crop?pad? -> (Z,?,?) -> original
    size_orgin is a tube of (D, H, W)
    :return:
    '''

    # pad
    Mask_original_size = np.zeros(size_origin)

    Mask = np.pad(Mask, ((0, 0), (CROP, CROP), (CROP, CROP)), 'constant')
    (N, H, D) = size_origin

    for label in range(N_CLASSES):
        mask = Mask == label + 1

        # clean some artifact noise, this may be critical
        # mask = clean_mask(mask)

        # rescale to INPLANT 512
        mask = resize(mask, (mask.shape[0], INPLANE_SIZE, INPLANE_SIZE))
        mask[mask < .5] = 0
        mask[mask >= .5] = 1

        # crop?pad?
        inplane_scale = PixelSpacing/PIXEL_SPACING #slices[0].PixelSpacing[0] / PIXEL_SPACING
        inplane_size = int(np.rint(inplane_scale * H / 2) * 2)
        if inplane_size > INPLANE_SIZE:
            # pad
            pad = (inplane_size - INPLANE_SIZE) / 2
            mask = np.pad(mask, ((0, 0), (pad, pad), (pad, pad)), 'constant')
        else:  # crop
            crop = - (inplane_size - INPLANE_SIZE) / 2
            mask = mask[:, crop: INPLANE_SIZE - crop, crop: INPLANE_SIZE - crop]

        # scale to original size
        if mask.shape != (N, H, W):
            mask = resize(mask, (N, H, W))
            # mask[mask < .5] = 0
            # mask[mask >= .5] = 1

        # convert to labels
        Mask_original_size[mask >= .5] = label + 1

    # compute_middle_size
    # crop_pad_2_midde_size
    # rescale

    return Mask_original_size

def clean_mask(mask):
    '''
    Is there a function to remove small noise without remove the disconnected contour due to
    not perfect segmentation
    aapp. per slice or volume
    :param mask:
    :return:
    '''
    labels = measure.label(mask) #label the rois in a bin mask
    area = []
    for l in range(1, np.amax(labels) + 1):
        area.append( np.sum(labels == l))

    out_contour = mask #it is just a copy
    out_contour[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0 #only get the largest obj

    return out_contour

if __name__ == '__main__':
    test_data_folder = 'Data/SegThor/test' #to get some original info
    mask_hdf5_folder = 'Data/Hdf5-SegThor-channels-last/test'

    subjects = sorted(glob.glob( os.path.join( mask_hdf5_folder, '*.hdf5.mask')))

    for sub in subjects:
        #get the output name e.g 'Data/../test/Patient11.nii.gz.hdf5.mask' ->Patient11.nii
        basename = os.path.basename( sub)
        basenames = basename.split('.')
        out_name = os.path.join( mask_hdf5_folder, basenames[0]+'.nii')
        image_name = os.path.join( test_data_folder, basenames[0]+ '.nii.gz')

        print out_name

        #get image infor
        nib_data = nib.load( image_name)
        origin_size = nib_data.shape
        PixelSpacing = abs(nib_data.affine[0][0])

        #swap
        (H,W,D) =  origin_size

        #get mask
        f = h5py.File( sub, 'r')
        mask_prob = f['Masks_predict']
        mask = np.argmax( mask_prob, axis=-1)


        #convert hdf5 mask to real mask
        Mask_org_size = convert2original(mask, (D, H, W), PixelSpacing)
        #move axis to match nibabel data format
        Mask_org_size = np.swapaxes(Mask_org_size, -1, 0)

        #save to nii
        nib_data_mask = nib.Nifti1Image( Mask_org_size, nib_data.affine)
        nib_data_mask.to_filename( out_name)