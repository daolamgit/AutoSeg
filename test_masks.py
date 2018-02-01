import os
import h5py
import glob
import numpy as np

import matplotlib.pyplot as plt

res_path = 'aapm_journal_localtest/Small/Hdf5/test'
mask_pred_paths = glob.glob( os.path.join( res_path, '*.mask'))

Mask_preds    = []
for path in mask_pred_paths:
    f_h5 = h5py.File( path, 'r')
    Mask_prob = np.asarray( f_h5['Masks_predict'], dtype = np.float32)
    Mask = np.argmax( Mask_prob, axis = 1 )
    Mask_preds.append( Mask)



mask_paths = glob.glob( os.path.join( res_path, '*.hdf5'))
Mask_gts    = []

Volumes     = []
for path in mask_paths:
    f_h5 = h5py.File( path, 'r')
    Mask_gt = np.asarray( f_h5['Masks_resize'], dtype = np.float32)
    Mask_gts.append( Mask_gt)
    Volume  = np.asarray( f_h5['Volume_resize'])
    Volumes.append( Volume)


f, (ax1, ax2, ax3) = plt.subplots( 1, 3, sharey=True)
for i in range( len( Mask_gts)):
    for j in range( len( Mask_gts[i])):
        ax1.imshow( Mask_gts[i][j])
        plt.title( "GT", loc = 'left')
        ax2.imshow(Mask_preds[i][j])
        plt.title( "Slice {}".format( j))
        plt.title("Predict", loc = 'right')

        ax3.imshow( Volumes[i][j])
        plt.pause(.1)