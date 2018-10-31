import glob
import os
import cv2
import numpy as np

from keras_contrib.applications.densenet import DenseNetFCN

from metrics import *

#from AutoSeg.Models.metrics import *


if __name__ == '__main__':
    #create input
    #get fom tumorcontouring lungpatint lungs test train
    #create label
    # train_folder    = '/media/radonc/OS/Users/dlam/Data/TumorContouring/LungPatients/LungPatientsData/train_small'
    # test_folder     = '/media/radonc/OS/Users/dlam/Data/TumorContouring/LungPatients/LungPatientsData/test_small'
    train_folder    = '/media/radonc/OS/Users/dlam/Data/TumorContouring/LungPatients/EsophagusZoom/train_small'
    test_folder     = '/media/radonc/OS/Users/dlam/Data/TumorContouring/LungPatients/EsophagusZoom/test_small'

    Size = 256

    train_list = glob.glob( os.path.join( train_folder, '*.tif'))
    N = len( train_list) /2
    imgs0 = np.ndarray( (N, 1, Size, Size), dtype= np.float)
    # masks0 = np.ndarray( (N, 512, 512, 1), dtype= np.float) #due to the top layer switch channel/no_class
    masks0 = np.ndarray((N, 1, Size, Size), dtype=np.float)  # due to the top layer switch channel/no_class
    i = 0
    for filename in train_list:
        if 'mask' in filename:
            continue
        else:
            mask_name = filename.split('.')[0] + '_mask.tif'
            img     = cv2.imread( filename, cv2.IMREAD_GRAYSCALE)
            mask    = cv2.imread( mask_name, cv2.IMREAD_GRAYSCALE)/255.0 #because white is 255

            img     = cv2.resize( img, (Size, Size))
            mask    = cv2.resize(mask, (Size, Size))
            mask[mask>=0.5] = 1.0
            mask[mask<.5]   = 0.0

            imgs0[i] = img
            # masks0[i,:,:,0] = mask
            masks0[i] = mask
            i += 1

    test_list = glob.glob(os.path.join(train_folder, '*.tif'))
    N = len(test_list) / 2
    imgs1 = np.ndarray((N, 1, Size, Size), dtype=np.float)
    masks1 = np.ndarray((N, 1, Size, Size), dtype=np.float)  # due to the top layer switch channel/no_class
    i = 0
    for filename in test_list:
        if 'mask' in filename:
            continue
        else:
            mask_name = filename.split('.')[0] + '_mask.tif'
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)/255.0

            img     = cv2.resize( img, (Size, Size))
            mask    = cv2.resize(mask, (Size, Size))
            mask[mask>=0.5] = 1.0
            mask[mask<.5]   = 0.0

            imgs1[i] = img
            masks1[i] = mask
            i += 1


    #split train test

    #train model
    model = DenseNetFCN( (1, Size, Size), nb_dense_block = 4, nb_layers_per_block= [ 3, 4, 5, 6, 6], upsampling_type= 'deconv')

    print model.summary()
    # model.compile( optimizer = 'Adam', loss = cross_entropy_densenet_fcn, metrics=['accuracy'])
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy', dice_coef]) #working now
    #
    model.fit( imgs0, masks0, batch_size = 1, epochs=400)

    #mask2 = model.predict( imgs1, batch_size=4)
    x = model.evaluate( imgs1, masks1, batch_size= 1)
    print x,  " All is welll"


    #plot train test
