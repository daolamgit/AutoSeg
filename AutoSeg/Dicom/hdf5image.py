import numpy as np
import h5py
import AutoSeg.Dicom.dicomimage as dcm
from AutoSeg.Dicom.image_constants import *

from skimage.transform import resize, warp, AffineTransform
from skimage import measure
from transforms3d.euler import euler2mat
from transforms3d.affines import compose

import keras.backend as K

class Patient_Train(object):
    def __init__(self, file, roi, im_size):
        self.roi    = roi
        self.im_size    = im_size #this is scalar

        f_h5    = h5py.File( file, 'r')
        if roi == -1:
            self.Volume_resize    = np.asarray( f_h5['Volume_resize'], dtype=np.float32)
            self.Masks_resize    = np.asarray( f_h5['Masks_resize'], dtype=np.float32)
        else:
            # 2nd pass in handling fine tune training
            pass
        f_h5.close()

        #self.Volume_augmentation    = self.augmentation() #move it out of constructor so that mean and std not augmented


    def augmentation(self):
        if self.roi == -1:
            #assert self.im_size == self.Hyper_volume.shape[2:4] #make sure we use the correct resizse image
            im_size     = self.Volume_resize.shape

            #prepare the augmentation params
            translation = [0, np.random.uniform(-8, 8), np.random.uniform(-8,8)]
            rotation    = euler2mat( np.random.uniform( -5, 5) / 180 * np.pi, 0 ,0, 'sxyz')
            scale       = [1, np.random.uniform( .9, 1.1), np.random.uniform( .9, 1.1)]
            warp_mat    = compose( translation, rotation, scale)
            tform_coords= self.get_tform_coords( im_size)
            w           = np.dot( warp_mat, tform_coords)
            w[0] = w[0] + im_size[0] / 2
            w[1] = w[1] + im_size[1] / 2
            w[2] = w[2] + im_size[2] / 2
            warp_coords     = w[0:3].reshape(3, im_size[0], im_size[1], im_size[2])

            final_images    = warp( self.Volume_resize, warp_coords)

            #warp the label
            nclass  = N_CLASSES + 1
            #channel first, but batch is always front
            #final_labels    = np.empty( (nclass,) + im_size , dtype=np.float32 )
            final_labels = np.empty((im_size[0], nclass, im_size[1], im_size[2]), dtype=np.float32)
            for z in range( 1, nclass):
                temp = warp( (self.Masks_resize == z).astype( np.float32), warp_coords)
                temp[ temp < .5] = 0
                temp[ temp >= .5] = 1
                final_labels[:,z] = temp

            #the labels will be several binary planes ,with plane 0 is the background
            final_labels[:,0] = np.amax( final_labels[:, 1:,], axis = 1) == 0

            #move axis if channel last
            if K.image_data_format()  == 'channels_last':
                final_labels = np.moveaxis( final_labels, 1, -1)

        else: #fine tune
            pass

        return final_images, final_labels

    @property
    def Hyper_volume(self):
        '''
        technically it creates 3 self params but only return 1 param
        :return:
        '''
        self.Volume_augmentation, self.Masks_augmentation = self.augmentation()
        return dcm.Patient_Train.create_hyper_volume(self.Volume_augmentation)

    @staticmethod
    def get_tform_coords( im_size):
        coords0, coords1, coords2 = np.mgrid[:im_size[0], :im_size[1], :im_size[2]]
        coords  = np.array( [coords0 - im_size[0] /2, coords1 - im_size[1]/2, coords2 - im_size[2] / 2])
        return np.append( coords.reshape(3, -1), np.ones( (1, np.prod(im_size))), axis=0) #create homo coordinate

class Patient_Test(object):
    def __init__(self, file, roi, im_size):
        self.roi = roi
        self.im_size = im_size  # this is scalar

        f_h5 = h5py.File(file, 'r')
        if roi == -1:
            self.Hyper_volume = np.asarray(f_h5['Hyper_volume'], dtype=np.float32)
        else:
            # 2nd pass in handling fine tune training
            pass
        f_h5.close()



if __name__ == '__main__':
    #pt = Patient_Train( )
    pass
