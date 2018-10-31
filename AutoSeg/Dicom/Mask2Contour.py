import os
import h5py
import glob
import pydicom as dicom
import numpy as np

from skimage.transform import resize
from skimage import measure

from image_constants import *
import matplotlib.pyplot as plt

from rtstructures import RS_Create

class MaskToContour():
    def __init__(self, Mask, Slices ):
        self.Mask = mask
        self.Slices = Slices

        self.Mask_original = self.convert2original()

        self.Contours2D = self.contour_extract()
        self.Contours3D = self.convert2xyz()

    def convert2original(self):
        '''
        convert from NN output to: z,224, 224 -> Z, 512,512 -> crop?pad? -> (Z,?,?) -> original
        :return:
        '''

        #pad
        Mask_original_size = np.zeros( (len( self.Slices), self.Slices[0].Rows, self.Slices[0].Columns) )
        Mask = self.Mask
        Mask = np.pad( Mask, ( (0, 0), (CROP, CROP), (CROP, CROP)), 'constant')

        for label in range( N_CLASSES):
            mask = Mask == label + 1

            #clean some artifact noise
            mask = self.clean_mask( mask)

            #rescale to INPLANT 512
            mask = resize( mask, ( mask.shape[0], INPLANE_SIZE, INPLANE_SIZE) )
            mask[ mask <.5]     = 0
            mask[mask >= .5]    = 1

            #crop?pad?
            inplane_scale  = slices[0].PixelSpacing[0] / PIXEL_SPACING
            inplane_size    = int( np.rint( inplane_scale * self.Slices[0].Rows /2) * 2)
            if inplane_size > INPLANE_SIZE:
                #pad
                pad = (inplane_size - INPLANE_SIZE) / 2
                mask = np.pad( mask, ((0, 0), (pad, pad), (pad, pad)), 'constant')
            else: #crop
                crop = - (inplane_size - INPLANE_SIZE) / 2
                mask = mask[:, crop : INPLANE_SIZE - crop, crop : INPLANE_SIZE - crop ]

            #scale to original size
            if mask.shape  != ( len( self.Slices), self.Slices[0].Rows, self.Slices[0].Columns):
                mask = resize( mask, ( len( self.Slices), self.Slices[0].Rows, self.Slices[0].Columns) )
                # mask[mask < .5] = 0
                # mask[mask >= .5] = 1

            #convert to labels
            Mask_original_size[mask >= .5] = label + 1

        # compute_middle_size
        # crop_pad_2_midde_size
        # rescale

        return Mask_original_size

    def clean_mask(self ,mask):
        '''
        Is there a function to remove small noise without remove the disconnected contour due to
        not perfect segmentation
        aapp. per slice or volume
        :param mask:
        :return:
        '''
        labels = measure.label(mask)
        area = []
        for l in range(1, np.amax(labels) + 1):
            area.append( np.sum(labels == l))
        out_contour = mask
        out_contour[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0

        return out_contour

    def contour_extract(self):
        # for i in range( len(self.Mask_original)): #display for fun
        #     for c in range( N_CLASSES):
        #         mask = self.Mask_original[i] == c+1
        #
        #         contours_c = measure.find_contours( mask ,.9)
        #
        #         fig, ax = plt.subplots()
        #         ax.imshow(self.Mask_original[i], interpolation='nearest', cmap=plt.cm.gray)
        #
        #         for n, contour in enumerate(contours_c):
        #             ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
        #
        #         ax.axis('image')
        #         ax.set_xticks([])
        #         ax.set_yticks([])
        #         plt.show()

        #write to 2D pixel contour
        contours = []
        for c in range( N_CLASSES):
            contour = {}
            contour['name'] = ROI_ORDER[c]
            contour['number'] = c + 1
            contour['contour'] = []
            contour['indice']  = []
            #contour slice
            for i in range( len( self.Mask_original)):
                mask = self.Mask_original[i] == c + 1
                contours_c = measure.find_contours( mask, .9) #may have multi contour, could be a problem of

                if contours_c:
                    contour_agg = np.array([], dtype='float').reshape(0, 2) #in case multiple contours
                    for n, contour_c in enumerate( contours_c):
                        if len( contours_c) >3 and len( contour_c) < 10:
                            continue
                        contour_agg = np.concatenate( ( contour_agg, contour_c), axis = 0)
                    contour['contour'].append( contour_agg)
                    contour['indice'].append( i)

            contours.append( contour)

        return contours

    def convert2xyz(self):
        '''
        convert from 2D to 3D with z
        :return:
        '''
        Contours3D = self.Contours2D

        pos_r       = self.Slices[0].ImagePositionPatient[1]
        spacing_r   = self.Slices[0].PixelSpacing[1]
        pos_c       = self.Slices[0].ImagePositionPatient[0]
        spacing_c   = self.Slices[0].PixelSpacing[0]

        SliceThickness          = abs( self.Slices[1].ImagePositionPatient[2] - self.Slices[0].ImagePositionPatient[2])
        z0          = self.Slices[0].ImagePositionPatient[2]

        for roi in range( len(self.Contours2D)):
            for c in range( len( self.Contours2D[roi]['contour'])):
                pixels = self.Contours2D[roi]['contour'][c]

                #find z

                y = pixels[:, 0] * spacing_r + pos_r
                x = pixels[:, 1] * spacing_c + pos_c
                z1 = z0 + self.Contours2D[roi]['indice'][c] * SliceThickness
                z = [z1] * len( y)
                Contours3D[roi]['contour'][c] = np.array( (x,y,z)).transpose().flatten()

        return Contours3D

    def compare2groundtruth(self):
        '''
        the problem of nested computation is I need to go the the whole process to get this
        Spin off class also requires read all the data
        :return:
        '''
        pass

if __name__ == '__main__':
    #test_dicom_folder   = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/test'
    #test_dicom_folder = '/media/radonc/OS/Users/dlam/Data/raw_data_aapm/test offsite'
    # test_dicom_folder = '/media/radonc/OS/Users/dlam/Data/raw_data_aapm/test onsite'
    test_dicom_folder = '/media/radonc/OS/Users/dlam/Data/raw_data_aapm/validation 6'

    test_hdf5_folder    = ''

    # mask_hdf5_folder    = '/media/radonc/OS/Users/dlam/Projects/ContourSegmentation/Code/AutoSeg/aapm_journal_localtest/Small/Hdf5/test'
    mask_hdf5_folder = '/media/radonc/OS/Users/dlam/Data/aapm_journal/validation 6'

    generic_rs_file     = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/test/LCTSC-Train-S3-001/1.3.6.1.4.1.14519.5.2.1.7014.4598.158333268414958887774422778786/1.3.6.1.4.1.14519.5.2.1.7014.4598.225838432888111624925452649725/000000.dcm'


    subjects = [os.path.join( os.path.join (test_dicom_folder), name)
                for name in sorted( os.listdir( test_dicom_folder)) if os.path.isdir( os.path.join( test_dicom_folder, name))]

    for sub in subjects:
        print sub
        patient_name = os.path.basename( sub)

        try:
            f = h5py.File( os.path.join( mask_hdf5_folder, patient_name + '.hdf5.mask'), 'r' )
        except ValueError:
            print ValueError

        mask_prob = f['Masks_predict']
        mask = np.argmax(mask_prob, axis=1)

        gt_rs_file = ''
        for subdir, dirs, files in os.walk( sub):
            dcms = glob.glob(os.path.join(subdir, '*.dcm'))
            if len( dcms) > 1:
                slices = [dicom.read_file(dcm) for dcm in dcms]
                slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))

                image_file = dcms[0]
                #images = np.stack([s.pixel_array for s in slices], axis=0).astype(np.float32)
                #images = images + slices[0].RescaleIntercept

            if len( dcms) ==  1: # gt structure fiel
                gt_rs_file = dcms[0]

        mask2contour = MaskToContour( mask, slices)

        #need first slice, generic_rs, test_rs, otours, output
        rs_folder_name = os.path.split(image_file)
        rs_folder_name = os.path.split(rs_folder_name[0])[0]
        if not os.path.exists( os.path.join( rs_folder_name, 'rs')):
            os.makedirs( os.path.join( rs_folder_name, 'rs'))

        new_rs = RS_Create( image_file, generic_rs_file, gt_rs_file, mask2contour.Contours3D, rs_folder_name + '/rs/new_rs.dcm')


        #for submistion test
        # new_rs = RS_Create(image_file, generic_rs_file, gt_rs_file, mask2contour.Contours3D,
        #                    '../../aapm_journal_localtest/' + patient_name + '.dcm')