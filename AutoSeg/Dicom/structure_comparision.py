import pydicom as dicom
import os, glob
import numpy as np

from image_constants import *
from skimage.draw import polygon

#from scipy.spatial.distance import directed_hausdorff,dice

from distance_metrics import Surface

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Comparison( ):
    def __init__(self, dicom_folder, gt_rs_file, ai_rs_file):
        '''
        dicom_path
        ai_rs_file
        gt_rs_file
        '''
        self.dicom_folder   = dicom_folder
        self.gt_rs_file     = gt_rs_file
        self.ai_rs_file     = ai_rs_file

        self.Slices = self.read_dicom()
        self.Volume, self.Voxel = self.init_volume()

        self.Contours_ai    = self.get_contours( ai_rs_file)
        self.Contours_gt    = self.get_contours( gt_rs_file)

        #self.Mask_gt, self.Mask_ai     = self.create_masks()
        self.Mask_gt        = self.create_masks( self.Contours_gt)
        self.Mask_ai        = self.create_masks( self.Contours_ai)

        res = self.comparison()

    def create_masks(self, Contours):
        '''
        contour_gt and ai may have different size so we need to call this 2 times
        Can't do it in 1 shot. This is even neater
        :param Contours:
        :return:
        '''
        Mask = np.zeros_like( self.Volume, dtype='float32')

        z           = [np.around( s.ImagePositionPatient[2], 1) for s in self.Slices]
        pos_r       = self.Slices[0].ImagePositionPatient[1]
        spacing_r   = self.Slices[0].PixelSpacing[1]
        pos_c       = self.Slices[0].ImagePositionPatient[0]
        spacing_c   = self.Slices[0].PixelSpacing[0]

        for con in Contours:
            num = ROI_ORDER.index( con['name']) + 1

            for c in con['contour']:
                nodes = np.array( c).reshape( (-1, 3))
                assert np.amax( np.abs( np.diff( nodes[:, 2]))) == 0
                z_index = z.index( np.around( nodes[0, 2], 1))

                r = (nodes[:, 1] - pos_r)/spacing_r
                c = (nodes[:, 0] - pos_c)/spacing_c
                rr, cc = polygon( r, c)
                Mask[ z_index, rr, cc] = num

        return Mask

    def read_dicom(self):
        '''
        Can't avoid reading images
        :return:
        '''
        dcms    = glob.glob( os.path.join( self.dicom_folder, '*.dcm'))
        slices = [dicom.read_file( dcm) for dcm in dcms]
        slices.sort( key = lambda x: float( x.ImagePositionPatient[2]))

        return slices

    def init_volume(self):
        '''
        Init a zero fill volume of no dicoms and H X W
        Init voxel size
        :return:
        '''
        dcms = glob.glob(os.path.join(self.dicom_folder, '*.dcm'))

        slice0 = dicom.dcmread( dcms[0])
        image  = slice0.pixel_array
        (H, W) = image.shape

        vx      = slice0.PixelSpacing[0]
        vy      = slice0.PixelSpacing[1]
        try:
            vz      = slice0.SliceThickness
        except:

            vz      = abs( self.Slices[0].ImagePositionPatient[2] -
                           self.Slices[1].ImagePositionPatient[2])

        volume = np.zeros( (len(dcms), H, W)).astype('float32')

        voxel  = (vz, vy, vx) #because depth row col

        return (volume, voxel)

    def get_contours(self, rs_file):
        '''
        Get contour from file
        :param rs_file:
        :return:
        '''

        structure = dicom.dcmread( rs_file)
        contours = []
        for i in range(len(structure.ROIContourSequence)):
            contour = {}
            contour['name']     = structure.StructureSetROISequence[i].ROIName
            contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
            assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
            contour['contour']  = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            contours.append( contour)
        return contours


    def comparison(self):
        res = []
        for i in range( ( N_CLASSES)):
            mask_gt = self.Mask_gt == i + 1
            mask_ai = self.Mask_ai == i + 1

            dice_num        = self.dice_compute( mask_gt.flatten(), mask_ai.flatten())
            hausdorff_num   = self.hausdorff_compute( mask_gt, mask_ai) #hausdauff(mask_gt, mask_ai)
            print dice_num, hausdorff_num

            #self.plot_3D( mask_gt, mask_ai)

            res.append((dice_num, hausdorff_num))


        with open( 'scores.txt', 'a') as file_score:
            for item in res:
                file_score.write( '%f, %f \n' % (item[0], item[1]))

        return res # 5 contour


    def dice_compute(self, mask_gt, mask_ai):
        '''
       input as 1D
        :param mask_gt:
        :param mask_ai:
        :return:
        '''

        intersect = np.sum( mask_gt & mask_ai)

        try:
            dc = 2* intersect /float( np.sum(mask_gt) + np.sum(mask_ai))
        except ZeroDivisionError:
            dc = 0

        return dc

    def hausdorff_compute(self, mask_gt, mask_ai):
        '''the order of gt and ai the other way around from me'''
        sur = Surface( mask_ai, mask_gt, physical_voxel_spacing= self.Voxel)

        #sur = Surface(mask_ai, mask_gt)
        return sur.get_maximum_symmetric_surface_distance()

    def plot_3D(self, mask_gt, mask_ai):


        itemindex = np.where(mask_gt == 1)
        factor = len(itemindex[0]) // 1000
        if factor == 0:
            factor = 1

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        zs = itemindex[0][::factor]
        xs = itemindex[1][::factor]
        ys = itemindex[2][::factor]
        ax.scatter(xs, ys, zs)

        itemindex = np.where(mask_ai == 1)

        zs = itemindex[0][::factor]
        xs = itemindex[1][::factor]
        ys = itemindex[2][::factor]
        ax.scatter(xs, ys, zs)

        plt.show()



if __name__ == '__main__':
    test_dicom_folder = '/media/radonc/OS/Users/dlam/Data/raw_data_aapm/validation 6'

    subjects = [os.path.join(os.path.join(test_dicom_folder), name)
                for name in sorted(os.listdir(test_dicom_folder)) if
                os.path.isdir(os.path.join(test_dicom_folder, name))]

    for sub in subjects:
        print sub

        patient_name = os.path.basename(sub)

        for subdir, dirs, files in os.walk( sub):
            dcms = glob.glob( os.path.join( subdir, '*.dcm'))

            if len( dcms) > 1:
                dicom_folder = subdir

            if len( dcms) == 1:
                if os.path.basename(dcms[0]) == '000000.dcm':
                    gt_rs_file = dcms[0]
                elif os.path.basename(dcms[0]) == 'new_rs.dcm':
                    ai_rs_file = dcms[0]

        comp = Comparison( dicom_folder, gt_rs_file, ai_rs_file)
