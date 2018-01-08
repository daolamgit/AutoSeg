import dicom as dicom
import os, glob
import numpy as np
from skimage.draw import polygon
from skimage.transform import resize
import h5py

from image_constants import *

class Patient(object):

    def __init__(self,file_path):
        self.File_path                          = file_path
        self.Images_path, self.Structure_path   = self.find_images_contour_path()
        self.Slices, self.Volume                = self.create_volume()  # 3D tensor
        self.Volume_rescale                     = self.get_volume_rescale() #make every scan same physical size
        #self.z_scale, self.inplane_scale = self.get_scale_xyz()

        self.Volume_resize                      = self.get_volume_resize() #input to NN
        self.Hyper_volume                       = self.create_hyper_volume( self.Volume_resize)  # 4D tensor

        ###Scale and Preprocess Volumes


    def find_images_contour_path(self):
        '''
        :return: at least it returns both images and structures path
         so we don't need to overload in subclasses
        '''
        Images_path     = os.path.join(self.File_path, 'Images')
        Structure_path  = os.path.join(self.File_path, 'Structure')
        for subdir, dirs, files in os.walk(self.File_path):
            dcms = glob.glob(os.path.join(subdir,'*.dcm'))
            if len(dcms) > 1:
                Images_path     = subdir
            elif len(dcms) == 1:
                Structure_path  = subdir
        return (Images_path, Structure_path)

    @staticmethod #because we going to call this after augmentation instead
    #but I keep it here for the legacy

    def create_hyper_volume(Volume_resize):
        '''
        every HYPER_VOLUME_FACTOR become a volume
        :return:
        '''
        pad = int(HYPER_VOLUME_FACTOR / 2)
        array = np.pad( Volume_resize, ((pad, pad), (0, 0), (0, 0)), 'constant')
        [D, H, W] = array.shape

        Hyper_volume    = np.zeros( (D-HYPER_VOLUME_FACTOR+1, HYPER_VOLUME_FACTOR, H, W),dtype=np.float32)
        for i in range( D-HYPER_VOLUME_FACTOR + 1):
            Hyper_volume[i] = array[i : i + HYPER_VOLUME_FACTOR]
        return Hyper_volume

    def create_volume(self):
        '''Read all the images and sort them into list of slices and a 3D Volume
        '''
        dcms = glob.glob(os.path.join(self.Images_path,'*.dcm'))
        slices = [dicom.read_file(dcm) for dcm in dcms]
        slices.sort( key = lambda x: float(x.ImagePositionPatient[2]))
        images = np.stack( [s.pixel_array* slices[0].RescaleSlope+ slices[0].RescaleIntercept for s in slices], axis=0).astype(np.float32)
        images = self.normalize( images)
        return (slices, images)

    def get_volume_rescale(self):
        '''
        Rescale the Volume into the same physical size. In particular with the size in the constants
        :return:
        '''
        if self.inplane_scale['size'] == INPLANE_SIZE and self.zplane_scale['scale'] == 1:
            return self.Volume
        else:
            #Volume      = self.Volume;
        #if self.inplane_scale['size'] != INPLANE_SIZE or self.zplane_scale['scale'] != 1:
            Volume = resize( self.Volume,
                        (self.zplane_scale['size'], self.inplane_scale['size'], self.inplane_scale['size']),
                        mode='constant')
            if self.inplane_scale['size'] > INPLANE_SIZE:
                return self.crop( Volume, INPLANE_SIZE)
            else:
                return self.pad( Volume, INPLANE_SIZE)
            #return self.crop_pad( Volume)

    def get_volume_resize(self):
        Volume  = resize( self.Volume_rescale, (self.zplane_scale['size'], RE_SIZE + 2 * CROP, RE_SIZE + 2* CROP))
        Volume  = self.crop( Volume, RE_SIZE)
        return Volume

    def crop(self, array, new_size):
        '''
        Crop a Volume or Mask in xy axis to fit to new image size
        :param array: 3D array, new_size is just for inplane size
        :return:
        '''
        #crop = int((self.inplane_scale['size'] - new_size) / 2)
        crop    = int(( array.shape[2] - new_size) / 2)
        array = array[:, crop : -crop , crop : -crop]
        return array

    def pad(self, array, new_size):
        #pad = int((new_size - self.inplane_scale['size']) / 2)
        pad = int(( new_size - array.shape[2]) / 2)
        array = np.pad(array, ((0, 0), (pad, pad), (pad, pad)),'constant')
        return array

    # def get_scale_xyz(self):
    #
    #
    #     return (inplane_scale, z_scale)

    @property
    def inplane_scale(self):
        inplane = {}
        slice0          = self.Slices[0]
        inplane['scale']   = slice0.PixelSpacing[0] / PIXEL_SPACING
        inplane['size']    = int( np.rint( inplane['scale']  * slice0.Rows /2) *2) #so that its an even number
        return inplane

    @property
    def zplane_scale(self):
        zplane                  = {}
        slice0                  = self.Slices[0]
        zplane['scale']         = slice0.SliceThickness / SLICE_THICKNESS
        zplane['size']          = int( np.rint( zplane['scale'] * self.Volume.shape[0]))
        return zplane

    @staticmethod
    def normalize(im_input):
        im_output   = im_input + 1000
        im_output[im_output < 0] = 0
        im_output[im_output > 1600] = 1600
        im_output = im_output / 1600.0
        return im_output


class Patient_Train(Patient):
    def __init__(self, file_path):
        '''
        Some properties assigment can be replaced by property methods if  they are not sequential called
        :param file_path:
        '''
        super(Patient_Train, self).__init__(file_path)
        self.Contours = self.get_contours()
        self.Masks = self.create_masks()  # 3D tensor
        self.Masks_rescale = self.get_masks_rescale()
        self.Masks_resize   = self.get_masks_resize()

    def get_masks_resize(self):
        size = (self.zplane_scale['size'], RE_SIZE + 2 * CROP, RE_SIZE + 2 * CROP)
        Masks = np.zeros( size, dtype=np.float32)
        for z in range(N_CLASSES):
            roi = resize( (self.Masks_rescale == z + 1).astype(np.float32), size, mode='constant')
            Masks[ roi >= .5] = z + 1
        Masks   = self.crop( Masks, RE_SIZE)
        return Masks

    def get_masks_rescale(self):
        '''
        Rescale the masks of train data into the same physical size. In particular with the size in the constants
        :return:
        Although Mask was initialize with the same size of Volume_rescale so no need to crop or pad again
        '''
        #Masks      = self.Masks
        if self.inplane_scale['size'] == INPLANE_SIZE and self.zplane_scale['scale'] == 1:
            return self.Masks
        else:
        #Volume_rescale = self.Volume_rescale
        #Masks_rescale      = self.Masks
        #if self.inplane_scale['size'] != INPLANE_SIZE or self.zplane_scale['scale'] != 1:
            Masks_rescale      = np.zeros( (self.zplane_scale['size'], self.inplane_scale['size'], self.inplane_scale['size']),
                                           dtype=np.float32)
            for z in range( N_CLASSES):
                roi = resize((self.Masks == z+1).astype(np.float32),
                             (self.zplane_scale['size'], self.inplane_scale['size'], self.inplane_scale['size']),
                             mode='constant')
                Masks_rescale[ roi >= .5] = z + 1

            if self.inplane_scale['size'] > INPLANE_SIZE:
                return self.crop( Masks_rescale, INPLANE_SIZE)
            else:
                return self.pad( Masks_rescale, INPLANE_SIZE)


    def create_masks(self):
        '''
        input: array of 3D points in self.Contours
        :return: array of the size of Volume shape and labeled in organ indices
        '''
        masks = np.zeros( self.Volume.shape, dtype=np.float32)
        z           = [np.around( s.ImagePositionPatient[2], 1) for s in self.Slices]
        pos_r       = self.Slices[0].ImagePositionPatient[1]
        spacing_r   = self.Slices[0].PixelSpacing[1]
        pos_c       = self.Slices[0].ImagePositionPatient[0]
        spacing_c   = self.Slices[0].PixelSpacing[0]

        for con in self.Contours:
            num = ROI_ORDER.index( con['name']) +1
            for c in con['contour']:
                nodes = np.array(c).reshape((-1, 3)) #xyz
                assert np.amax( np.abs( np.diff( nodes[:, 2]))) == 0 #all z the same
                z_index     = z.index( np.around( nodes[0, 2], 1))

                r = (nodes[:, 1] - pos_r)/spacing_r
                c = (nodes[:, 0] - pos_c)/spacing_c
                rr, cc = polygon(r ,c)
                masks[ z_index, rr, cc] = num
        return masks

    def read_structure(self, structure):
        contours = []
        for i in range(len(structure.ROIContourSequence)):
            contour = {}
            contour['name']     = structure.StructureSetROISequence[i].ROIName
            contour['number'] = structure.ROIContourSequence[i].ReferencedROINumber
            assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
            contour['contour']  = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            contours.append( contour)
        return contours

    def get_contours(self):
        dcm = glob.glob(os.path.join(self.Structure_path, '*.dcm'))
        structure = dicom.read_file(dcm[0])
        contours = self.read_structure(structure)
        return contours

class Patient_Test(Patient):
    def __init__(self, file_path):
        super(Patient_Test,self).__init__(file_path)

        #propery not good to put them in constructor because it needs NN
        self.Mask = self.compute_masks() #from test process in NN
        self.Contours = self.get_contours()

    def compute_masks(self):
        pass

    def get_contours(self):
        pass



class Slice(object):
    """"Hold in formation about 1 slice
    """
    def __init__(self):
        pass

class Structures(object):

    def __init__(self, file_path):
        #pass
        self.file_path = file_path
        self.structures = dicom.read_file(file_path)



if __name__ == '__main__':
    #st = Structures(r'C:\Users\dlam\Data\raw_data_aapm\train\LCTSC-Train-S1-001\1.3.6.1.4.1.14519.5.2.1.7014.4598.117430069853376188015337939664\1.3.6.1.4.1.14519.5.2.1.7014.4598.267594131248797648024467762948\000000.dcm')

    #pt = Patient_Train(r'C:\Users\dlam\Data\raw_data_aapm\train\LCTSC-Train-S1-001\1.3.6.1.4.1.14519.5.2.1.7014.4598.117430069853376188015337939664')
    #pt = Patient_Train(r'C:\Users\dlam\Data\raw_data_aapm\train\LCTSC-Train-S3-005\1.3.6.1.4.1.14519.5.2.1.7014.4598.564375304681767491558739261597')
    # pt = Patient_Train(r'C:\Users\dlam\Data\raw_data_aapm\train\LCTSC-Train-S3-005')
    # #pt = Patient_Test(r'C:\Users\dlam\Data\raw_data_aapm\test offsite\LCTSC-Test-S3-102')
    # print pt.Images_path, pt.Structure_path
    # #a bunch of shit test
    # assert pt.Volume.shape == (188, 512, 512)
    # assert pt.Volume_rescale.shape == (125, 512, 512) #after rescale and crop
    # assert pt.Masks.shape    == (188, 512, 512)
    # assert pt.Masks_rescale.shape == (125, 512, 512)  # after rescale and crop
    #
    # output_path = r'C:\Users\dlam\Data\TumorContouring\aapm_h5'
    # if not os.path.exists( output_path):
    #     os.makedirs(output_path)
    #
    # hdf5_file = h5py.File(os.path.join( output_path, 'test.hdf5'))
    # hdf5_file.create_dataset( 'Hyper_volume', data=pt.Hyper_volume)
    # hdf5_file.create_dataset( 'Masks_resize', data=pt.Masks_resize)
    # hdf5_file.close()

    # #for display
    # import matplotlib.pyplot as plt
    # for i in range(80, 110):
    #     plt.figure(1)
    #     plt.imshow( pt.Volume_rescale[i])
    #     plt.pause(2)
    #     plt.figure(2)
    #     plt.imshow( pt.Masks_rescale[i])
    #     plt.pause(2)

    # input_path_train    = r'C:\Users\dlam\Data\raw_data_aapm\train'
    # input_path_test     = r'C:\Users\dlam\Data\raw_data_aapm\test offsite'
    # output_path_train   = r'C:\Users\dlam\Data\aapm_journal\train'
    # output_path_test    = r'C:\Users\dlam\Data\aapm_journal\test'

    input_path_train    = r'/media/radonc/OS/Users/dlam/Data/aapm_journal_shittest/Small/Dicom'
    input_path_test     = r'C:\Users\dlam\Data\raw_data_aapm\test offsite'
    output_path_train   = r'/media/radonc/OS/Users/dlam/Data/aapm_journal_shittest/Small/Hdf5/train'
    output_path_test    = r'C:\Users\dlam\Data\aapm_journal\test'

    FLAGS_train         = 1
    FLAGS_test          = 0

    if not os.path.exists( output_path_train):
        os.makedirs( output_path_train)
    if not os.path.exists( output_path_test):
        os.makedirs( output_path_test)

    #train
    if FLAGS_train:
        subjects = [ os.path.join( input_path_train, name)
                     for name in sorted( os.listdir( input_path_train)) if os.path.join(input_path_train, name)]
        for sub in subjects:
            name = os.path.basename( sub)
            print name
            pt = Patient_Train( sub)
            hdf5 = h5py.File( os.path.join( output_path_train, name + '.hdf5'), 'w')
            hdf5.create_dataset( 'Hyper_volume', data=pt.Hyper_volume)
            hdf5.create_dataset( 'Volume_resize', data=pt.Volume_resize)
            hdf5.create_dataset('Masks_resize', data=pt.Masks_resize)

    #test
    if FLAGS_test:
        subjects = [ os.path.join( input_path_test, name)
                     for name in sorted( os.listdir( input_path_test)) if os.path.join(input_path_test, name)]
        for sub in subjects:
            name = os.path.basename( sub)
            print name
            pt = Patient_Test( sub)
            hdf5 = h5py.File( os.path.join( output_path_test, name + '.hdf5'), 'w')
            hdf5.create_dataset( 'Hyper_volume', data=pt.Hyper_volume)
            hdf5.create_dataset('Volume_resize', data=pt.Volume_resize)
            #hdf5.create_dataset('Masks_resize', data=pt.Masks_resize)