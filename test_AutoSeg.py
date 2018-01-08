#stat the slice thickness and spacing pixel in the dataset
import os
import pydicom as dicom
import glob

File_path = r'C:\Users\dlam\Data\raw_data_aapm\train'
i = 0
for subdir, dirs, files in os.walk( File_path):
    for subdir1, dirs1, files1 in os.walk( subdir):
        dcms = glob.glob( os.path.join( subdir, '*.dcm'))
        if len( dcms) >1:
            slice0 = dicom.read_file( dcms[0])
            i +=1
            print subdir
            print 'PixelSpacing: ',slice0.PixelSpacing[0]
            print 'SliceThickness:', slice0.SliceThickness