import pydicom as dicom
import os
import copy
import numpy as np

from pydicom.sequence import Sequence
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue

class RS():
    '''
    Structure from a structure file
    '''
    def __init__(self, rs_file):
        if not rs_file:
            return #emtpy structure

        structure = dicom.read_file( rs_file)

        contours = []
        for i in range( len( structure.ROIContourSequence)):
            contour             = {}
            contour['name']     = structure.StructureSetROISequence[i].ROIName
            contour['number']   = structure.ROIContourSequence[i].ReferencedROINumber
            assert contour['number'] == structure.StructureSetROISequence[i].ROINumber
            contour['contour']          = [s.ContourData for s in structure.ROIContourSequence[i].ContourSequence]
            contours.append( contour)

        self.structure  = structure
        self.contours   = contours
        self.rs_file    = rs_file

        print "Current info: ", self.structure.data_element( 'PatientID')
        print "Current info: ", self.structure.data_element('PatientBirthDate')

    def print_rs(self):
        for i in range( len( self.structure.ROIContourSequence)):
            print self.structure.StructureSetROISequence[i].ROIName
            print self.structure.StructureSetROISequence[i].ROINumber


    def change_rs_basic(self):
        self.structure.data_element('PatientBirthDate').value = '01011900'

    def change_rs_complex(self, other_rs):

        #delete the current structure and save
        for i in range( len( self.structure.ROIContourSequence)):
            self.structure.StructureSetROISequence[i].ROIName = 'Item_' + str(i)


    def write_rs(self, file_name):
        dicom.dcmwrite( file_name, self.structure)


class RS_Create( ):
    '''
    Create RS file from a dicom, a generic rs, a test rs, a contour
    '''
    def __init__(self, image_file, generic_rs_file, test_rs_file, contours, rs_file_name = 'rs.dcm'):
        '''

        :param image_file: a dicom image to get the info from
        :param generic_rs_file: a rs template
        :param test_rs_file: a ground truth, don't even need, but nice to have for 3d/dice comparison
        :param contours: computed from deep learning segmenation
        :param rs_file_name: output rs
        '''
        self.info = dicom.dcmread( image_file)
        self.rs_generic     = RS( generic_rs_file) #or just dicom.dcmread(generic_rs_file)
        self.rs_test        = RS( test_rs_file)
        self.rs             = self.rs_generic # slow :copy.deepcopy(  self.rs_generic) #to be modified, check if this is deep or shallow
        self.contours       = contours
        self.rs.rs_file      = rs_file_name

        self.create_rs( )

    def create_rs(self ):
        '''
        contours: structure need to be stored in dicom
        :param dc:
        :param generic_rs:
        :param contours:
        :return:
        '''
        self.copy_info( )
        self.copy_contours( )
        # self.copy_from_rs()
        dicom.dcmwrite( self.rs.rs_file, self.rs.structure)



    def copy_info(self):
        '''
        Copy preamble from info to rs
        :param self:
        :return:
        '''
        self.rs.structure.SOPInstanceUID    = self.info.SOPInstanceUID
        self.rs.structure.StudyDescription  = self.info.StudyDescription
        self.rs.structure.PatientName       = self.info.PatientName
        self.rs.structure.PatientID         = self.info.PatientID
        self.rs.structure.BodyPartExamined  = self.info.BodyPartExamined
        self.rs.structure.StudyInstanceUID  = self.info.StudyInstanceUID
        self.rs.structure.SeriesInstanceUID = self.info.SeriesInstanceUID
        self.rs.structure.SeriesDescription = 'Whisperer'

    def copy_contours(self):
        '''
        copy contours to rs
        :param self:
        :return:
        '''
        ROIContourSequence  = Sequence()
        #ContourSequence     = Sequence()

        StructureSetROISequence = Sequence()
        for i in range( len( self.contours)):
            StructureSetROI = Dataset()
            StructureSetROI.ROINumber    = self.contours[i]['number']
            StructureSetROI.ROIName      = self.contours[i]['name']
            StructureSetROI.ROIGenerationAlgorithm = 'MANUAL'
            StructureSetROI.ReferencedFrameOfReferenceUID = ''
            StructureSetROISequence.append(StructureSetROI)

            ContourSequence = Sequence()
            for k in range( len( self.contours[i]['contour'])):
                contour_slice = Dataset()
                contour_slice.ContourData = MultiValue( dicom.valuerep.DSfloat, self.contours[i]['contour'][k])
                contour_slice.ContourGeometricType = 'CLOSED_PLANAR'
                contour_slice.NumberOfContourPoints = len( contour_slice.ContourData) /3 ########wrong

                contour_slice.ContourImageSequence = Sequence()

                ContourSequence.append( contour_slice)

            ROIContour = Dataset()
            ROIContour.ContourSequence = ContourSequence
            ROIContour.ROIDisplayColor = self.rs_generic.structure.ROIContourSequence[i].ROIDisplayColor
            ROIContour.ReferencedROINumber = self.rs_generic.structure.ROIContourSequence[i].ReferencedROINumber
            ROIContourSequence.append( ROIContour)


        self.rs.structure.ROIContourSequence = ROIContourSequence
        self.rs.structure.StructureSetROISequence = StructureSetROISequence

        return 0


if __name__ == '__main__':
    rs_file = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/LCTSC-Train-S1-001/1.3.6.1.4.1.14519.5.2.1.7014.4598.117430069853376188015337939664/1.3.6.1.4.1.14519.5.2.1.7014.4598.267594131248797648024467762948/000000.dcm'
    rs = RS( rs_file)

    #rs_file1 = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/LCTSC-Train-S1-006/1.3.6.1.4.1.14519.5.2.1.7014.4598.248828253636339996946962229949/1.3.6.1.4.1.14519.5.2.1.7014.4598.161109536301962939847313975001/000000.dcm'
    rs_file1 = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/LCTSC-Train-S3-001/1.3.6.1.4.1.14519.5.2.1.7014.4598.158333268414958887774422778786/1.3.6.1.4.1.14519.5.2.1.7014.4598.225838432888111624925452649725/000000.dcm'
    rs1 =RS( rs_file1)

    dc_file = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/LCTSC-Train-S1-001/1.3.6.1.4.1.14519.5.2.1.7014.4598.117430069853376188015337939664/1.3.6.1.4.1.14519.5.2.1.7014.4598.330486033168582850130357263530/000001.dcm'
    dc      = dicom.dcmread( dc_file)

    rs_folder_name = os.path.split( dc_file)
    rs_folder_name = os.path.split( rs_folder_name[0])[0]
    new_rs = RS_Create( dc_file, rs_file1, rs_file, rs.contours, rs_folder_name + '/rs/new_rs.dcm')

    # new_rs = RS_Create(dc_file, rs_file1, rs.contours, rs_folder_name + '/rs/new_rs.dcm')

    #assert new_rs and rs are the same by checking contours info and load to slicer

    # rs.change_rs_basic()
    # rs.change_rs_complex( rs1)
    #
    # newfile = 'new_rs.dcm'
    # rs.write_rs(newfile)
    #
    # rs_new  = RS(newfile)
    # rs_new.print_rs()