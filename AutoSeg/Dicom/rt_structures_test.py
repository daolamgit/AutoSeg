import pydicom as dicom
import os
import copy
import numpy as np

from pydicom.sequence import Sequence
from pydicom.dataset import Dataset
from pydicom.multival import MultiValue

class RS():
    def __init__(self, rs_file):
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
        self.rs             = copy.deepcopy(  self.rs_generic) #to be modified, check if this is deep or shallow
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

                # ContourImageSequence = Sequence()
                # ContourImage = Dataset()
                # ContourImage.ReferencedSOPClassUID = self.rs_generic.structure.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPClassUID
                # ContourImage.ReferencedSOPInstanceUID = \
                # self.rs_generic.structure.ROIContourSequence[0].ContourSequence[0].ContourImageSequence[0].ReferencedSOPInstanceUID
                # ContourImageSequence.append( ContourImage)

                contour_slice.ContourImageSequence = Sequence()

                ContourSequence.append( contour_slice)

            ROIContour = Dataset()
            ROIContour.ContourSequence = ContourSequence
            ROIContour.ROIDisplayColor = self.rs_generic.structure.ROIContourSequence[i].ROIDisplayColor
            ROIContour.ReferencedROINumber = self.rs_generic.structure.ROIContourSequence[i].ReferencedROINumber
            ROIContourSequence.append( ROIContour)

            # contour = Dataset()
            # contour.ContourData = Sequence(self.contours[i]['contour'])
            # ContourSequence.append( [ContourData])

        self.rs.structure.ROIContourSequence = ROIContourSequence
        self.rs.structure.StructureSetROISequence = StructureSetROISequence

        return 0

        # It's better to go inside and change the structure
        # but not work either because of the difference in contour Sequence
        # for i in range( len( self.contours)):
        #     for k in range( len( self.contours[i]['contour'])):
        #         contour = MultiValue( dicom.valuerep.DSfloat, self.contours[i]['contour'][k])
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourData = contour

    def copy_from_rs(self):
        '''
        Just change the contour in the structure, not create a new structure
        :return:
        '''
        K = 20

        # for i in range(1, 5):
        #     self.rs.structure.ROIContourSequence[i].ContourSequence = self.rs.structure.ROIContourSequence[
        #                                                                   i].ContourSequence[K:K]


        #points = [[8, 9, -238],[8,9, -241],[8,9,-244]]
        #points = [[8, 9, -238], [8, 9, -241], [8, 9, -244]]
        # points = [[8.057, -194.784, -238], [8.545, -194.784, -238], [14.648, -195.272, -238]]
        #points = [[8, -194, -238], [8, -194, -241], [14, -195, -244]]
        #import numpy as np
        #points = 10 * np.random.random([K, 3] )
        # for i in [0,1,2,3,4]:
        #     for k in range( K):
        #         contour =  MultiValue( dicom.valuerep.DSfloat, np.array(self.contours[i]['contour'][k][:30])*1)
        #         #contour = MultiValue( dicom.valuerep.DSfloat, points[k])
        #         print contour
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourData = contour
        #     self.rs.structure.ROIContourSequence[i].ContourSequence = self.rs.structure.ROIContourSequence[i].ContourSequence[:K]

        # for i in [0,1,2,3,4]:
        #     for k in range( K):
        #         contour =  MultiValue( dicom.valuerep.DSfloat, np.array(self.contours[i]['contour'][k])*10)
        #         # contour = MultiValue( dicom.valuerep.DSfloat, points[k])
        #         print contour
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourData = contour
        #     #     self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourData = self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourData
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourImageSequence =  self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourImageSequence
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourGeometricType = self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourGeometricType
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].NumberOfContourPoints = self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].NumberOfContourPoints
        #     self.rs.structure.ROIContourSequence[i].ContourSequence = self.rs.structure.ROIContourSequence[i].ContourSequence[:K]

            # self.rs.structure.ROIContourSequence[i] = self.rs_test.structure.ROIContourSequence[i]

        # for i in range(len( self.contours)):
        #     for k in range(K): #range( len( self.contours[i]['contour'])):
        #         contour =  MultiValue( dicom.valuerep.DSfloat, np.array(self.contours[i]['contour'][k]))
        #         # contour = MultiValue( dicom.valuerep.DSfloat, points[k])
        #         #print contour
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourData = contour
        #     #     self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourData = self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourData
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourImageSequence =  Sequence() #self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourImageSequence
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].ContourGeometricType = 'CLOSED_PLANAR'#self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourGeometricType
        #         self.rs.structure.ROIContourSequence[i].ContourSequence[k].NumberOfContourPoints = len(contour)/3 #self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].NumberOfContourPoints
        #     self.rs.structure.ROIContourSequence[i].ContourSequence = self.rs.structure.ROIContourSequence[i].ContourSequence[:K]


        for i in range(len( self.contours)):
            self.rs.structure.ROIContourSequence[i].ContourSequence = Sequence()
            for k in range( len( self.contours[i]['contour'])):
                contour =  MultiValue( dicom.valuerep.DSfloat, np.array(self.contours[i]['contour'][k]))
                # contour = MultiValue( dicom.valuerep.DSfloat, points[k])
                #print contour
                ContourSequence = Dataset()
                ContourSequence.ContourData = contour
                ContourSequence.ContourImageSequence =  Sequence() #self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourImageSequence
                ContourSequence.ContourGeometricType = 'CLOSED_PLANAR'#self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].ContourGeometricType
                ContourSequence.NumberOfContourPoints = len(contour)/3 #self.rs_test.structure.ROIContourSequence[i].ContourSequence[k].NumberOfContourPoints
                self.rs.structure.ROIContourSequence[i].ContourSequence.append( ContourSequence)

if __name__ == '__main__':
    rs_file = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/LCTSC-Train-S1-001/1.3.6.1.4.1.14519.5.2.1.7014.4598.117430069853376188015337939664/1.3.6.1.4.1.14519.5.2.1.7014.4598.267594131248797648024467762948/000000.dcm'
    rs = RS( rs_file)

    rs_file1 = '/media/radonc/OS/Users/dlam/Data/aapm_journal_test/Small/Dicom/LCTSC-Train-S1-006/1.3.6.1.4.1.14519.5.2.1.7014.4598.248828253636339996946962229949/1.3.6.1.4.1.14519.5.2.1.7014.4598.161109536301962939847313975001/000000.dcm'
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