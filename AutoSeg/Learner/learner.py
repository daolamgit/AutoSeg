import os

from AutoSeg.Dicom.image_constants import *

import AutoSeg.Dicom.hdf5image as hdf5image

from learner_constants import *
import numpy as np

from keras.optimizers import Adam

from keras.models import load_model

from AutoSeg.Models.metrics import * #for load model

import h5py

class Unet25( object):
    def __init__(self, checkpoint_dir, log_dir, training_paths, testing_paths, roi, im_size, nclass,
                 model_func, train_model_name=None, test_model_name=None, batch_size = 1, testing_gt_available=True,
                 loss_type='cross_entroy', class_weights = None):
        '''
        Loading all information needed for Unet to work, mainly data input and net conf and train params
        :param checkpoint_dir:
        :param log_dir:
        :param training_paths:
        :param testing_paths:
        :param roi:
        :param im_size: its is 2D param because of the network is 2.5D
        :param nclass:
        :param model:
        :param batch_size:
        :param loss_type:
        :param class_weights:
        '''
        self.checking_dir       = checkpoint_dir
        self.log_dir            = log_dir
        self.training_paths     = training_paths
        self.testing_paths      = testing_paths
        self.nclass             = nclass
        self.roi                = roi
        self.im_size            = im_size
        self.testing_gt_available= testing_gt_available
        self.batch_size         = batch_size
        self.loss_type          = loss_type
        self.class_weights      = class_weights

        self.train_model_name   = train_model_name
        self.test_model_name    = test_model_name

        self.mean, self.std     = self.estimate_mean_std()



        self.best_train_score   = 0 #accuracy or loss min
        self.best_val_score     = 0

        #move model here
        optimizer = Adam()
        self.model = model_func(optimizer)

        #create checkpoint dir if needed
        if not os.path.exists( os.path.join( checkpoint_dir, roi[1])):
            os.makedirs( os.path.join( checkpoint_dir, roi[1]))

    def train(self):
        print "Training ...."

        counter = 0

        if self.train_model_name: #resume training from some previous training
            self.model = load_model( os.path.join( self.checking_dir, self.roi[1], self.train_model_name))

        print self.model.summary()

        for epoch in range(EPOCH):
            print "Epoch: ", epoch
            training_paths = np.random.permutation( self.training_paths)

            score_train = 0
            score_ent = 0
            for i in range( len( training_paths)):
                print "Training: ", training_paths[i]
                pt_train  = hdf5image.Patient_Train( training_paths[i], self.roi[0], self.im_size) #roi[0] notifies site to train, roi[1] string value
                score_ent1, score_train1    = self.one_patient_train( pt_train, FLAGS_train= True)
                score_train     += score_train1
                score_ent       += score_ent1
                print score_ent1, score_train1
                counter += 1

            score_train /= len(training_paths)
            score_ent   /= len(training_paths)
            print "Epoch score train is ", score_ent, score_train
            with open( os.path.join(self.log_dir, 'train.txt'), 'a') as file_train:
                file_train.write('%f, %f\n' % (score_ent, score_train))

            #####Valiation after each epoch#######
            if self.testing_gt_available and np.mod( epoch, N_EPOCH_VAL) == 0:
                score_val = 0
                score_ent = 0
                for j in range(len(self.testing_paths)):
                    print "Validating: ", self.testing_paths[j]
                    #testing_paths = np.random.permutation(( self.testing_paths))
                    pt_val = hdf5image.Patient_Train( self.testing_paths[j], self.roi[0], self.im_size)
                    score_ent1, score_val1 = self.one_patient_train( pt_val, FLAGS_train=False)
                    score_val  += score_val1
                    score_ent  += score_ent1

                    print score_ent1, score_val1

                score_val /= len( self.testing_paths)
                score_ent /= len( self.testing_paths)
                print "Epoch score validation is ", score_ent, score_val

                with open(os.path.join(self.log_dir, 'validation.txt'), 'a') as file_val:
                    file_val.write('%f, %f\n' % (score_ent, score_val))

                if score_val > self.best_val_score: #save the current best model
                    #update
                    self.best_val_score = score_val
                    model_name = "best_val_score_model.hdf5"
                    filepath = os.path.join(self.checking_dir, self.roi[1], model_name)
                    print "Saving best validation model", filepath
                    self.model.save( filepath, overwrite = True)



            #save checkpoint every N_EPOCH_CHECK
            if np.mod( epoch, N_EPOCH_CHECK)==0:
                #save model check point
                model_name = "{}_unet25_{}_train_acc _{}.hdf5".format( self.roi[1], epoch, score_train)
                print "Save check point: ", model_name
                filepath = os.path.join( self.checking_dir, self.roi[1], model_name)
                self.model.save(filepath, overwrite= True)



    def test(self):
        print "Testing ..."

        #model_name
        if self.test_model_name == None:
            self.test_model_name = "best_val_score_model.hdf5"

        #load model
        filepath = os.path.join( self.checking_dir, self.roi[1], self.test_model_name)
        self.model = load_model( filepath,
                    custom_objects={'cross_entropy_weighted_loss_by_samples': cross_entropy_weighted_loss_by_samples,
                                    'volume_accuracy': volume_accuracy})

        #test model
        for j in range( len( self.testing_paths)):
            print "Testing: ", self.testing_paths[j]
            pt_test = hdf5image.Patient_Test( self.testing_paths[j], self.roi[0], self.im_size)

            Masks_prob   = self.one_patient_predict( pt_test)

            #save Masks as hdf5
            filepath = self.testing_paths[j] + '.mask'
            hdf5 = h5py.File(filepath, 'w')
            hdf5.create_dataset('Masks_predict', data=Masks_prob)

            # contours = self.create_contours( Masks_prob)
            # rs_c    = RS_Create( pt_test.info, )


    def one_patient_predict(self, pt):
        Hyper_volume = pt.Hyper_volume
        Hyper_volume = (Hyper_volume - self.mean) / self.std

        Size        = Hyper_volume.shape
        # Masks       = np.zeros( (Size[0], self.nclass, Size[2], Size[3]))
        Masks       = np.zeros(( Size[0], Size[1], Size[2], self.nclass))
        print Masks.shape
        B = len(Hyper_volume) // BATCH_SIZE

        for b in range( B):
            volumes = Hyper_volume[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]
            Masks[b * BATCH_SIZE: (b + 1) * BATCH_SIZE] = self.model.predict_on_batch( volumes)

            # the remainder probably doesn;t matter but
        if np.mod(len(Hyper_volume), BATCH_SIZE):
            volumes = Hyper_volume[B * BATCH_SIZE:]
            Masks[B * BATCH_SIZE:] = self.model.predict_on_batch( volumes)

        return Masks

    def one_patient_train(self, pt, FLAGS_train=True):
        '''
        batch of BATCH_SIZE of slice for train and test, i.e. only one gradient update
        :param pt: patient
        :param FLAGS_train: True: train; False: Validation
        :return:
        '''
        Hyper_volume = pt.Hyper_volume  # this is augmentation volume
        Hyper_volume = (Hyper_volume - self.mean) / self.std
        #labels = pt.Masks_augmentation

        #####Train###########
        # get a batch of batch from Hyper_volume
        B = len(Hyper_volume) // BATCH_SIZE
        score_acc = 0 #one_patient acc_score
        score_ent = 0 #one_patient loss score

        for b in range( B):
            volumes     = Hyper_volume[b * BATCH_SIZE: (b + 1)*BATCH_SIZE]
            labels      = pt.Masks_augmentation[b * BATCH_SIZE: (b + 1) * BATCH_SIZE]

            if FLAGS_train:
                res = self.model.train_on_batch(volumes, labels)  # call sequentially, how to have weights?
            else:
                res = self.model.test_on_batch(volumes, labels)

            score_ent += res[0]
            score_acc += res[1] # used metric as score not loss
            #print "Batch:", b, "Res: ", res

        #the remainder probably doesn;t matter but
        if np.mod( len(Hyper_volume), BATCH_SIZE):
            volumes = Hyper_volume[B * BATCH_SIZE: ]
            labels = pt.Masks_augmentation[B * BATCH_SIZE:]
            if FLAGS_train:
                res = self.model.train_on_batch(volumes, labels)  # call sequentially
            else:
                res = self.model.test_on_batch(volumes, labels)
            #print "Batch:", B, "Res: ", res
            #score_acc += res[1]

        return (score_ent/B, score_acc /B)

    def estimate_mean_std(self):
        '''
        loading the dataset one by one and compute the mean std and put them into the list
        Get the mean of mean and std from those lists
        :return:
        '''
        means   = []
        stds     = []
        for i in range( len(self.training_paths)):
            pt =  hdf5image.Patient_Train( self.training_paths[i], self.roi[0], self.im_size)
            means.append( np.mean( pt.Volume_resize))
            stds.append( np.std( pt.Volume_resize))

            #a = pt.Volume_augmentation
        return np.mean( means), np.mean( stds)