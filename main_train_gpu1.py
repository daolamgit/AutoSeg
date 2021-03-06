from AutoSeg.Learner.learner import *
import os
import pickle
import glob

from AutoSeg.Dicom.image_constants import *
import imp

from keras_contrib.applications.densenet import DenseNetFCN

import random

###some config lines should be pasted every time
import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)


#learner = Unet25('','','','','','','','')
FLAGS_train             = 0
# FLAGS_train_data_dir    = 'aapm_journal_localtest/Small/Hdf5-channel-last/train'
# FLAGS_test_data_dir     = 'aapm_journal_localtest/Small/Hdf5-channel-last/train'

# FLAGS_train_data_dir    = 'Data/Hdf5-channels-last/train'
# FLAGS_test_data_dir     = 'Data/Hdf5-channels-last/train'

# FLAGS_checkpoint_dir    = 'checkpoint5'
# FLAGS_log_dir           = 'logs5'

FLAGS_train_data_dir    = 'Data/Hdf5-SegThor-channels-last/train'
FLAGS_test_data_dir     = 'Data/Hdf5-SegThor-channels-last/test'
FLAGS_checkpoint_dir    = 'checkpoint_SegThor'
FLAGS_log_dir           = 'logs_SegThor'

def main():

    #setup training, testing list
    if FLAGS_train_data_dir == FLAGS_test_data_dir: #training with 2/3, validation with 1/3
        testing_gt_available = True
        if os.path.exists( os.path.join( FLAGS_train_data_dir, 'files.log')):
            with open( os.path.join( FLAGS_train_data_dir, 'files.log'), 'r') as f:
                training_paths, testing_paths = pickle.load(f)
        else:
            all_subjects    = sorted([ os.path.join( FLAGS_train_data_dir, name) for name in os.listdir( FLAGS_train_data_dir)])

            #when testing don't do hard things
            #np.random.seed(1)
            #all_subjects            = np.random.permutation( ( all_subjects))
            #all_subjects    = all_subjects[perm]

            n_training      = int( len(all_subjects) * 2 /3)
            training_paths  = all_subjects[:n_training]
            testing_paths   = all_subjects[n_training:]
            with open( os.path.join( FLAGS_train_data_dir, 'files.log'), 'w') as f:
                pickle.dump( [training_paths, testing_paths], f)

    else:
        if not FLAGS_train:
            training_paths  = sorted([ name for name in glob.glob( os.path.join( FLAGS_train_data_dir, '*.hdf5'))])
            testing_paths   = sorted([ name for name in glob.glob( os.path.join( FLAGS_test_data_dir, '*.hdf5'))])
        else: #train and test together, not recommend
            raise ValueError ('Not recommend for prototype because train and test together is at the submission only')

    if not os.path.exists( FLAGS_checkpoint_dir):
        os.makedirs(FLAGS_checkpoint_dir)

    if not os.path.exists( FLAGS_log_dir):
        os.makedirs( FLAGS_log_dir)

    # Unet25
    # model = imp.load_source('model', 'AutoSeg/Models/Unet25_model.py')
    # model_func = model.get_unet

    model = imp.load_source('model', 'AutoSeg/Models/Dense_model.py')
    model_func = model.get_unet

    learner_all = Unet25(checkpoint_dir= FLAGS_checkpoint_dir, log_dir = FLAGS_log_dir,
                         training_paths=training_paths, testing_paths = testing_paths,
                         roi = (-1, 'All'), im_size=RE_SIZE,
                         model_func=model_func,
                         train_model_name=None, test_model_name=None,
                         nclass=N_CLASSES + 1 )

    if FLAGS_train:
        learner_all.train()
    else:
        learner_all.test()

    # print ('Result is good')

if __name__ == '__main__':
    main()