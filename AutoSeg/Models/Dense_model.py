import sys
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dense, Activation, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
#from metric import dice_coef, dice_coef_loss
from keras.layers.merge import concatenate, add

from AutoSeg.Dicom.image_constants import *

from keras.regularizers import l2

from AutoSeg.Models.metrics import *

from keras_contrib.applications.densenet import DenseNetFCN


IMG_ROWS = RE_SIZE
IMG_COLS = RE_SIZE



def densnet_func( optimizer):
    model = DenseNetFCN( (5, RE_SIZE, RE_SIZE),
                                nb_dense_block = 4, nb_layers_per_block= [ 3, 4, 5, 6, 6],
                                 upsampling_type= 'deconv', classes = N_CLASSES + 1)

    model.compile(optimizer=optimizer,
                  loss=cross_entropy_weighted_loss_by_samples,
                  metrics=[volume_accuracy])

    return model

get_unet = densnet_func