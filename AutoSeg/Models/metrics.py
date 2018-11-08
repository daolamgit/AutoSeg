from keras import backend as K
import numpy as np


# LOSS_WEIGHTS    = (1.0, 2.0, 3.0)
# N_CLASSES = 2
# RE_SIZE = 2

# from ..Dicom.image_constants import *
# from ..Learner.learner_constants import *

from AutoSeg.Dicom.image_constants import *
from AutoSeg.Learner.learner_constants import *

def loss_weights( weights):
    '''
    A nested customed loss function to overcome the drawback of hardcoded loss function in Keras
    Can be used to pass weight during model compile
    :param weights:
    :return:
    '''
    def loss( y_true, y_pred):
        return 1
    return loss

def cross_entropy_densenet_fcn( y_true, y_pred):
    '''
    Due to volume output of denset, normal keras won't work, it works if swap channel
    However a custom never works so far
    :param y_true:
    :param y_pred:
    :return:
    '''

    # Kt = K.flatten(y_true)
    # Kp = K.flatten(y_pred)
    #
    # Kt = K.reshape(y_true, (1, -1))
    # Kp = K.reshape( y_pred, (1, -1))
    # return K.binary_crossentropy( Kt, Kp, from_logits=False)
    #return K.max(Kt)
    # return K.maximum( 0., 1.)

    # return K.mean(K.binary_crossentropy( Kt, Kp), axis = -1)

    # return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=True))

def cross_entropy_weighted_loss_by_labels( y_true, y_pred):
    '''
    This function is a customed loss function with a parameter loss is outside of Keras scope.
    The solution is passing a Global constants Weights and access them without having to think about how  to pass it.

    This implementation return only 1 loss for the whole batch
    i.e. somewhat regularize the error over the batch
    Also: Its cross entropy over the pixel distribution rather than prediction distribution
    :param y_true:
    :param y_pred:
    :return:
    '''
    Kt = K.permute_dimensions(y_true, (1, 0, 2, 3)) #swap class to front
    Kp = K.permute_dimensions(y_pred, (1, 0, 2, 3))
    Kt  = K.reshape( Kt, (N_CLASSES +1, -1))
    Kp  = K.reshape( Kp, (N_CLASSES +1, -1))

    #print N_CLASSES + 1
    a = K.categorical_crossentropy(Kt, Kp, from_logits='True')
    #print a.eval()
    class_weight = K.constant( np.asarray( LOSS_WEIGHTS, dtype=np.float32))
    weighted_loss = K.dot( class_weight, a)
    return weighted_loss

def cross_entropy_weighted_loss_by_samples( y_true, y_pred):
    '''
    This function is a customed loss function with a parameter loss is outside of Keras scope.
    The solution is passing a Global constants Weights and access them without having to think about how  to pass it.

    This function return batch of loss, ie. works as a normal mini batch
    :param y_true: 4D tensor B x C x H x W
    :param y_pred:
    :return: a tensor of Batch, sum of all loss in 1 image
    '''
    # Kt = K.permute_dimensions(y_true, (0, 3, 2, 1)) #swap class to end

    # Kp = K.permute_dimensions(y_pred, (0, 3, 2, 1))
    Kt  = K.reshape( y_true, (-1, N_CLASSES +1))

    #print "Kt reshape", Kt.eval()

    Kp  = K.reshape( y_pred, (-1, N_CLASSES +1))

    #print N_CLASSES + 1
    a = K.categorical_crossentropy(Kt, Kp, from_logits= True)

    #print "Cross entropy", a.eval()
    class_weight = K.constant( np.asarray( LOSS_WEIGHTS, dtype=np.float32).reshape(-1,1))
    weight_map      = K.dot( Kp, class_weight)
    #
    # ll = K.eval( weight_map)
    # print ll.shape
    weight_map = K.squeeze( weight_map, axis=-1)
    map_loss =  a * weight_map

    #print "map_loss", map_loss.eval()

    # weighted_loss = K.sum( K.reshape( map_loss, (-1, RE_SIZE * RE_SIZE)), axis = -1 )
    weighted_loss = K.sum( map_loss)
    # a = K.shape(weighted_loss)
    #
    # print "Weighted shape: ", a.eval()
    # print "Weighted ", weighted_loss.eval()

    return weighted_loss

def cross_entropy_weighted_loss_by_samples_channel_first( y_true, y_pred):
    '''
    This function is a customed loss function with a parameter loss is outside of Keras scope.
    The solution is passing a Global constants Weights and access them without having to think about how  to pass it.

    This function return batch of loss, ie. works as a normal mini batch
    :param y_true: 4D tensor B x C x H x W
    :param y_pred:
    :return: a tensor of Batch, sum of all loss in 1 image
    '''
    Kt = K.permute_dimensions(y_true, (0, 3, 2, 1)) #swap class to end

    Kp = K.permute_dimensions(y_pred, (0, 3, 2, 1))
    Kt  = K.reshape( Kt, (-1, N_CLASSES +1))

    #print "Kt reshape", Kt.eval()

    Kp  = K.reshape( Kp, (-1, N_CLASSES +1))

    #print N_CLASSES + 1
    a = K.categorical_crossentropy(Kt, Kp, from_logits='True')

    #print "Cross entropy", a.eval()
    class_weight = K.constant( np.asarray( LOSS_WEIGHTS, dtype=np.float32).transpose())
    weight_map      = K.dot( Kt, class_weight)

    map_loss =  a * weight_map
    #print "map_loss", map_loss.eval()

    weighted_loss = K.sum( K.reshape( map_loss, (-1, RE_SIZE * RE_SIZE)), axis = -1 )
    #a = K.shape(weighted_loss)

    #print "Weighted shape: ", a.eval()
    #print "Weighted ", weighted_loss.eval()

    return weighted_loss


def volume_accuracy(y_true, y_pred):
    return K.mean( K.equal( K.argmax( y_true, axis = -1), K.argmax( y_pred, axis = -1)))


# def volume_accuracy(y_true, y_pred):
#     '''only labels > 0'''
#     indices = K.argmax(y_true)> N_CLASSES-2 #only the last 2
#     # y_t = y_true[indices]
#     # y_p = y_pred[indices]_
#     return K.mean( K.equal( K.argmax( y_true[indices], axis= 1), K.argmax( y_pred[indices], axis = 1)))

def dice_coef(y_true, y_pred):
    smooth = .1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

if __name__ == '__main__':
    uu = np.array( [ [[[1, 0],[0 ,1]],
                        [[1, 0],[0 ,1]],
                        [[3, 2],[0 ,1]]],

                        [[[1, 0],[0 ,1]],
                        [[2, 0],[0 ,1]],
                        [[2, 0],[0 ,1]]]])


    kp = K.variable( uu)

    kt = K.variable( [ [[[1, 0],[0 ,1]],
                        [[0, 0],[0 ,1]],
                        [[0, 2],[0 ,1]]],

                        [[[0, 0],[0 ,1]],
                        [[1, 0],[0 ,1]],
                        [[0, 0],[0 ,1]]]])

    a = cross_entropy_weighted_loss_by_labels( kt, kp)
    print "label cross entropy", a.eval()

    b = cross_entropy_weighted_loss_by_samples( kt, kp)
    print "samples cross entropy", b.eval()

    b = volume_accuracy( kp, kt)
    print 'Accuracy', b.eval()
