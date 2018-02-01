import sys
from keras.models import Model
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
#from metric import dice_coef, dice_coef_loss
from keras.layers.merge import concatenate, add

from AutoSeg.Dicom.image_constants import *

from AutoSeg.Models.metrics import *


IMG_ROWS = RE_SIZE
IMG_COLS = RE_SIZE


def _shortcut(_input, residual):
    stride_width = _input._keras_shape[2] / residual._keras_shape[2]
    stride_height = _input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == _input._keras_shape[1]

    shortcut = _input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                          strides=(stride_width, stride_height),
                          kernel_initializer="he_normal", padding="valid")(_input)

    return add([shortcut, residual])


def inception_block(inputs, depth, batch_mode=0, splitted=False, activation='relu'):
    assert depth % 16 == 0
    actv = activation == 'relu' and (lambda: LeakyReLU(0.0)) or activation == 'elu' and (lambda: ELU(1.0)) or None

    c1_1 = Conv2D(depth / 4, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)

    c2_1 = Conv2D(depth / 8 * 3, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    c2_1 = actv()(c2_1)
    if splitted:
        c2_2 = Conv2D(depth / 2, (1, 3), kernel_initializer='he_normal', padding='same')(c2_1)
        c2_2 = BatchNormalization(axis=1)(c2_2)
        c2_2 = actv()(c2_2)
        c2_3 = Conv2D(depth / 2, (3, 1), kernel_initializer='he_normal', padding='same')(c2_2)
    else:
        c2_3 = Conv2D(depth / 2, (3, 3), kernel_initializer='he_normal', padding='same')(c2_1)

    c3_1 = Conv2D(depth / 16, (1, 1), kernel_initializer='he_normal', padding='same')(inputs)
    # missed batch norm
    c3_1 = actv()(c3_1)
    if splitted:
        c3_2 = Conv2D(depth / 8, (1, 5), kernel_initializer='he_normal', padding='same')(c3_1)
        c3_2 = BatchNormalization(axis=1)(c3_2)
        c3_2 = actv()(c3_2)
        c3_3 = Conv2D(depth / 8, (5, 1), kernel_initializer='he_normal', padding='same')(c3_2)
    else:
        c3_3 = Conv2D(depth / 8, (5, 5), kernel_initializer='he_normal', padding='same')(c3_1)

    p4_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    c4_2 = Conv2D(depth / 8, (1, 1), kernel_initializer='he_normal', padding='same')(p4_1)

    res = concatenate([c1_1, c2_3, c3_3, c4_2], axis=1)
    res = BatchNormalization(axis=1)(res)
    res = actv()(res)
    return res


def rblock(inputs, num, depth, scale=0.1):
    residual = Conv2D(depth, (num, num), padding='same')(inputs)
    residual = BatchNormalization(axis=1)(residual)
    residual = Lambda(lambda x: x * scale)(residual)
    res = _shortcut(inputs, residual)
    return ELU()(res)


def NConv2D(nb_filter, nb_row, nb_col, padding='same', strides=(1, 1)):
    def f(_input):
        conv = Conv2D(filters=nb_filter, kernel_size=(nb_row, nb_col), strides=strides,
                      padding=padding)(_input)
        norm = BatchNormalization(axis=1)(conv)
        return ELU()(norm)

    return f


def BNA(_input):
    inputs_norm = BatchNormalization(axis=1)(_input)
    return ELU()(inputs_norm)



def get_unet_inception_mod(optimizer):
    splitted = True
    act = 'elu'

    inputs = Input((HYPER_VOLUME_FACTOR, IMG_ROWS, IMG_COLS), name='main_input')
    conv1 = inception_block(inputs, 64, batch_mode=2, splitted=splitted, activation=act)
    # conv1 = inception_block(conv1, 32, batch_mode=2, splitted=splitted, activation=act)

    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = NConv2D(64, 3, 3, padding='same', strides=(2, 2))(conv1)
    pool1 = Dropout(0.5)(pool1)

    conv2 = inception_block(pool1, 128, batch_mode=2, splitted=splitted, activation=act)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = NConv2D(128, 3, 3, padding='same', strides=(2, 2))(conv2)
    pool2 = Dropout(0.5)(pool2)

    conv3 = inception_block(pool2, 256, batch_mode=2, splitted=splitted, activation=act)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = NConv2D(256, 3, 3, padding='same', strides=(2, 2))(conv3)
    pool3 = Dropout(0.5)(pool3)

    conv4 = inception_block(pool3, 512, batch_mode=2, splitted=splitted, activation=act)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = NConv2D(512, 3, 3, padding='same', strides=(2, 2))(conv4)
    pool4 = Dropout(0.5)(pool4)

    conv5 = inception_block(pool4, 512, batch_mode=2, splitted=splitted, activation=act)
    # conv5 = inception_block(conv5, 512, batch_mode=2, splitted=splitted, activation=act)
    conv5 = Dropout(0.5)(conv5)

    ####Auxiliarry isn't neccessary
    #pre = Conv2D(1, (1, 1), kernel_initializer='he_normal', activation='sigmoid')(conv5)
    #pre = Flatten()(pre)
    #aux_out = Dense(1, activation='sigmoid', name='aux_output')(pre)
    #

    after_conv4 = rblock(conv4, 1, 512)
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), after_conv4], axis=1)
    conv6 = inception_block(up6, 512, batch_mode=2, splitted=splitted, activation=act)
    conv6 = Dropout(0.5)(conv6)

    after_conv3 = rblock(conv3, 1, 256)
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), after_conv3], axis=1)
    conv7 = inception_block(up7, 256, batch_mode=2, splitted=splitted, activation=act)
    conv7 = Dropout(0.5)(conv7)

    after_conv2 = rblock(conv2, 1, 128)
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), after_conv2], axis=1)
    conv8 = inception_block(up8, 128, batch_mode=2, splitted=splitted, activation=act)
    conv8 = Dropout(0.5)(conv8)

    after_conv1 = rblock(conv1, 1, 64)
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), after_conv1], axis=1)
    conv9 = inception_block(up9, 64, batch_mode=2, splitted=splitted, activation=act)
    # conv9 = inception_block(conv9, 32, batch_mode=2, splitted=splitted, activation=act)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Conv2D(N_CLASSES + 1, (1, 1), kernel_initializer='he_normal', activation='sigmoid', name='main_output')(conv9)
    # print conv10._keras_shape

    model = Model(inputs=inputs, outputs=[conv10])
    model.compile(optimizer=optimizer,
                  loss={'main_output': cross_entropy_weighted_loss_by_samples},
                  metrics={'main_output': volume_accuracy},
                  loss_weights={'main_output': 1.})

    return model


get_unet = get_unet_inception_mod


def main():
    from keras.optimizers import Adam, RMSprop, SGD
    import numpy as np
    img_rows = IMG_ROWS
    img_cols = IMG_COLS

    optimizer = RMSprop(lr=0.045, rho=0.9, epsilon=1.0)
    model = get_unet(Adam(lr=1e-5))
    model.compile(optimizer=optimizer,
                  loss={'main_output': cross_entropy_weighted_loss},
                  metrics={'main_output': volume_accuracy},
                  loss_weights={'main_output': 1.})

    x = np.random.random((1, HYPER_VOLUME_FACTOR, img_rows, img_cols))
    res = model.predict(x, 1)
    print res
    # print 'res', res[0].shape
    print 'params', model.count_params()
    print 'layer num', len(model.layers)
    print model.summary()
    #


if __name__ == '__main__':
    sys.exit(main())

