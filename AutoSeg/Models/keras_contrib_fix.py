if K.image_data_format() == 'channels_first':
    x = Reshape((nb_classes, row, col))(x)
else:
    x = Reshape((row, col, nb_classes))(x)