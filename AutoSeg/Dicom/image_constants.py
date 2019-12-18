HYPER_VOLUME_FACTOR         = 5 #number of slices to make a 3D volume
LUNG                        = 'LUNG'
# LOSS_WEIGHTS    = (1.0, 20.0, 2.0, 2.0, 20.0, 40.0)
#ROI_ORDER                   = ['SpinalCord', 'Lung_R', 'Lung_L', 'Heart', 'Esophagus']

# SLICE_THICKNESS             = 3
PIXEL_SPACING               = 0.9765625
INPLANE_SIZE                = 512

RE_SIZE                     = 208 #48 #208 # Resize to 72, 256, 256 then crop to 208
CROP                        = 24 #8 #24


N_CLASSES                   = 5

CHANNELS_LAST                = 1

N_CLASSES                   = 4
ROI_ORDER                   = ['heart', 'aorta', 'trachea', 'esophagus']
SLICE_THICKNESS             = 2.5

LOSS_WEIGHTS    = (1.0, 2.0, 2.0, 2.0, 4.)

