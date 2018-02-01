import keras.backend as K
import numpy as np


uu = np.array([[[[1, 0], [0, 1]],
                [[1, 0], [0, 1]],
                [[3, 2], [0, 1]]],

               [[[1, 0], [0, 1]],
                [[2, 0], [0, 1]],
                [[2, 0], [0, 1]]]])

a = K.variable( uu) # 2 3 2 2

ll = np.array( [ 2, 1]).transpose()
b = K.variable( ll)

c = K.dot( a, b)

print c.eval()

d = a * a
print d.eval()
