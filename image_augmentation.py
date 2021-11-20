# -*- coding: utf-8 -*-
"""
make project that use data augmentation to create changed images from given ones.
Use rotation, shift, zoom, brightness, flip.
Use Fashion MNIST data for this project.
The data contains 28x28 images of different clothing as the respective label.
There are 60000 images in the train and 10000 images in the test images 
"""

import numpy as np
from keras.utils import np_utils
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# matplotlib inline
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_test[i])
    plt.axis('off')
plt.show()
print('Labels: ', y_test[0:5])

    