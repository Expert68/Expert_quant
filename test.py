from __future__ import  print_function
import numpy as np
np.random.seed(1337)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
np_epoch = 10
img_size = 28*28
(x_train,y_train),(x_test,y_test) = mnist.load_data()
x_train = x_train.reshape(y_train.shape[0],img_size).astype('float32')/255
x_test = x_test.reshape(y_test.shape[0],img_size).astype('float32')/255

print(x_train.shape,x_test.shape)



