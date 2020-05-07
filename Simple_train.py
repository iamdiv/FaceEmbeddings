import os
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt


from pylab import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.datasets import mnist

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform,he_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
from keras.utils import plot_model,normalize

from sklearn.metrics import roc_curve,roc_auc_score

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
## dataset
from keras.datasets import mnist
from keras.regularizers import l2
## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout,Conv2D,MaxPooling2D,Lambda
from keras.optimizers import Adam

from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

## for visualizing 
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

input_image_shape = (250,250,3)
batch_size = 256
epochs = 150
no_of_component = 2
embedding_size = 128


  
img_path = "D:/facenet/image"

all_imag = os.listdir(img_path)

img_arr = []
label = []
for i in all_imag:

    print("*************************************",img_path +"/"+i)
    img_p = os.listdir(img_path +"/"+i)
    print(img_p)
    for p in img_p:
        path = img_path + "/" + i +"/"+ p
        print(path)
        img_arr.append(cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),(250,250),interpolation=cv2.INTER_CUBIC))
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        label.append(i)

#df = pd.DataFrame(all_image,columns = ["img","label"])

x_train = img_arr
x_train = np.array(x_train)
print(x_train.shape)
label = pd.DataFrame(label)
le = LabelEncoder()
label = le.fit_transform(label)
y_train = label
print("checked")

#model = Model(inputs = input_images,outputs=labels_plus_embeddings)
opt = Adam(0.0001)
# Convolutional Neural Network
network = Sequential()
network.add(Conv2D(128, (7,7), activation='relu',
                  input_shape=input_image_shape,
                  kernel_initializer='he_uniform',
                  kernel_regularizer=l2(2e-4)))

network.add(MaxPooling2D())
network.add(Conv2D(128, (3,3), activation='relu', kernel_initializer='he_uniform',
                  kernel_regularizer=l2(2e-4)))
network.add(MaxPooling2D())
network.add(Conv2D(256, (3,3), activation='relu', kernel_initializer='he_uniform',
                  kernel_regularizer=l2(2e-4)))
network.add(Flatten())
network.add(Dense(4096, activation='relu',
                kernel_regularizer=l2(1e-3),
                kernel_initializer='he_uniform'))


network.add(Dense(embedding_size, activation=None,
                kernel_regularizer=l2(1e-3),
                kernel_initializer='he_uniform'))

#Force the encoding to live on the d-dimentional hypershpere
network.add(Lambda(lambda x: K.l2_normalize(x,axis=-1)))
#plot_model(base_network, to_file='base_network.png', show_shapes=True, show_layer_names=True)
print(network.summary())
network.compile(loss='categorical_crossentropy',
                      optimizer=opt)
print("compilation successful")
predicted = network.predict(x_train)
print(predicted)
dummy_gt_train = np.zeros((len(x_train), embedding_size))
#print("=========================>",x_train.shape[1])
"""

H = network.fit(
            x=x_train,
            y = dummy_gt_train,
            
            epochs=epochs,
            )
model.save("facenet_None.h5")
"""
