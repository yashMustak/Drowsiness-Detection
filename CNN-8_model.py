import numpy as np
from numpy import *
import keras
import os
import sys
import cv2
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from keras.utils import np_utils
from keras.optimizers import SGD
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
import warnings
warnings.filterwarnings('ignore')

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()
    return model

def load_data(channel, img_rows, img_cols, num_classes):
    path1 = 'eyes'  #path of traning data for close eyes
    path2 = 'eyes_oe' #path of traning data for open eyes
    path3 = 'eye_val' #path of validation data for close eyes
    path4 = 'eye_val_oe' #path of validation data for open eyes

    immatrix = []
    listing1 = os.listdir(path1)
    listing2 = os.listdir(path2)
    for files in listing1:
        img = cv2.imread(os.path.join(path1, files), 0)
        arr = array(img).flatten()
        immatrix.append(arr)
    for files in listing2:
        img = cv2.imread(os.path.join(path2, files), 0)
        arr = array(img).flatten()
        immatrix.append(arr)
    num_samples=size(listing1)+size(listing2)

    label=np.ones((num_samples,),dtype = int)
    #List the number of images for close eyes
    label[0:1898]=0 #close_eye
    #List the number of images for open eyes after close eyes images
    label[1898:3748]=1 #open_eye
    X_train,Y_train = shuffle(immatrix,label, random_state=2)

    immatrixv = []
    listingv1 = os.listdir(path3)
    listingv2 = os.listdir(path4)
    for files in listingv1:
        img = cv2.imread(os.path.join(path3, files), 0)
        arr = array(img).flatten()
        immatrixv.append(arr)
    for files in listingv2:
        img = cv2.imread(os.path.join(path4, files), 0)
        arr = array(img).flatten()
        immatrixv.append(arr)
    num_samplesv=size(listingv1)+size(listingv2)

    labelv=np.ones((num_samplesv,),dtype = int)
    labelv[0:150]=0
    labelv[150:300]=1
    X_valid,Y_valid = shuffle(immatrixv,labelv, random_state=2)
    X_train= np.array(X_train)
    X_valid= np.array(X_valid)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, channel)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, channel)

    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')

    X_train /=255
    X_valid /=255

    Y_train = np_utils.to_categorical(Y_train, num_classes)
    Y_valid = np_utils.to_categorical(Y_valid, num_classes)

    return X_train, Y_train, X_valid, Y_valid

input_shape = [224,224,1]
n_epochs = 1
batch_size = 2
n_classes = 2
channel = 1
img_rows, img_cols = 224, 224
X_train, Y_train, X_valid, Y_valid = load_data(channel, img_rows, img_cols, n_classes)
model = build_model(input_shape, n_classes)
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
# datagen = ImageDataGenerator(
#     featurewise_center=True,
#     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)
# datagen.fit(X_train)
model.compile(optimizer=sgd, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
# model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
#                     steps_per_epoch=len(X_train) / batch_size, epochs=n_epochs)
model.fit(X_train, Y_train,
        batch_size=batch_size,
        nb_epoch=n_epochs,
        shuffle=True,
        verbose=1,
        validation_data=(X_valid, Y_valid),
        )
predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
score = log_loss(Y_valid, predictions_valid)
model.save('simple_model.h5')
print('training completed')
