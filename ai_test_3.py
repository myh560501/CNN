import numpy as np
import os
import PIL
import PIL.Image
from keras import *
from keras.models import *
from keras.layers import *
from torch import *
import torch.nn as nn
from keras.optimizers import *
#from keras.model import *
from keras.datasets import cifar100
from keras.losses import sparse_categorical_crossentropy

model = Sequential()
 
v = 32
#v = 224

model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(v,v,3),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1000,activation='softmax'))

class Network(object):
    def __init__(self):
        
        #self = model
        
         
        (self.train, self.train_label), (self.test, self.test_label) = \
        cifar100.load_data()

        self.train = self.train.reshape(-1, 32, 32, 3) / 255.0
        self.test = self.test.reshape(-1, 32, 32, 3) / 255.0

    def compile(self):
        #opt = Adam()
        #loss = sparse_categorical_crossentropy()
        
        #model.compile(loss='mean_squared_error', optimizer='sgd')

        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        return None

    def fit(self):
        model.fit(self.train, self.train_label, epochs=1, batch_size=32)
        
        return None
    
    def evaluate(self):

        test_loss, test_acc = model.evaluate(self.test, self.test_label)
        
        print(test_loss, test_acc)
        return None


if __name__ == '__main__':

    import time

    cnn = Network()

    cnn.compile()

    print('Start training.....')

    for adadadadadada in [1,2,3,4,5]:
        print('-------------' * 8  +  '%' + str(time.localtime()))
        
        cnn.fit()
        
        cnn.predict()
        
        print(model.summary())
    print('Finshed training......')
    model.save('AIMODELS')
    