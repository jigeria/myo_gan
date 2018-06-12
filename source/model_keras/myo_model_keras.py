import glob
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # 也可以使用 tensorflow
# os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,exception_verbosity=high'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda,optimizer=fast_compile'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import keras.backend as K

K.set_image_data_format('channels_last')

import time
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, LSTM, Concatenate, Dense, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten, UpSampling2D, AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.datasets import mnist
from urllib.request import urlretrieve
from keras.optimizers import RMSprop, SGD, Adam, sgd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import model_from_json

# from read_data import *
#a = % pwd
#print(a)

#% run / root / jupyter / inspace / sang - min / myo_proejct / load_data.py
from load_data import DataLoader_Continous

# %load load_data import DataLoader
# from load_data import DataLoader

class MYO_GAN():
    def __init__(self):
        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/', is_real_image=False)

        self.lstm_size = (300, 16)
        self.noise_size = 100
        self.image_size = 128
        self.input_size = 100
        self.image_channel = 1
        self.learning_rate = 2e-4
        self.epoch = 1000
        self.batch_size = 32
        self.emg_size = 8

        self.image_input = Input(shape=(self.image_size, self.image_size, self.image_channel))
        self.emg_input = Input(shape=(self.emg_size, ))

        self.adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)  # as described in appendix A of DeepMind's AC-GAN paper


    def make_condition_model(self):
        _ = inputs = Input(shape=(self.image_size, self.image_size, self.image_channel))

        _ = Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(128, 128, 1))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=512, kernel_size=(1, 1), strides=2, padding='same', input_shape=(64, 64, 256))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(32, 32, 512))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(16, 16, 256))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(8, 8, 128))(_)
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=self.image_channel, kernel_size=(3, 3), strides=1, padding='same', input_shape=(4, 4, 128))(
            _)

        outputs = Flatten()(_)
        outputs = Dense(8, activation='relu')(outputs)

        return Model(inputs=inputs, outputs=outputs)

    def train_condition_model(self, make_condition_model):

        #intput = Input(shape=(self.emg_size, ))
        model = Sequential()
        model.add(make_condition_model)


    def train(self, net_condition):
        i = 0

        condition_output = net_condition(self.image_input)
        train_y = Input(shape=(self.emg_size, ))

        estimate_net_c = Model(inputs=[self.image_input], outputs=[condition_output], name='estimate_net_c')

        net_condition.compile(loss='mean_squared_error', optimizer=self.adam)
        estimate_net_c.compile(loss='mean_squared_error', optimizer=self.adam)

        while i <= self.epoch:
            images = self.loader.get_images(self.batch_size)
            emg_data = self.loader.get_emg_datas(self.batch_size)

            loss = estimate_net_c.train_on_batch(images, emg_data)

            print("%d [loss: %f]" % (loss))


if __name__ =='__main__':
    myo_gan = MYO_GAN()

    make_condition = myo_gan.make_condition_model()
    make_condition.summary()

    myo_gan.train(make_condition)


