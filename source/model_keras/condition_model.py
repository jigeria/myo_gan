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
        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/', is_real_image=False, data_type=2)

        self.noise_size = 100
        self.image_size = 128
        self.input_size = 100
        self.image_channel = 1
        self.learning_rate = 2e-4
        self.epoch = 100
        self.batch_size = 16
        self.emg_size = (80,)

        self.image_input = Input(shape=(self.image_size, self.image_size, self.image_channel))
        self.emg_input = Input(shape=self.emg_size)

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
        outputs = Dense(80, activation='relu')(outputs)
        #outputs = Reshape((10, 8), input_shape=(80, ))(outputs)

        return Model(inputs=inputs, outputs=outputs)

    def train(self):
        i = 0

        self.net_condition = self.make_condition_model()
        self.net_condition.summary()
        condition_output = self.net_condition(self.image_input)

        self.estimate_net_c = Model(inputs=[self.image_input], outputs=[condition_output], name='estimate_net_c')
        self.estimate_net_c.summary()

        self.net_condition.compile(loss='mean_squared_error', optimizer=self.adam)
        self.estimate_net_c.compile(loss='mean_squared_error', optimizer=self.adam)

        while i <= self.epoch:
            images = self.loader.get_images(self.batch_size) / 127.5
            emg_data = self.loader.get_emg_datas(self.batch_size)

            loss = self.estimate_net_c.train_on_batch(images, emg_data)

            print("%d [loss: %f]" % (i, loss))

            i += 1

    def save_model(self):
        #self.net_condition.save_weights("./condition_model_save/condition_model.h5")
        #self.estimate_net_c.save_weights("./condition_model_save/estimate_model.h5")

        self.net_condition.save_weights("./condition_model_save/condition_model.h5")
        self.estimate_net_c.save_weights("./condition_model_save/estimate_model.h5")

        condition_model = self.net_condition.to_json()
        with open("./condition_model_save/condition_model.json", "w") as json_file:
            json_file.write(condition_model)

        estimate_model = self.estimate_net_c.to_json()
        with open("./condition_model_save/estimate_model.json", "w") as json_file:
            json_file.write(estimate_model)

        print("Saved model to disk")

if __name__ =='__main__':
    myo_gan = MYO_GAN()
    myo_gan.train()
    #myo_gan.save_model()


