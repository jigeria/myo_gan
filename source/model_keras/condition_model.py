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
        self.emg_length = 100
        self.emg_feature = 8

        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/',is_real_image=False, data_type=2, is_flatten=True, emg_length=self.emg_length)

        self.noise_size = 100
        self.image_size = 128
        self.input_size = 100
        self.image_channel = 1
        self.learning_rate = 2e-4
        self.epoch = 1000
        self.batch_size = 16
        self.emg_size = (int)(self.emg_length / 2) * self.emg_feature

        self.loss_history = []

        self.image_input = Input(shape=(self.image_size, self.image_size, self.image_channel))
        self.emg_input = Input(shape=(self.emg_size,))

        self.adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)  # as described in appendix A of DeepMind's AC-GAN paper


    def make_condition_model(self):
        _ = inputs = Input(shape=(self.image_size, self.image_size, self.image_channel))

        _ = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02), bias_initializer=RandomNormal(0, 0.02), input_shape=(128, 128, 1))(_) # 64 64
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same',kernel_initializer=RandomNormal(0, 0.02), bias_initializer=RandomNormal(0, 0.02), input_shape=(64, 64, 256))(_) # 32 32
        _ = LeakyReLU(alpha=0.2)(_)

        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Conv2D(filters=512, kernel_size=(3, 3), strides=2, padding='same', kernel_initializer=RandomNormal(0, 0.02), bias_initializer=RandomNormal(0, 0.02), input_shape=(32, 32, 512))(_) #16 16
        _ = LeakyReLU(alpha=0.2)(_)

        _ = Flatten()(_)

        outputs = Dense(self.emg_size, activation='relu', kernel_initializer=RandomNormal(0, 0.02), bias_initializer='glorot_normal')(_)

        return Model(inputs=inputs, outputs=outputs)

    def train(self):
        i = 0

        self.net_condition = self.make_condition_model()
        self.net_condition.summary()
        #condition_output = self.net_condition(self.image_input)

        #self.estimate_net_c = Model(inputs=[self.image_input], outputs=[condition_output], name='estimate_net_c')
        #self.estimate_net_c.summary()

        self.net_condition.compile(loss='mean_squared_error', optimizer=self.adam)
        #self.estimate_net_c.compile(loss='mean_squared_error', optimizer=self.adam)

        while i <= self.epoch:
            images = self.loader.get_images(self.batch_size) / 255.0
            emg_data = self.loader.get_emg_datas(self.batch_size)

            loss = self.net_condition.train_on_batch(images, emg_data)
            self.loss_history.append(loss)

            print("%d [loss: %f]" % (i, loss))
            print(loss)

            i += 1

    def save_model(self):
        self.net_condition.save_weights("./condition_model_save/condition_model.h5")
        #self.save_weights("./condition_model_save/estimate_model.h5")

        condition_model = self.net_condition.to_json()
        with open("./condition_model_save/condition_model.json", "w") as json_file:
            json_file.write(condition_model)

        #estimate_model = self.estimate_net_c.to_json()
        #with open("./condition_model_save/estimate_model.json", "w") as json_file:
         #   json_file.write(estimate_model)

        print("Saved model to disk")

    def show_history(self):
        plt.figure(1, figsize=(16, 8))
        plt.plot(self.loss_history)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./loss_history.png')

if __name__ =='__main__':
    myo_gan = MYO_GAN()

    myo_gan.train()
    myo_gan.save_model()
    myo_gan.show_history()



