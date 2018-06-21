import glob
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # 也可以使用 tensorflow
# os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,exception_verbosity=high'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda,optimizer=fast_compile'
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import keras.backend as K

K.set_image_data_format('channels_last')

import time
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, LSTM, Concatenate, Dense, concatenate, MaxPooling2D
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.datasets import mnist
from urllib.request import urlretrieve
from keras.utils import np_utils
from keras.optimizers import RMSprop, SGD, Adam
import matplotlib.pyplot as plt
import cv2
import numpy as np

from load_data import DataLoader_Continous

class MYO_LSTM():
    def __init__(self):
        self.emg_length = 100  # 0.5s
        self.emg_feature = 8
        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/', is_real_image=False, data_type=2,is_flatten=False, emg_length=self.emg_length)
        self.adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)  # as described in appendix A of DeepMind's AC-GAN paper

        self.epoch = 100
        self.time_0 = time.time()
        self.batch_size = 64

        self.d_step = 1
        self.g_step = 2

        self.loss_history = []

        self.lstm_size = ((int)(self.emg_length / 2), self.emg_feature)  # data_type = 2 -> (10, 8)  / 0 -> (20, 8)
        self.noise_size = 90
        self.lstm_output_size = 64
        self.image_size = 128
        self.input_size = 100
        self.image_channel = 1
        self.learning_rate = 2e-4
        self.emg_size = (int)(self.emg_length / 2) * self.emg_feature
        self.image_input = Input(shape=(self.image_size, self.image_size, self.image_channel))

        self.lstm_input = Input(shape=self.lstm_size,)

        self.lstm_model = self.lstm()
        #self.image_to_condition_model = self.image_to_condition()

        self.lstm_model.summary()
        #self.image_to_condition_model.summary()

        #elf.condition = self.image_to_condition_model(self.image_input)

        #self.condition_model = Model(inputs=self.lstm_input, outputs=self.condition, name='condition_model')
        #self.condition_model.summary()

        self.lstm_model.compile(loss='mean_squared_error', optimizer=self.adam)
        #self.condition_model.compile(loss='mean_squared_error', optimizer=self.adam)


    def resize_image(self, input_image):
        images = []

        for _ in range(self.batch_size):
            img = cv2.resize(input_image[_], None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            img = img.flatten()

            images.append(img)

        images = np.asarray(images)
#        print(images)

        return images

    def train(self):
        i = 0

        while i <= self.epoch:
            images = self.loader.get_images(self.batch_size)
            emg = self.loader.get_emg_datas(self.batch_size)

            resized_images = self.resize_image(images)
            print(resized_images.shape)

            #cv2.imwrite('./myo_lstm_output/' + 'real_image' + str(i) + '.png', images[0] * 127.5)
            #temp = np.reshape(resized_images[0], (32, 32))
            #cv2.imwrite('./myo_lstm_output/' + 'resize_image' + str(i) + '.png', temp * 127.5)

            loss = self.lstm_model.train_on_batch(emg, resized_images)
            self.loss_history.append(loss)

            print("%d [loss: %f]" % (i, loss))

            if i % 10 == 0:
                self.save_model(i)

            i += 1

    def show_history(self):
        plt.figure(1, figsize=(16, 8))
        plt.plot(self.loss_history)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./myo_lstm_output/lstm_model_loss.png')

    def save_model(self, num):
        self.lstm_model.save_weights("./myo_lstm_output/lstm_model_" + str(num) + ".h5")

        lstm_model_json = self.lstm_model.to_json()
        with open("./myo_lstm_output/lstm_model_" + str(num) + ".json", "w") as json_file:
            json_file.write(lstm_model_json)

        print('model saved')

    def lstm(self):

        _ = LSTM(self.lstm_output_size, input_shape=self.lstm_size)(self.lstm_input)
        _ = Dropout(0.3)(_)
        _ = Dense(64, input_shape=(self.lstm_output_size,), activation='relu')(_)

        return Model(inputs=self.lstm_input, outputs=_)

    def image_to_condition(self):

        _ = MaxPooling2D(pool_size=(2, 2))(self.image_input) # 128 to 64
        _ = MaxPooling2D(pool_size=(2, 2))(_) # 64 to 32
        _ = MaxPooling2D(pool_size=(2, 2))(_) # 32 16
        _ = MaxPooling2D(pool_size=(2, 2))(_) # 16 8
        # _ = MaxPooling2D(pool_size=(2, 2))(_) # 8 4
        _ = Flatten()(_)

        return Model(inputs=self.image_input, outputs=_)

if __name__ == '__main__':

    myo_lstm = MYO_LSTM()
    myo_lstm.train()
    myo_lstm.show_history()
    myo_lstm.save_model(myo_lstm.batch_size)
