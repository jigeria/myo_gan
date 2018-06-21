import glob
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # 也可以使用 tensorflow
# os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,exception_verbosity=high'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda,optimizer=fast_compile'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
#import DataLoader

# %load load_data import DataLoader

from load_data import DataLoader_Continous


class MYO_GAN():
    def __init__(self):
        self.conv_init = RandomNormal(0, 0.02)
        self.gamma_init = RandomNormal(1., 0.02)

        self.emg_length = 100  # 0.5s
        self.emg_feature = 8
        self.loader = DataLoader_Continous(data_path='./dataset_2018_05_16/', is_real_image=False, data_type=2, is_flatten=False, emg_length=self.emg_length)
        self.adam = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999)  # as described in appendix A of DeepMind's AC-GAN paper

        self.epoch = 2
        self.batch_size = 16

        self.d_step = 1
        self.g_step = 2

        self.g_loss_history = []
        self.d_loss_real_history = []
        self.d_loss_fake_history = []

        self.lstm_size = ((int)(self.emg_length / 2), self.emg_feature)  # data_type = 2 -> (10, 8)  / 0 -> (20, 8)
        self.lstm_output_size = 64
        self.noise_size = (int)(100 - self.lstm_output_size)
        self.image_size = 128
        self.input_size = 100
        self.image_channel = 1
        self.learning_rate = 2e-4

        self.lstm_input = Input(shape=self.lstm_size,)
        self.condition_intput = Input(shape=(self.lstm_output_size,))
        self.noise_input = Input(shape=(self.noise_size,))

        real_image = Input(shape=(self.image_size, self.image_size, self.image_channel))


    def build_model(self):
        self.lstm_model = self.load_lstm()
        self.lstm_model.summary()

        self.net_g = self.generative_model()
        fake_image = self.net_g([self.noise_input, self.condition_intput])
        self.net_g.summary()

        self.net_d = self.discriminative_model()
        self.net_d.summary()

        combined_output = self.net_d(fake_image)
        self.combined_model = Model(inputs=[self.noise_input, self.condition_intput], outputs=[combined_output], name='combined')

        '''
        net_g, net_d, combined_model = load_model()

        net_g.summary()
        net_d.summary()

        fake_image = net_g(self.noise_input)
        '''

        self.net_g.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.net_d.compile(loss='binary_crossentropy', optimizer=self.adam)
        self.net_d.trainable = False
        self.combined_model.compile(loss='binary_crossentropy', optimizer=self.adam)

        self.combined_model.summary()

    def load_lstm(self):

        json_file = open('./model_load/myo_lstm_output/lstm_model_4000.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        load_lstm_model = model_from_json(loaded_model_json)
        # load weights into new model
        load_lstm_model.load_weights("./model_load/myo_lstm_output/lstm_model_4000.h5")

        print('load model')

        return load_lstm_model

    def generative_model(self):

        concat = Concatenate(axis=-1)([self.noise_input, self.condition_intput])

        _ = Dense(256, input_shape=(100,), activation='relu')(concat)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Reshape((16, 16, 1), input_shape=(256,))(_)

        _ = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', input_shape=(16, 16, 1))(
            _)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(16, 16, 128))(_)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=512, kernel_size=3, padding='same', input_shape=(32, 32, 256))(_)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        _ = UpSampling2D()(_)
        _ = Conv2D(filters=256, kernel_size=3, padding='same', input_shape=(64, 64, 512))(_)
        _ = BatchNormalization(axis=1)(_, training=1)
        _ = Activation(activation='relu')(_)

        # _ = UpSampling2D()(_)
        _ = Conv2D(filters=1, kernel_size=3, padding='same', input_shape=(128, 128, 256))(_)
        _ = Activation(activation='tanh')(_)

        return Model(inputs=[self.noise_input, self.condition_intput], outputs=_)

    def discriminative_model(self):
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
        _ = Conv2D(filters=self.image_channel, kernel_size=(2, 2), strides=1, padding='same', input_shape=(4, 4, 128))(
            _)
        # _ = LeakyReLU(alpha=0.2)(_)

        outputs = Flatten()(_)
        outputs = Dense(1, activation='sigmoid')(outputs)

        return Model(inputs=inputs, outputs=outputs)

    def load_model(self):
        json_file = open('./model3_output/load_model/g_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        load_g_model = model_from_json(loaded_model_json)
        # load weights into new model
        load_g_model.load_weights("./model3_output/load_model/g_model.h5")

        json_file = open('./model3_output/load_model/d_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        load_d_model = model_from_json(loaded_model_json)
        # load weights into new model
        load_d_model.load_weights("./model3_output/load_model/d_model.h5")

        json_file = open('./model3_output/load_model/combined_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        load_combined_model = model_from_json(loaded_model_json)
        # load weights into new model
        load_combined_model.load_weights("./model3_output/load_model/combined_model.h5")

        print("Loaded model from disk")

        return [load_g_model, load_d_model, load_combined_model]

    def sample_generation(self):
        emg = self.loader.get_emg_datas(self.batch_size)

        for _ in range(self.batch_size):
            noise = np.random.normal(size=[self.batch_size, self.noise_size])
            condition = self.lstm_model.predict(emg)
            gan_image = self.net_g.predict(noise, condition)
            cv2.imwrite('./model2_output/image/' + 'sample image' + str(_) + '.png', gan_image[_] * 127.5)

        print("generated image")

    def save_model(self):
        self.net_g.save_weights("./model3_output/save_model/g_model.h5")
        self.net_d.save_weights("./model3_output/save_model/d_model.h5")
        self.combined_model.save_weights("./model3_output/save_model/combined_model.h5")

        g_model_json = self.net_g.to_json()
        with open("./model3_output/save_model/g_model.json", "w") as json_file:
            json_file.write(g_model_json)

        d_model_json = self.net_d.to_json()
        with open("./model3_output/save_model/d_model.json", "w") as json_file:
            json_file.write(d_model_json)

        combined_model_json = self.combined_model.to_json()
        with open("./model3_output/save_model/combined_model.json", "w") as json_file:
            json_file.write(combined_model_json)

        print("Saved model to disk")

    def show_history(self):
        plt.figure(1, figsize=(16, 8))
        plt.plot(self.d_loss_real_history)
        plt.ylabel('d_loss_real')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./model2_output/image/d_loss_real_history.png')

        plt.figure(2, figsize=(16, 8))
        plt.plot(self.d_loss_fake_history)
        plt.ylabel('d_loss_fake')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./model2_output/image/d_loss_fake_history.png')

        plt.figure(3, figsize=(16, 8))
        plt.plot(self.g_loss_history)
        plt.ylabel('g_loss')
        plt.xlabel('epoch')
        plt.legend(['train'], loc='upper left')

        plt.savefig('./model2_output/image/g_loss_history.png')

        plt.show()

    def train(self):
        i = 0

        while i <= self.epoch:
            images = self.loader.get_images(self.batch_size)
            emg = self.loader.get_emg_datas(self.batch_size)

            for _ in range(self.d_step):
                noise = np.random.normal(size=[self.batch_size, self.noise_size])

                condition = self.lstm_model.predict_on_batch(emg)

                g_z = self.net_g.predict([noise, condition])

                d_loss_real = self.net_d.train_on_batch(images,
                                                        np.random.uniform(low=0.7, high=1.2, size=self.batch_size))
                d_loss_fake = self.net_d.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.3, size=self.batch_size))

                self.d_loss_real_history.append(d_loss_real)
                self.d_loss_fake_history.append(d_loss_fake)
                self.d_loss = np.sum([d_loss_fake, d_loss_real])

            for _ in range(self.g_step):
                noise = np.random.normal(size=[self.batch_size, self.noise_size])
                combined_loss = self.combined_model.train_on_batch([noise, condition], np.random.uniform(low=0.7, high=1.2,
                                                                                            size=self.batch_size))

                self.g_loss_history.append(combined_loss)

            print("%d [D loss real: %f] [D loss fake: %f] [D loss: %f] [G loss: %f]" % (i, d_loss_real, d_loss_fake, self.d_loss, combined_loss))

            if i % 500 == 0:
                gan_image = self.net_g.predict([noise, condition])
                print("gan imaga2 : ", gan_image[0].shape)
                cv2.imwrite('./model2_output/image/' + 'fake_image' + str(i) + '.png', gan_image[0] * 127.5)
                # cv2.imwrite('./output_image3/' + 'real_image'+ str(i) + '.png', images[0] * 127.5)

            i += 1;

        #self.sample_generation()

if __name__ == '__main__':
    myo_gan = MYO_GAN()
    myo_gan.build_model()
    myo_gan.train()
    myo_gan.show_history()
    myo_gan.save_model()

    print("Finish")