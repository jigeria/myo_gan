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
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, LSTM, Concatenate, Dense
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.datasets import mnist
from urllib.request import urlretrieve
from keras.optimizers import RMSprop, SGD, Adam
import numpy as np
import cv2

# from read_data import *
from load_data import DataLoader

'''
Model structure

Input : (100) Vector
Output : (64, 64, 1) image

Generator
100 -> 256
256 -> 16 x 16
16 x 16 -> 32 x 32
32 x 32 -> 64 x 64
64 x 64 -> 128 x 128

Discriminator
128 x 128 -> 64 x 64
64 x 64 -> 32 x 32
32 x 32 -> 16 x 16
16 x 16 -> 8 x 8
8 x 8 -> 64
64 -> 32 (16, 10)

'''
loader = DataLoader(data_path='./dataset_2018_05_06/')

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)

    return model


def generative_model(noise_size):
    # lstm_layer = LSTM(80, input_shape=lstm_size)(lstm_input)
    # _ = Concatenate(axis=-1)([lstm_layer, noise_input])
    print(" _ : ", noise_input)
    _ = Dense(256, input_shape=(100,), activation='relu', bias_initializer='glorot_normal',
              kernel_initializer='glorot_normal')(noise_input)
    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Reshape((16, 16, 1), input_shape=(256,))(_)

    _ = Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same', kernel_initializer=conv_init, input_shape=(16, 16, 1))(
        _)
    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Activation(activation='relu')(_)

    _ = UpSampling2D()(_)
    _ = Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer=conv_init,
                        input_shape=(16, 16, 128))(_)
    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Activation(activation='relu')(_)

    _ = UpSampling2D()(_)
    _ = Conv2D(filters=512, kernel_size=3, padding='same',  kernel_initializer=conv_init,
                        input_shape=(32, 32, 256))(_)
    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Activation(activation='relu')(_)

    _ = UpSampling2D()(_)
    _ = Conv2D(filters=256, kernel_size=3, padding='same', kernel_initializer=conv_init,
                        input_shape=(64, 64, 512))(_)
    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Activation(activation='relu')(_)

    #_ = UpSampling2D()(_)
    _ = Conv2D(filters=1, kernel_size=3, padding='same', kernel_initializer=conv_init,
               input_shape=(128, 128, 256))(_)
    _ = Activation(activation='relu')(_)

    return Model(inputs=noise_input, outputs=_)


def discriminative_model(image_size, image_channel):
    _ = inputs = Input(shape=(image_size, image_size, image_channel))

    _ = Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(128, 128, 1),
               kernel_initializer=conv_init)(_)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Conv2D(filters=512, kernel_size=(1, 1), strides=2, padding='same', input_shape=(64, 64, 256),
               kernel_initializer=conv_init)(_)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(32, 32, 512),
               kernel_initializer=conv_init)(_)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(16, 16, 256),
               kernel_initializer=conv_init)(_)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(8, 8, 128),
               kernel_initializer=conv_init)(_)
    _ = LeakyReLU(alpha=0.2)(_)

    _ = BatchNormalization(axis=1, gamma_initializer=gamma_init)(_, training=1)
    _ = Conv2D(filters=1, kernel_size=(2, 2), strides=1, padding='same', input_shape=(4, 4, 128),
               kernel_initializer=conv_init)(_)
    # _ = LeakyReLU(alpha=0.2)(_)

    outputs = Flatten()(_)
    outputs = Dense(1, activation='sigmoid')(outputs)

    return Model(inputs=inputs, outputs=outputs)


lstm_size = (300, 16)
noise_size = 100
image_size = 128
input_size = 100
image_channel = 1
learning_rate = 2e-4
optimizer = Adam(0.0002, 0.5)

# lstm_input = Input(shape=lstm_size)
noise_input = Input(shape=(noise_size,))
real_image = Input(shape=(image_size, image_size, image_channel))

net_g = generative_model(noise_size)
net_g.summary()

net_d = discriminative_model(image_size, image_channel)
net_d.summary()

net_d.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

fake_image = net_g(noise_input)

net_d.trainable = False

valid = net_d(fake_image)

combined_model = Model(inputs=noise_input, outputs=valid)
combined_model.compile(loss='binary_crossentropy', optimizer=optimizer)

# net_g.compile(loss='binary_crossentropy', optimizer='SGD')


'''
loader = DataLoader(data_path='./MYO_Dataset_label/')

emg = loader.load_emg_data()
image, label = loader.load_image()

print(emg.shape)
print(image.shape, label)

emg = loader.get_emg_datas(10)
images, labels = loader.get_images(10)
print(emg.shape, images.shape, labels.shape)

'''

epoch = 30000
i = 0
time_0 = time.time()
err_d = err_g = 0
err_d_sum = 0
err_g_sum = 0
batch_size = 32

d_label = np.zeros(shape=[batch_size])
g_label = np.zeros(shape=[batch_size])

print(d_label)
print(d_label.shape)

while i < epoch:
    # x_train = loader.get_emg_datas(batch_size)
    images, labels = loader.get_images(batch_size)
    noise = np.random.normal(size=(batch_size, noise_size))

    gan_image = net_g.predict(noise)
    print("gan imaga1 : ", gan_image.shape)

    d_loss_real = net_d.train_on_batch(images, np.ones(shape=(batch_size, 1)))
    d_loss_fake = net_d.train_on_batch(gan_image, np.zeros(shape=(batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # print("epoch[%d] d_loss[%f]" % (i, d_loss))

    g_loss = combined_model.train_on_batch(noise, np.ones(shape=(batch_size, 1)))

    # print("epoch[%d] g_loss[%f]" % (i, g_loss))

    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, d_loss[0], 100 * d_loss[1], g_loss))

    if i % 500 == 0:
        gan_image = net_g.predict(noise)
        print("gan imaga2 : ", gan_image[0].shape)
        cv2.imwrite('./output_image3/' + 'fake_image'+ str(i) + '.png', gan_image[0] * 127.5)
        cv2.imwrite('./output_image3/' + 'real_imadge'+ str(i) + '.png', images[0] * 127.5)

    i += 1;