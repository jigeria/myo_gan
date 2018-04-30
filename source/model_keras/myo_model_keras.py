import glob
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # 也可以使用 tensorflow
# os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,exception_verbosity=high'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda,optimizer=fast_compile'
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import keras.backend as K

K.set_image_data_format('channels_last')

import time
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, LSTM, Concatenate, Dense, concatenate
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.datasets import mnist
from urllib.request import urlretrieve
from keras.optimizers import RMSprop, SGD, Adam

#from read_data import *
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
loader = DataLoader(emg_data_path='../data_preprocessing/DataLoader/Sample_data/emg.csv', image_path='../data_preprocessing/DataLoader/Sample_data/hand_images/')

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

lstm_size = (300, 16)
#print("lstm size : ", lstm_size)
noise_size = 20
image_size = 128
input_size = 100
image_channel = 1
learning_rate = 2e-4

lstm_input = Input(shape=(lstm_size))
noise_input = Input(shape=(noise_size,))
image_input = Input(shape=(image_size, image_size, image_channel))

lstm_model = Sequential([
    LSTM(80, input_shape=(lstm_size)),
    Activation(activation='tanh')
])
lstm_model.summary()

lstm_output = lstm_model(lstm_input)

lstm_layer = Model(inputs=lstm_input, outputs=lstm_output)

generative_model = Sequential([
    Dense(256, input_shape=(100,), activation='relu', bias_initializer='glorot_normal',
          kernel_initializer='glorot_normal'),
    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Reshape((16, 16, 1), input_shape=(256, )),

    Conv2D(filters=128, kernel_size=(2, 2), padding='same', kernel_initializer=conv_init,
            input_shape=(16, 16, 1)),
    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Activation(activation='relu'),

    Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=2, kernel_initializer=conv_init,
                    input_shape=(16, 16, 128)),
    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Activation(activation='relu'),

    Conv2DTranspose(filters=512, kernel_size=(2, 2), strides=2, kernel_initializer=conv_init,
                    input_shape=(32, 32, 256)),
    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Activation(activation='relu'),

    Conv2DTranspose(filters=256, kernel_size=(2, 2), strides=2, kernel_initializer=conv_init,
                    input_shape=(64, 64, 512)),
    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Activation(activation='relu'),

    Conv2D(filters=1, kernel_size=(128, 128), padding='same', kernel_initializer=conv_init,
           input_shape=(128, 128, 256)),
    Activation(activation='relu')

])
generative_model.summary()

g_model_input = Concatenate(axis=-1)([lstm_output, noise_input])
g_model_output = generative_model(g_model_input)

combined_model = Model(inputs=[lstm_input, noise_input], outputs=[g_model_output])


discriminative_model = Sequential([

    Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(128, 128, 1), kernel_initializer=conv_init),
    LeakyReLU(alpha=0.2),

    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Conv2D(filters=512, kernel_size=(2, 2), strides=2, padding='same', input_shape=(64, 64, 256), kernel_initializer=conv_init),
    LeakyReLU(alpha=0.2),

    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Conv2D(filters=256, kernel_size=(2, 2), strides=2, padding='same', input_shape=(32, 32, 512), kernel_initializer=conv_init),
    LeakyReLU(alpha=0.2),

    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(16, 16, 256), kernel_initializer=conv_init),
    LeakyReLU(alpha=0.2),

    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Conv2D(filters=128, kernel_size=(2, 2), strides=2, padding='same', input_shape=(8, 8, 128), kernel_initializer=conv_init),
    LeakyReLU(alpha=0.2),

    BatchNormalization(axis=1, gamma_initializer=gamma_init),
    Conv2D(filters=1, kernel_size=(2, 2), strides=1, padding='same', input_shape=(4, 4, 128), kernel_initializer=conv_init),
    #_ = LeakyReLU(alpha=0.2)(_)

    Flatten()
])
discriminative_model.summary()

d_model_output = discriminative_model(image_input)

d_model = Model(inputs=image_input, outputs=d_model_output)
