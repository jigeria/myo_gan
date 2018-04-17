import glob
import os

os.environ['KERAS_BACKEND'] = 'tensorflow'  # 也可以使用 tensorflow
# os.environ['THEANO_FLAGS']='floatX=float32,device=cuda,exception_verbosity=high'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cuda,optimizer=fast_compile'
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import keras.backend as K

K.set_image_data_format('channels_first')
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, LSTM, Concatenate
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.datasets import mnist
from urllib.request import urlretrieve

from read_data import *
from load_data import DataLoader

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

from PIL import Image
import numpy as np
import tarfile

def make_lstm(myo_data, nz):

    input = Input(shape=(1, 1))
    noisev = Input(shape=(20,))

    _ = LSTM(nz, input_shape=(myo_data[:, 1:].size, 16))
    _ = (_)(input)

    #lstm_layer = Concatenate(axis=1)([_, noisev])

    return _

def DCGAN_D(isize, nz, nc, ndf, n_extra_layers=0):
    assert isize % 2 == 0
    _ = inputs = Input(shape=(nc, isize, isize))
    _ = Conv2D(filters=ndf, kernel_size=4, strides=2, use_bias=False,
               padding="same",
               kernel_initializer=conv_init,
               name='initial.conv.{0}-{1}'.format(nc, ndf)
               )(_)
    _ = LeakyReLU(alpha=0.2, name='initial.relu.{0}'.format(ndf))(_)
    csize, cndf = isize // 2, ndf # // == 몫 ndf == 64
    while csize > 5:
        assert csize % 2 == 0
        in_feat = cndf
        out_feat = cndf * 2
        _ = Conv2D(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
                   padding="same",
                   kernel_initializer=conv_init,
                   name='pyramid.{0}-{1}.conv'.format(in_feat, out_feat)
                   )(_)
        if 0:  # toggle batchnormalization
            _ = BatchNormalization(name='pyramid.{0}.batchnorm'.format(out_feat),
                                   momentum=0.9, axis=1, epsilon=1.01e-5,
                                   gamma_initializer=gamma_init,
                                   )(_, training=1)
        _ = LeakyReLU(alpha=0.2, name='pyramid.{0}.relu'.format(out_feat))(_)
        csize, cndf = (csize + 1) // 2, cndf * 2
    _ = Conv2D(filters=1, kernel_size=csize, strides=1, use_bias=False,
               kernel_initializer=conv_init,
               name='final.{0}-{1}.conv'.format(cndf, 1)
               )(_)
    outputs = Flatten()(_)
    return Model(inputs=inputs, outputs=outputs)

def DCGAN_G(isize, nz, nc, ngf, n_extra_layers=0):
    cngf = ngf // 2
    tisize = isize
    while tisize > 5:
        cngf = cngf * 2
        assert tisize % 2 == 0
        tisize = tisize // 2
    _ = inputs = Input(shape=(nz, ))
    _ = Reshape((nz, 1, 1))(_)
    _ = Conv2DTranspose(filters=cngf, kernel_size=tisize, strides=1, use_bias=False,
                        kernel_initializer=conv_init,
                        name='initial.{0}-{1}.convt'.format(nz, cngf))(_)
    _ = BatchNormalization(gamma_initializer=gamma_init, momentum=0.9, axis=1, epsilon=1.01e-5,
                           name='initial.{0}.batchnorm'.format(cngf))(_, training=1)
    _ = Activation("relu", name='initial.{0}.relu'.format(cngf))(_)
    csize, cndf = tisize, cngf

    while csize < isize // 2:
        in_feat = cngf
        out_feat = cngf // 2
        _ = Conv2DTranspose(filters=out_feat, kernel_size=4, strides=2, use_bias=False,
                            kernel_initializer=conv_init, padding="same",
                            name='pyramid.{0}-{1}.convt'.format(in_feat, out_feat)
                            )(_)
        _ = BatchNormalization(gamma_initializer=gamma_init,
                               momentum=0.9, axis=1, epsilon=1.01e-5,
                               name='pyramid.{0}.batchnorm'.format(out_feat))(_, training=1)

        _ = Activation("relu", name='pyramid.{0}.relu'.format(out_feat))(_)
        csize, cngf = csize * 2, cngf // 2
    _ = Conv2DTranspose(filters=nc, kernel_size=4, strides=2, use_bias=False,
                        kernel_initializer=conv_init, padding="same",
                        name='final.{0}-{1}.convt'.format(cngf, nc)
                        )(_)
    outputs = Activation("tanh", name='final.{0}.tanh'.format(nc))(_)
    return Model(inputs=inputs, outputs=outputs)

loader = DataLoader(emg_data_path='./Sample_data/time_match.csv', image_path='./Sample_data/hand_images/')

data_counter = 0
nc = 3
nz = 1598# noise z random vector
ngf = 64
ndf = 64
n_extra_layers = 0
Diters = 5
Lambda = 10

imageSize = 128 # image size
batchSize = loader.num_images

lrD = 1e-4 # learning rate
lrG = 1e-4 # learning rate

netD = DCGAN_D(imageSize, nz, nc, ndf, n_extra_layers)
netD.summary()

netG = DCGAN_G(imageSize, nz, nc, ngf, n_extra_layers)
netG.summary()

from keras.optimizers import RMSprop, SGD, Adam

netD_real_input = Input(shape=(nc, imageSize, imageSize))
emg_data = loader.get_emg_datas(1)
lstm_layer = make_lstm(emg_data, nz)

#lstm_layer = make_lstm(emg_data, time, per_second)
#lstm_layer = Input(shape=(nz,))
netD_fake_input = netG(lstm_layer)
#   data_counter += (int)(per_second * time)

Epsilon_input = K.placeholder(shape=(None, nc, imageSize, imageSize))
netD_mixed_input = Input(shape=(nc, imageSize, imageSize), tensor=netD_real_input + Epsilon_input)

loss_real = K.mean(netD(netD_real_input))
loss_fake = K.mean(netD(netD_fake_input))

grad_mixed = K.gradients(netD(netD_mixed_input), [netD_mixed_input])[0]
norm_grad_mixed = K.sqrt(K.sum(K.square(grad_mixed), axis=[1, 2, 3]))
grad_penalty = K.mean(K.square(norm_grad_mixed - 1))

loss = loss_fake - loss_real + Lambda * grad_penalty

imgae, data = loader.get_next_batch(batchSize)

train_X = imgae
print('imgae shape', train_X.shape)
training_updates = Adam(lr=lrD).get_updates(netD.trainable_weights, [], loss)
netD_train = K.function([netD_real_input, lstm_layer, Epsilon_input],
                        [loss_real, loss_fake],
                        training_updates)

loss = -loss_fake
training_updates = Adam(lr=lrG).get_updates(netG.trainable_weights, [], loss)
netG_train = K.function([lstm_layer], [loss], training_updates)

#train_X = np.concatenate([train_X, test_X])
#train_X = np.concatenate([train_X[:, :, :, ::-1], train_X])

fixed_noise = np.random.normal(size=(batchSize, nz)).astype('float32')

import time

count_1 = 0
count_2 = 0

t0 = time.time()
niter = 100
gen_iterations = 0
errG = 0
targetD = np.float32([2] * batchSize + [-2] * batchSize)[:, None]
targetG = np.ones(batchSize, dtype=np.float32)[:, None]

for epoch in range(niter):
    i = 0
    #print(count_1)
    #print(count_2)
    count_1 = 0
    count_2 = 0
    #np.random.shuffle(train_X)
    batches = train_X.shape[0] # batchSize
    #batches = 21
    print(batches)
    #print(batchSize)

    while i < batches:
        print('i x batchsize :', i * batchSize, (i + 1) * batchSize, len(train_X))
        real_data = train_X[i * batchSize:(i + 1) * batchSize]  # i가 1 이상이 되는 순간 인덱스를 넘어버림
        # real_data = train_X[i : i+1]
        # i += 1    # 여기서 i가 올라감
        # lstm_noise_layer =
        emg_data = loader.get_emg_datas(1)

        emg_data = emg_data.reshape(emg_data[0, :, 0].size, emg_data[0, 0, :].size)
        # print(emg_data[:, 0].size)
        # print(emg_data[0, :].size)
        size = emg_data[:, 0].size / 3
        size = int(size)
        # print(size)
        count = 0
        data = []

        for i in range(3):
            data_1 = emg_data[count: count + size, :]
            data_1 = data_1.flatten()
            # print(data_1.shape)
            data.append(data_1)

        # emg_data = emg_data.flatten()
        data = np.asarray(data)
        # print(data.shape)
        emg_data = data
        print(emg_data.shape)

        count_2 += 1
        Epsilon = real_data.std() * np.random.uniform(-0.5, 0.5, size=real_data.shape)
        print(real_data.shape)
        # print(np.random.uniform(size=(batchSize, 1, 1, 1)).shape)
        Epsilon *= np.random.uniform(size=(batchSize, 1, 1, 1))
        errD_real, errD_fake = netD_train([real_data, emg_data, Epsilon])
        errD = errD_real - errD_fake

        if gen_iterations % 10 == 0:
            # if gen_iterations % 500 == 0:
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, niter, i, batches, gen_iterations, errD, errG, errD_real, errD_fake), time.time() - t0)
            fake = netG.predict(fixed_noise)
            # showX(fake, 4)

        # noise = np.random.normal(size=(batchSize, nz))
        emg_datas = loader.get_emg_datas(3)
        errG, = netG_train([emg_data])
        gen_iterations += 1

'''
    while i < batches:
        count_1 += 1
        if gen_iterations < 25 or gen_iterations % 500 == 0:
            _Diters = 10 # 100
        else:
            _Diters = Diters
        j = 0
        print(j, '/', _Diters)
        while j < _Diters and i < batches:
            print(j, '/', _Diters)
            j += 1
            i = 0       # 인덱스 넘는 걸 방지하기 위해서 매 Iteration마다 0으로 초가화
            print('i x batchsize :', i * batchSize, (i+1) * batchSize, len(train_X))
            real_data = train_X[i * batchSize:(i + 1) * batchSize]      # i가 1 이상이 되는 순간 인덱스를 넘어버림
            # real_data = train_X[i : i+1]
            # i += 1    # 여기서 i가 올라감
            #lstm_noise_layer =
            emg_data = loader.get_emg_datas(1)

            emg_data = emg_data.reshape(emg_data[0, :, 0].size, emg_data[0, 0, :].size)
            #print(emg_data[:, 0].size)
            #print(emg_data[0, :].size)
            size = emg_data[:, 0].size / 3
            size = int(size)
            #print(size)
            count = 0
            data = []

            for i in range(3):
                data_1 = emg_data[count : count + size, :]
                data_1 = data_1.flatten()
                #print(data_1.shape)
                data.append(data_1)

            #emg_data = emg_data.flatten()
            data = np.asarray(data)
            #print(data.shape)
            emg_data = data
            #print(emg_data.shape)


            #h_size = 200 - emg_datas[:, 0].size
            #w_size = 100 - emg_datas[0, :].size
            #noise = np.random.normal(size=(100, 200))
            #print(noise.shape)
            #lstm_noise_layer = make_lstm(emg_datas)

            count_2 += 1
            Epsilon = real_data.std() * np.random.uniform(-0.5, 0.5, size=real_data.shape)
            print(real_data.shape)
            # print(np.random.uniform(size=(batchSize, 1, 1, 1)).shape)
            Epsilon *= np.random.uniform(size=(batchSize, 1, 1, 1))
            errD_real, errD_fake = netD_train([real_data, emg_data , Epsilon])
            errD = errD_real - errD_fake

        i += 1      # 테스트

        if gen_iterations % 10 == 0:
        # if gen_iterations % 500 == 0:
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
                  % (epoch, niter, i, batches, gen_iterations, errD, errG, errD_real, errD_fake), time.time() - t0)
            fake = netG.predict(fixed_noise)
            #showX(fake, 4)

        #noise = np.random.normal(size=(batchSize, nz))
        emg_datas = loader.get_emg_datas(3)
        errG, = netG_train([emg_data])
        gen_iterations += 1
'''