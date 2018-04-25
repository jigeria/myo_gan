'''
Park sang-min
jigeria@naver.com / jigeria114@gmail.com

This source will read myo csv files and concatenate myo data

You should modify file path

2018-01-25

'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.utils import np_utils
from keras import backend as K

'''

    print(data[0])
    print(emg[:, 0].size)
    print(data[:, 0].size)
    print(data[0, :].size)

    print(data[0, 10])
    print(data.shape)

    data array structure 

    data[:, 0] - time_stamp(emg data)
    data[:, 1:9] - emg_data1
    data[:, 9:17] - emg_data2
    data[:, 17:20] - accelerometer data
    data[:, 20:23] - gyro data
    data[:, 23:27] - orientation data
    data[:, 27:30] - orientationEuler data

    '''

input_dir = []
input_dir.append('../../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/accelerometer-.csv')
input_dir.append('../../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/gyro-.csv')
input_dir.append('../../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/orientation-.csv')
input_dir.append('../../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/orientationEuler-.csv')

data = []

for i in input_dir:
    read_file = pd.read_csv(i)
    data.append(read_file)

all_data = pd.concat(data, axis=1, join='inner', ignore_index='False')

del data, read_file

all_data = all_data.drop([4, 8, 13], axis=1)

# all_data = all_data.values.tolist()

myo_data = np.asarray(all_data, dtype=float)
# print(myo_data[:, 0].size)

emg = pd.read_csv('../../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/emg-.csv')

emg = emg.values.tolist()
emg = np.asarray(emg)

emg_data1 = []
emg_data2 = []

for i in range(emg[:, 0].size):
    if i % 2 == 0:
        emg_data1.append(emg[i])
    elif i % 2 != 0:
        emg_data2.append(emg[i])

emg_data1 = np.array(emg_data1)
emg_data2 = np.array(emg_data2)

emg_data2 = np.delete(emg_data2, np.s_[:1], axis=1)

# print(myo_data[:, 0].size)
# print(emg_data1[:, 0].size)
# print(emg_data2[:, 0].size)

myo = []
count = 0
size = (myo_data[:, 0].size * 2) - emg_data1[:, 0].size
size = emg_data1[:, 0].size - size

# print(size)

for i in range(myo_data[:, 0].size):
    myo.append(myo_data[i, 1:])
    count = count + 1

    if count <= size:
        myo.append(myo_data[i, 1:])
        count = count + 1

myo = np.asarray(myo)

# print(myo)
# print(myo[:, 0].size)

data = np.concatenate((emg_data1, emg_data2), axis=1)
myo_data = np.concatenate((data, myo), axis=1)

#print(myo_data[:, 0].size)
#print(myo_data[0, :].size)

del emg, myo, emg_data1, emg_data2, data


def show_all_data(data):

    plt.figure(1, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 1:9])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('emg1')

    plt.savefig('../plot_image/emg1.png')

    plt.figure(2, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 9:17])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('emg2')

    plt.savefig('../plot_image/emg2.png')

    plt.figure(3, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 17:20])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('accelerometer')

    plt.savefig('../plot_image/accelerometer.png')

    plt.figure(4, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 20:23])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('gyro')

    plt.savefig('../plot_image/gyro.png')

    plt.figure(5, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 23:27])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('orientation ')

    plt.savefig('../plot_image/orientation.png')

    plt.figure(6, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 27:30])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('orientationEuler')

    plt.savefig('../plot_image/orientationEuler.png')

    plt.show()

def show_data(data):

    # data.tofile('data.csv', sep=',', format='%10.5f')
    np.savetxt('.././execl/myo_data.csv', data.astype(float), delimiter=',')

    plt.figure(1, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 1:])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('data')

    plt.savefig('../plot_image/plot_image.png')
    plt.savefig('../plot_image/plot_image.pdf')

    plt.show()

def read_emg_data(myo_data):

    emg_data = myo_data[:, 0:17]
    np.savetxt('.././execl/emg_data.csv', emg_data.astype(float), delimiter=',')

    #print(emg_data[:, 0].size)
    #print(emg_data[0, :].size)

    return emg_data

def sampling_data(data):

    count = 0
    round_count = 0
    data_gap = 0
    time = 10
    time_stamp_size = data[:, 0].size
    temp = time_stamp_size / 10
    sampling_period = temp / 50
    sampling_data = []

    for i in range(time):
        for j in range(50):
            round_count = round(count)
            data_gap = data[round_count -1] - data[round_count]

            if data_gap > 15 and data_gap < 15:
                sampling_data.append(data[round_count-1, :])
            else:
                sampling_data.append(data[round_count, :])

            count += sampling_period
            print(count, round(count))

            #print(sampling_data)

    #print( round(count))
    print(time_stamp_size)
    sampling_data = np.array(sampling_data)

    print(sampling_data[:, 0].size)
    print(sampling_data[0, :].size)

    plt.figure(1, figsize=(16, 8))
    plt.plot(sampling_data[:, 0], sampling_data[:, 1:])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('sampling_data')

    plt.figure(1, figsize=(16, 8))
    plt.plot(data[:, 0], data[:, 1:])
    plt.grid
    plt.xlabel('time stamp')
    plt.ylabel('emg_data')

    plt.show()

    np.savetxt('.././execl/sampling_emg_data.csv', sampling_data.astype(float), delimiter=',')

#show_data(myo_data)
#show_all_data(myo_data)
emg_data = read_emg_data(myo_data)
#sampling_data(emg_data)