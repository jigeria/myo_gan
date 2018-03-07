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
import csv

input_dir = []
input_dir.append('../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/accelerometer-.csv')
input_dir.append('../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/gyro-.csv')
input_dir.append('../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/orientation-.csv')
input_dir.append('../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/orientationEuler-.csv')

data = []

for i in input_dir:
    read_file = pd.read_csv(i)
    data.append(read_file)

all_data = pd.concat(data, axis=1, join='inner', ignore_index='False')

del data, read_file

all_data = all_data.drop([4, 8, 13], axis=1)

#all_data = all_data.values.tolist()

myo_data = np.asarray(all_data, dtype=float)

emg = pd.read_csv('../../Project/myo/myo-sdk-win-0.9.0/myo-sdk-win-0.9.0/samples/data/emg-.csv')

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

#print(myo_data[:, 0].size)
#print(emg_data1[:, 0].size)
#print(emg_data2[:, 0].size)

myo = []
count = 0
size = (myo_data[:, 0].size*2) - emg_data1[:, 0].size
size = emg_data1[:, 0].size - size

#print(size)

for i in range(myo_data[:, 0].size):
    myo.append(myo_data[i, 1:])
    count = count + 1

    if count <= size:
        myo.append(myo_data[i, 1:])
        count = count + 1

myo = np.asarray(myo)

#print(myo)
#print(myo[:, 0].size)

data = np.concatenate((emg_data1, emg_data2), axis=1)
data = np.concatenate((data, myo), axis=1)

print(data[0])
print(emg[:, 0].size)
print(data[:, 0].size)
print(data[0, :].size)

print(data[0, 10])
print(data.shape)

del emg, myo, myo_data, emg_data1, emg_data2

'''
data array structure 

data[:, 1] - time_stamp(emg data)
data[:, 1:9] - emg_data1
data[:, 9:17] - emg_data2
data[:, 17:20] - accelerometer data
data[:, 20:23] - gyro data
data[:, 23:27] - orientation data
data[:, 27:30] - orientationEuler data

'''
#data.tofile('data.csv', sep=',', format='%10.5f')
np.savetxt('data.csv', data.astype(float), delimiter=',')

plt.figure(1, figsize=(16, 8))
plt.plot(data[:, 0], data[:, 1:])
plt.grid
plt.xlabel('time stamp')
plt.ylabel('data')

plt.savefig('plot_image.png')
plt.savefig('plot_image.pdf')

plt.show()

