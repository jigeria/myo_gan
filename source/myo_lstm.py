'''
Park sang-min
jigeria@naver.com / jigeria114@gmail.com

LSTM Layer and Random Z vactor concatenate

LSTM output layer change tensor -> tensor z vactor concatenate with LSTM layer output

2018-03-07

'''

import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Input, merge, concatenate, Concatenate
from keras.utils import np_utils

from read_data import *

myo_data = read_data()
#show_data(myo_data)

print(myo_data[:, 0].size)

input = Input(shape=(1, 1))
input_2 = Input(shape=(20, ))

print(input_2)
_ = LSTM(80, input_shape=(myo_data[:, 0].size, 29))
print(_)

_ = (_)(input)
print(_)

concan = Concatenate(axis=1)([_, input_2])
print(concan)
