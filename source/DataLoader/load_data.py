import pandas as pd
import numpy as np
import cv2

import os

class DataLoader:
    def __init__(self, emg_data_path='./time_match.csv', image_path='./hand_images/'):
        self.emg_data_path = emg_data_path
        self.image_path = image_path

        self.emg_data = np.loadtxt(self.emg_data_path, delimiter=',')

        for dirname, dirnames, filenames in os.walk(self.image_path):
            self.num_images = len(filenames)

        self.data_batch_size = int(len(self.emg_data) / self.num_images)
        # self.data_index = 0
        self.image_index = 0
        self.emg_data_index = 0

    def get_images(self, num):
        images = []
        for i in range(self.image_index, self.image_index + num):
            # print(i%self.num_images)
            images.append(np.reshape(cv2.imread(self.image_path + 'hand' + str(i%self.num_images) + '.png'), (3, 128, 128)))

        return np.asarray(images)

    def get_emg_datas(self, num):
        emg_datas = []
        for i in range(self.emg_data_index, self.emg_data_index + num):
            emg_datas.append(self.emg_data[(i%self.num_images) * self.data_batch_size : (i%self.num_images) * self.data_batch_size + self.data_batch_size])

        return np.asarray(emg_datas)

    def get_next_batch(self, num):
        # print('Current batch :', self.data_index)
        images = self.get_images(num)
        emg_datas = self.get_emg_datas(num)
        self.emg_data_index = (self.emg_data_index + num) % self.num_images
        self.image_index = (self.image_index + num) % self.num_images

        return images, emg_datas

    def print_info(self):
        print(self.emg_data.shape)
        print(self.num_images)
        print(self.data_batch_size)


# Example
'''
loader = DataLoader(emg_data_path='./Sample_data/time_match.csv', image_path='./Sample_data/hand_images/')
image, data = loader.get_next_batch(5)
image, data = loader.get_next_batch(5)
'''