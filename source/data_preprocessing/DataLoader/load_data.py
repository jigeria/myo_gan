import pandas as pd
import numpy as np
import cv2

import os

class DataLoader:
    def __init__(self, emg_data_path='./time_match.csv', image_path='./hand_images/'):
        self.emg_data_path = emg_data_path
        self.image_path = image_path

        self.emg_data = np.array(pd.read_csv(self.emg_data_path, sep=',').values.tolist())
        self.total_emg_seconds = self.emg_data[-1][0]

        for dirname, dirnames, filenames in os.walk(self.image_path):
            self.total_image_number = len(filenames)

        self.image_index = 0
        self.emg_data_index = 0

    def load_emg_data(self):
        index = self.emg_data_index % self.total_emg_seconds
        emg_data = self.emg_data[index*200 : index*200+200, 1:]
        emg_data = np.reshape(emg_data, (100, 16))
        emg_data = np.asarray(emg_data, dtype=np.float32)
        self.emg_data_index += 1
        return emg_data

    def load_image(self):
        index = self.image_index % self.total_image_number
        filename = self.image_path + 'hand' + str(index) + '.png'
        image = np.reshape(cv2.imread(filename, cv2.IMREAD_GRAYSCALE), (128, 128, 1))
        self.image_index += 1
        return image

    def get_next_emgs(self, num):
        emgs = []
        for i in range(num):
            emgs.append(self.load_emg_data())

        return np.array(emgs)

    def get_next_images(self, num):
        images = []
        for i in range(num):
            images.append(self.load_image())

        return np.array(images)

    def get_next_second_emgs(self):
        emgs = []
        for i in range(3):
            emgs.append(self.load_emg_data())

        return np.reshape(np.array(emgs), (1, 300, 16))

# Example

'''
loader = DataLoader(emg_data_path='./Sample_data/emg.csv', image_path='./Sample_data/hand_images/')
print('Total image number :', loader.total_image_number, 'Total EMG seconds :', loader.total_emg_seconds)

# Get next EMG datas or Images
test_emg = loader.get_next_second_emgs()
test_image = loader.get_next_images(1)

print(type(test_emg), test_emg.shape)
print(type(test_image), test_image.shape)
test_emg = loader.get_next_second_emgs()
print(test_emg[0][0])
'''