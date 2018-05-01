'''
        Author          : MagmaTart
        Last Modified   : 05/01/2018
'''

import pandas as pd
import numpy as np
import cv2

import os

class DataLoader:
    def __init__(self, data_path='./MYO_Dataset_label/'):
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        self.data_path = data_path

        self.emg_file_index = 1
        self.image_dir_index = 1

        self.emg_index = 0
        self.image_index = 0

        number_of_files = len(os.listdir(self.data_path))
        assert number_of_files % 2 == 0, "Directory count and CSV files count are not matching"
        self.data_files_count = int(number_of_files / 2)

        self.image_dir_file_list = []
        self.image_file_names = []
        self.image_file_labels = []

        self.set_new_image_directory(self.image_dir_index)

        # Property : All images count (All Data count)

    def set_new_image_directory(self, image_dir_index):
        '''
                새로운 디렉토리로 순회를 변경할 때마다
                디렉토리 내부 이미지를 읽기 위한 세팅
        '''

        self.image_dir_file_list = os.listdir(self.data_path + str(image_dir_index) + '/')
        self.image_file_names = []
        self.image_file_labels = []

        for i in range(len(self.image_dir_file_list)):
            self.image_file_names.append(self.image_dir_file_list[i][:-4].split('-')[0])
            self.image_file_labels.append(int(self.image_dir_file_list[i][:-4].split('-')[1]))

            # print(self.image_file_names[0])

    def load_emg_data(self):
        '''
                3초 분량 EMG 데이터 로드
                return : (300, 16)
        '''

        csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())

        # 600개를 읽기 전 끊겨, 다음 파일로 넘겨 읽어야 할 경우
        if csv_file.shape[0] - self.emg_index < 600:
            emg_data_a = csv_file[self.emg_index:, 1:]
            remained_length = 600 - (csv_file.shape[0] - self.emg_index)
            self.emg_file_index = (self.emg_file_index % self.data_files_count) + 1
            csv_file = np.array(
                pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())
            emg_data_b = csv_file[0:remained_length, 1:]
            self.emg_index = remained_length
            emg_data = np.append(emg_data_a, emg_data_b, axis=0)
        else:
            emg_data = csv_file[self.emg_index:self.emg_index + 600, 1:]
            self.emg_index += 600

        emg_data = np.reshape(emg_data, (int(emg_data.shape[0] / 2), emg_data.shape[1] * 2))
        return emg_data

    def load_image(self):
        '''
                3초 분량 이미지 1장 로드
                return : (128, 128, 1)
        '''

        image_name = 'hand' + str(self.image_index)
        image_index = self.image_file_names.index(image_name)
        label = self.image_file_labels[image_index]
        image = cv2.imread(self.data_path + str(self.image_dir_index) + '/' + image_name + '-' + str(label) + '.png',
                           cv2.IMREAD_GRAYSCALE)
        image = np.reshape(image, (128, 128, 1))

        self.image_index += 1
        if self.image_index >= len(self.image_file_names):
            self.image_index = 0
            self.image_dir_index = (self.image_dir_index % self.data_files_count) + 1
            self.set_new_image_directory(self.image_dir_index)

        return image, label

    def get_emg_datas(self, num):
        emg_data = []

        for i in range(num):
            emg_data.append(self.load_emg_data())

        return np.array(emg_data)

    def get_images(self, num):
        images = []
        labels = []

        for i in range(num):
            image, label = self.load_image()
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)



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