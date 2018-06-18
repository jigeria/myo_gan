<<<<<<< HEAD
'''
        Author          : MagmaTart
        Last Modified   : 05/16/2018
'''

import pandas as pd
import numpy as np
import cv2

import os

class DataLoader_discrete:
    def __init__(self, data_path='./MYO_Dataset_label/', is_real_image=False, is_original_data=False): #data_type 0 is original data / 1 is calc_osclliation_degree
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        self.data_path = data_path

        self.is_real_image = is_real_image
        self.is_original_data = is_original_data

        if is_real_image:
            self.emg_file_index = 1
            self.image_dir_index = 1
        else:
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

        print(self.data_files_count)

        # Property : All images count (All Data count)

    def set_new_image_directory(self, image_dir_index):
        image_dir_path = self.data_path + str(image_dir_index) + '/'
        print(image_dir_index)

        if self.is_real_image:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'real' in name]
        else:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'edge' in name]

        self.image_file_names = []
        self.image_file_labels = []

        # print(self.image_dir_file_list)

        for i in range(len(self.image_dir_file_list)):
            self.image_file_names.append(self.image_dir_file_list[i][:-4].split('-')[0] + '-' +
                                         self.image_dir_file_list[i][:-4].split('-')[1])
            self.image_file_labels.append(int(self.image_dir_file_list[i][:-4].split('-')[2]))

        # print(self.image_file_names)
        # print(self.image_file_labels)

    def load_emg_data(self):
        emg_length = 40     # 0.2 sec

        csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())

        if csv_file.shape[0] - self.emg_index < emg_length:
            emg_data_a = csv_file[self.emg_index:, 1:]
            remained_length = emg_length - (csv_file.shape[0] - self.emg_index)
            self.emg_file_index = (self.emg_file_index % self.data_files_count) + 1
            csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())
            emg_data_b = csv_file[0:remained_length, 1:]
            self.emg_index = remained_length
            emg_data = np.append(emg_data_a, emg_data_b, axis=0)
        else:
            emg_data = csv_file[self.emg_index:self.emg_index + emg_length, 1:]
            self.emg_index += emg_length

        # 같은 Timestamp끼리 차의 평균 계산

        if self.is_original_data:
            return emg_data
        else:
            emg_data = self.calc_osclliation_degree(emg_data)
            max = sorted(emg_data)[-1]
            emg_data = emg_data / max

            return emg_data

    def calc_osclliation_degree(self, emg_data):
        TAG = 'calc_osclliation_degree >'
        osc = []
        total = 0

        # print(TAG, len(emg_data))

        for k in range(8):
            for i in range(0, len(emg_data), 2):
                if emg_data[i][k] > 0 and emg_data[i+1][k] > 0:
                    total += emg_data[i][k] + emg_data[i+1][k]
                elif emg_data[i][k] < 0 and emg_data[i+1][k] < 0:
                    total += emg_data[i][k] + emg_data[i+1][k]
                else:
                    # print(emg_data[i][k], emg_data[i+1][k], abs(emg_data[i][k] - emg_data[i+1][k]))
                    total += abs(emg_data[i][k] - emg_data[i+1][k])
            osc.append(total / int(len(emg_data)/2))
            total = 0

        return np.array(osc, dtype=np.float32)

    def load_image(self):
        if self.is_real_image:
            image_name = 'hand-real' + str(self.image_index)
            image_index = self.image_file_names.index(image_name)
            label = self.image_file_labels[image_index]
            image = cv2.imread(self.data_path + str(self.image_dir_index) + '/' + image_name + '-' + str(label) + '.png')
            image = np.reshape(image, (128, 128, 3))

        else:
            image_name = 'hand-edge' + str(self.image_index)
            image_index = self.image_file_names.index(image_name)
            label = self.image_file_labels[image_index]
            image = cv2.imread(self.data_path + str(self.image_dir_index) + '/' + image_name + '-' + str(label) + '.png',
                               cv2.IMREAD_GRAYSCALE)
            image = np.reshape(image, (128, 128, 1))

        self.image_index += 1

        if self.image_index >= len(self.image_dir_file_list):
            self.image_index = 0
            self.image_dir_index = (self.image_dir_index % self.data_files_count) + 1
            self.set_new_image_directory(self.image_dir_index)

        return image, label

    def get_emg_datas(self, num):
        emg_data = []

        for i in range(num):
            emg_data.append(self.load_emg_data())

        return np.array(emg_data, dtype=np.float32)

    def get_images(self, num):
        images = []
        labels = []

        for i in range(num):
            image, label = self.load_image()
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

class DataLoader_Continous:
    def __init__(self, data_path='./MYO_Dataset_label/', is_real_image=False, data_type=0): #data_type 0 is original data / 1 is calc_osclliation_degree
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        self.data_path = data_path

        self.is_real_image = is_real_image

        self.data_type = data_type

        if is_real_image:
            self.emg_file_index = 1
            self.image_dir_index = 1
        else:
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

        print(self.data_files_count)

        # Property : All images count (All Data count)

    def set_new_image_directory(self, image_dir_index):
        '''
                새로운 디렉토리로 순회를 변경할 때마다
                디렉토리 내부 이미지를 읽기 위한 세팅
        '''

        image_dir_path = self.data_path + str(image_dir_index) + '/'
        print(image_dir_path)

        if self.is_real_image:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'real' in name]
        else:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'edge' in name]

        # self.image_file_names = []
        # self.image_file_labels = []

        # print(self.image_dir_file_list)

        # for i in range(len(self.image_dir_file_list)):
        #     self.image_file_names.append(self.image_dir_file_list[i][:-4].split('-')[0] + '-' +
        #                                  self.image_dir_file_list[i][:-4].split('-')[1])
        #     self.image_file_labels.append(int(self.image_dir_file_list[i][:-4].split('-')[2]))

    def load_emg_data(self):
        emg_length = 20     # 0.1 sec

        csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())

        if csv_file.shape[0] - self.emg_index < emg_length:
            emg_data_a = csv_file[self.emg_index:, 1:]
            remained_length = emg_length - (csv_file.shape[0] - self.emg_index)
            self.emg_file_index = (self.emg_file_index % self.data_files_count) + 1
            csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())
            emg_data_b = csv_file[0:remained_length, 1:]
            self.emg_index = remained_length
            emg_data = np.append(emg_data_a, emg_data_b, axis=0)
        else:
            emg_data = csv_file[self.emg_index:self.emg_index + emg_length, 1:]
            self.emg_index += emg_length

        # 같은 Timestamp끼리 차의 평균 계산


        if self.data_type == 0:

            return emg_data

        elif self.data_type == 1:
            emg_data = self.calc_osclliation_degree(emg_data)
            max = sorted(emg_data)[-1]
            emg_data = emg_data / max

            return emg_data

        return emg_data

    def calc_osclliation_degree(self, emg_data):
        TAG = 'calc_osclliation_degree >'
        osc = []
        total = 0

        # print(TAG, len(emg_data))

        for k in range(8):
            for i in range(0, len(emg_data), 2):
                if emg_data[i][k] > 0 and emg_data[i+1][k] > 0:
                    total += emg_data[i][k] + emg_data[i+1][k]
                elif emg_data[i][k] < 0 and emg_data[i+1][k] < 0:
                    total += abs(emg_data[i][k] + emg_data[i+1][k])
                else:
                    # print(emg_data[i][k], emg_data[i+1][k], abs(emg_data[i][k] - emg_data[i+1][k]))
                    total += abs(emg_data[i][k] - emg_data[i+1][k])
            osc.append(total / int(len(emg_data)/2))
            total = 0

        return np.array(osc, dtype=np.float32)

    def load_image(self):
        '''
                이미지 1장 로드
                return : (128, 128, 1)
        '''

        if self.is_real_image:
            image_name = 'hand-real' + str(self.image_index) + '.png'
            # image_index = self.image_file_names.index(image_name)
            # image_index = self.image_dir_file_list.index(image_name)
            # label = self.image_file_labels[image_index]
            image = cv2.imread(
                self.data_path + str(self.image_dir_index) + '/' + image_name)
            image = np.reshape(image, (128, 128, 3))

        else:
            image_name = 'hand-edge' + str(self.image_index) + '.png'
            # image_index = self.image_file_names.index(image_name)
            # label = self.image_file_labels[image_index]
            # print(self.image_dir_index, image_name, label)
            image = cv2.imread(
                self.data_path + str(self.image_dir_index) + '/' + image_name,
                cv2.IMREAD_GRAYSCALE)
            # print(self.data_path + str(self.image_dir_index) + '/' + image_name + '-' + str(label) + '.png')
            image = np.reshape(image, (128, 128, 1))

        # print(self.image_dir_index)
        # print(self.data_files_count)
        self.image_index += 1
        # print(len(self.image_dir_file_list))
        if self.image_index >= len(self.image_dir_file_list):
            self.image_index = 0
            self.image_dir_index = (self.image_dir_index % self.data_files_count) + 1
            # self.image_dir_index = (self.image_dir_index % 5) + 1
            self.set_new_image_directory(self.image_dir_index)

        image = cv2.GaussianBlur(image, (7, 7), 0)
        image = np.reshape(image, (128, 128, 1))
               # print(image.shape)
        # cv2.imshow('Blurred', image)
        # cv2.waitKey(100000)
        # cv2.destroyAllWindows()
        return np.array(image, dtype=np.float32) / 127.5

    def get_emg_datas(self, num):
        emg_data = []

        emg_size_per_second = 20
        emg_data_channel = 8

        for i in range(num):
            emg_data.append(self.load_emg_data())

        emg_data = np.array(emg_data, dtype=np.float32)
        #emg_data = emg_data.reshape(num, emg_size_per_second*emg_data_channel)

        return emg_data

    def get_images(self, num):
        images = []
        # labels = []

        for i in range(num):
            # image, label = self.load_image()
            image = self.load_image()
            images.append(image)
            # labels.append(label)

        # return np.array(images), np.array(labels)
        return np.array(images)

def vis(emg):
    lst = []
    total = 0
    for k in range(8):
        for i in range(len(emg)):
            total += emg[i][k]
        lst.append(total)
        total = 0

    print(lst)

    for i in range(len(lst)):
        for n in range(int(lst[i] / 100)):
            print('■', end='')
        print()

    # max = sorted(lst)[-1]
    # npa = np.array(lst, dtype=np.float32) / max
    # print(npa)

# loader = DataLoader_Continous(data_path='./myo_test_dataset/')
# loader = DataLoader_Continous(data_path='./dataset_0516/')

# print('Paper')
#
# emg = loader.get_emg_datas(15)
# vis(emg)
#
# print('Scissor')
#
# emg = loader.get_emg_datas(15)
# # print(emg)
# vis(emg)
#
# print('Small scissor')
#
# emg = loader.get_emg_datas(15)
# # print(emg)
# vis(emg)
#
# print('Fist')

'''
for i in range(20):
    emg = loader.get_emg_datas(5)

    # if i == 0 or i == 1:
    print('Data number :', i)
    print(emg)
    # vis(emg)
'''

# emg = loader.load_emg_data()
# image, label = loader.load_image()

# print(emg)

# cv2.imshow('test', image)
# cv2.waitKey(100000)
# cv2.destroyAllWindows()

# emg = loader.load_emg_data()
# image, label = loader.load_image()

# print(emg)
# print(label)

# cv2.imshow('test', image)
# cv2.waitKey(100000)
# cv2.destroyAllWindows()
#
# emg = loader.get_emg_datas(7)
#
# for i in range(500):
#     images = loader.get_images(7)
#     print(i, images.shape)


loader = DataLoader_Continous(data_path='./dataset_2018_05_16/', is_real_image=False, data_type=0)
emg = loader.get_emg_datas(10)

print(emg)
print(emg.shape)

=======
'''
        Author          : MagmaTart
        Last Modified   : 06/18/2018
'''

import pandas as pd
import numpy as np
import cv2

import os

class DataLoader_discrete:
    def __init__(self, data_path='./MYO_Dataset_label/', is_real_image=False):
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        self.data_path = data_path

        self.is_real_image = is_real_image

        if is_real_image:
            self.emg_file_index = 1
            self.image_dir_index = 1
        else:
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

        print(self.data_files_count)

        # Property : All images count (All Data count)

    def set_new_image_directory(self, image_dir_index):
        image_dir_path = self.data_path + str(image_dir_index) + '/'
        print(image_dir_index)

        if self.is_real_image:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'real' in name]
        else:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'edge' in name]

        self.image_file_names = []
        self.image_file_labels = []

        # print(self.image_dir_file_list)

        for i in range(len(self.image_dir_file_list)):
            self.image_file_names.append(self.image_dir_file_list[i][:-4].split('-')[0] + '-' +
                                         self.image_dir_file_list[i][:-4].split('-')[1])
            self.image_file_labels.append(int(self.image_dir_file_list[i][:-4].split('-')[2]))

        # print(self.image_file_names)
        # print(self.image_file_labels)

    def load_emg_data(self):
        emg_length = 600     # 0.2 sec

        csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())

        if csv_file.shape[0] - self.emg_index < emg_length:
            emg_data_a = csv_file[self.emg_index:, 1:]
            remained_length = emg_length - (csv_file.shape[0] - self.emg_index)
            self.emg_file_index = (self.emg_file_index % self.data_files_count) + 1
            csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())
            emg_data_b = csv_file[0:remained_length, 1:]
            self.emg_index = remained_length
            emg_data = np.append(emg_data_a, emg_data_b, axis=0)
        else:
            emg_data = csv_file[self.emg_index:self.emg_index + emg_length, 1:]
            self.emg_index += emg_length

        emg_data = self.RMS_analyzer(emg_data)
        return emg_data

    def RMS_analyzer(self, emg_data):
        # print(emg_data.shape)
        rms = []
        total = 0

        for i in range(emg_data.shape[1]):
            for k in range(emg_data.shape[0]):
                emg_data[k][i] *= emg_data[k][i]

        for k in range(0, emg_data.shape[0], 2):
            lst = []
            for i in range(emg_data.shape[1]):
                lst.append( (emg_data[k][i] + emg_data[k+1][i]) / 2.0 )
            rms.append(lst)

        for i in range(len(rms[0])):
            for k in range(len(rms)):
                rms[k][i] = rms[k][i] ** 0.5

        # (300, 8)
        rms = np.array(rms, dtype=np.float32)
        return rms

    def calc_osclliation_degree(self, emg_data):
        TAG = 'calc_osclliation_degree >'
        osc = []
        total = 0

        # print(TAG, len(emg_data))

        for k in range(8):
            for i in range(0, len(emg_data), 2):
                if emg_data[i][k] > 0 and emg_data[i+1][k] > 0:
                    total += emg_data[i][k] + emg_data[i+1][k]
                elif emg_data[i][k] < 0 and emg_data[i+1][k] < 0:
                    total += emg_data[i][k] + emg_data[i+1][k]
                else:
                    # print(emg_data[i][k], emg_data[i+1][k], abs(emg_data[i][k] - emg_data[i+1][k]))
                    total += abs(emg_data[i][k] - emg_data[i+1][k])
            osc.append(total / int(len(emg_data)/2))
            total = 0

        return np.array(osc, dtype=np.float32)

    def load_image(self):
        if self.is_real_image:
            image_name = 'hand-real' + str(self.image_index)
            image_index = self.image_file_names.index(image_name)
            label = self.image_file_labels[image_index]
            image = cv2.imread(self.data_path + str(self.image_dir_index) + '/' + image_name + '-' + str(label) + '.png')
            image = np.reshape(image, (128, 128, 3))

        else:
            image_name = 'hand-edge' + str(self.image_index)
            image_index = self.image_file_names.index(image_name)
            label = self.image_file_labels[image_index]
            image = cv2.imread(self.data_path + str(self.image_dir_index) + '/' + image_name + '-' + str(label) + '.png',
                               cv2.IMREAD_GRAYSCALE)
            image = np.reshape(image, (128, 128, 1))

        self.image_index += 1

        if self.image_index >= len(self.image_dir_file_list):
            self.image_index = 0
            self.image_dir_index = (self.image_dir_index % self.data_files_count) + 1
            self.set_new_image_directory(self.image_dir_index)

        return image, label

    def get_emg_datas(self, num):
        emg_data = []

        for i in range(num):
            emg_data.append(self.load_emg_data())

        return np.array(emg_data, dtype=np.float32)

    def get_images(self, num):
        images = []
        labels = []

        for i in range(num):
            image, label = self.load_image()
            images.append(image)
            labels.append(label)

        return np.array(images), np.array(labels)

class DataLoader_Continous:
    def __init__(self, data_path='./MYO_Dataset_label/', emg_length=600, is_real_image=False):
        if data_path[-1] is not '/':
            data_path = data_path + '/'
        self.data_path = data_path

        self.emg_length = emg_length
        self.is_real_image = is_real_image

        if is_real_image:
            self.emg_file_index = 1
            self.image_dir_index = 1
        else:
            self.emg_file_index = 1
            self.image_dir_index = 1

        self.emg_index = 0
        self.image_index = 10

        number_of_files = len(os.listdir(self.data_path))
        assert number_of_files % 2 == 0, "Directory count and CSV files count are not matching"
        self.data_files_count = int(number_of_files / 2)

        self.image_dir_file_list = []
        self.image_file_names = []
        self.image_file_labels = []

        self.set_new_image_directory(self.image_dir_index)

        print(self.data_files_count)

        # Property : All images count (All Data count)

    def set_new_image_directory(self, image_dir_index):
        '''
                새로운 디렉토리로 순회를 변경할 때마다
                디렉토리 내부 이미지를 읽기 위한 세팅
        '''

        image_dir_path = self.data_path + str(image_dir_index) + '/'
        print(image_dir_path)

        if self.is_real_image:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'real' in name]
        else:
            self.image_dir_file_list = [name for name in os.listdir(image_dir_path) if 'edge' in name]

        # self.image_file_names = []
        # self.image_file_labels = []

        # print(self.image_dir_file_list)

        # for i in range(len(self.image_dir_file_list)):
        #     self.image_file_names.append(self.image_dir_file_list[i][:-4].split('-')[0] + '-' +
        #                                  self.image_dir_file_list[i][:-4].split('-')[1])
        #     self.image_file_labels.append(int(self.image_dir_file_list[i][:-4].split('-')[2]))

    def load_emg_data(self):
        csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())

        if csv_file.shape[0] - self.emg_index < self.emg_length:
            emg_data_a = csv_file[self.emg_index:, 1:]
            remained_length = self.emg_length - (csv_file.shape[0] - self.emg_index)
            self.emg_file_index = (self.emg_file_index % self.data_files_count) + 1
            csv_file = np.array(pd.read_csv(self.data_path + str(self.emg_file_index) + '.csv', sep=',').values.tolist())
            emg_data_b = csv_file[0:remained_length, 1:]
            self.emg_index = remained_length
            emg_data = np.append(emg_data_a, emg_data_b, axis=0)
        else:
            emg_data = csv_file[self.emg_index:self.emg_index + self.emg_length, 1:]
            self.emg_index += 20
            # self.emg_index += self.emg_length

        # 같은 Timestamp끼리 차의 평균 계산

        emg_data = self.RMS_analyzer(emg_data)
        return emg_data

    def RMS_analyzer(self, emg_data):
        # print(emg_data.shape)
        rms = []
        total = 0

        for i in range(emg_data.shape[1]):
            for k in range(emg_data.shape[0]):
                emg_data[k][i] *= emg_data[k][i]

        for k in range(0, emg_data.shape[0], 2):
            lst = []
            for i in range(emg_data.shape[1]):
                lst.append( (emg_data[k][i] + emg_data[k+1][i]) / 2.0 )
            rms.append(lst)

        for i in range(len(rms[0])):
            for k in range(len(rms)):
                rms[k][i] = rms[k][i] ** 0.5

        # (300, 8)
        rms = np.array(rms, dtype=np.float32)
        return rms

    def calc_osclliation_degree(self, emg_data):
        TAG = 'calc_osclliation_degree >'
        osc = []
        total = 0

        # print(TAG, len(emg_data))

        for k in range(8):
            for i in range(0, len(emg_data), 2):
                if emg_data[i][k] > 0 and emg_data[i+1][k] > 0:
                    total += emg_data[i][k] + emg_data[i+1][k]
                elif emg_data[i][k] < 0 and emg_data[i+1][k] < 0:
                    total += abs(emg_data[i][k] + emg_data[i+1][k])
                else:
                    # print(emg_data[i][k], emg_data[i+1][k], abs(emg_data[i][k] - emg_data[i+1][k]))
                    total += abs(emg_data[i][k] - emg_data[i+1][k])
            osc.append(total / int(len(emg_data)/2))
            total = 0

        return np.array(osc, dtype=np.float32)

    def load_image(self):
        '''
                이미지 1장 로드
                return : (128, 128, 1)
        '''

        if self.is_real_image:
            image_name = 'hand-real' + str(self.image_index) + '.png'
            # image_index = self.image_file_names.index(image_name)
            # image_index = self.image_dir_file_list.index(image_name)
            # label = self.image_file_labels[image_index]
            image = cv2.imread(
                self.data_path + str(self.image_dir_index) + '/' + image_name)
            image = np.reshape(image, (128, 128, 3))

        else:
            image_name = 'hand-edge' + str(self.image_index) + '.png'
            # image_index = self.image_file_names.index(image_name)
            # label = self.image_file_labels[image_index]
            # print(self.image_dir_index, image_name, label)
            image = cv2.imread(
                self.data_path + str(self.image_dir_index) + '/' + image_name,
                cv2.IMREAD_GRAYSCALE)
            # print(self.data_path + str(self.image_dir_index) + '/' + image_name + '-' + str(label) + '.png')
            image = np.reshape(image, (128, 128, 1))

        # print(self.image_dir_index)
        # print(self.data_files_count)
        self.image_index += 1
        # print(len(self.image_dir_file_list))
        if self.image_index >= len(self.image_dir_file_list):
            self.image_index = 0
            self.image_dir_index = (self.image_dir_index % self.data_files_count) + 1
            # self.image_dir_index = (self.image_dir_index % 5) + 1
            self.set_new_image_directory(self.image_dir_index)

        image = cv2.GaussianBlur(image, (7, 7), 0)
        image = np.reshape(image, (128, 128, 1))
               # print(image.shape)
        # cv2.imshow('Blurred', image)
        # cv2.waitKey(100000)
        # cv2.destroyAllWindows()
        return np.array(image, dtype=np.float32) / 127.5

    def get_emg_datas(self, num):
        emg_data = []

        for i in range(num):
            emg_data.append(self.load_emg_data())

        return np.array(emg_data, dtype=np.float32)

    def get_images(self, num):
        images = []
        # labels = []

        for i in range(num):
            # image, label = self.load_image()
            image = self.load_image()
            images.append(image)
            # labels.append(label)

        # return np.array(images), np.array(labels)
        return np.array(images)

def vis(emg):
    lst = []
    total = 0
    for k in range(8):
        for i in range(len(emg)):
            total += emg[i][k]
        lst.append(total)
        total = 0

    print(lst)

    for i in range(len(lst)):
        for n in range(int(lst[i] / 100)):
            print('■', end='')
        print()

    # max = sorted(lst)[-1]
    # npa = np.array(lst, dtype=np.float32) / max
    # print(npa)

# loader = DataLoader_Continous(data_path='./myo_test_dataset/')
# loader = DataLoader_Continous(data_path='./dataset_0516/', emg_length=200)
#
# while True:
#     emg = loader.get_emg_datas(90)
#     print(emg)
#     print(emg.shape)
#     image = loader.get_images(90)
#     cv2.imshow('emg', image[0])
#     cv2.waitKey(1000000)
#     cv2.destroyAllWindows()

# print(emg)
# cv2.imshow('emg', image)
# cv2.waitKey(50000)
# cv2.destroyAllWindows()

# print('Paper')
#
# emg = loader.get_emg_datas(15)
# vis(emg)
#
# print('Scissor')
#
# emg = loader.get_emg_datas(15)
# # print(emg)
# vis(emg)
#
# print('Small scissor')
#
# emg = loader.get_emg_datas(15)
# # print(emg)
# vis(emg)
#
# print('Fist')

'''
for i in range(20):
    emg = loader.get_emg_datas(5)

    # if i == 0 or i == 1:
    print('Data number :', i)
    print(emg)
    # vis(emg)
'''

# emg = loader.load_emg_data()
# image, label = loader.load_image()

# print(emg)

# cv2.imshow('test', image)
# cv2.waitKey(100000)
# cv2.destroyAllWindows()

# emg = loader.load_emg_data()
# image, label = loader.load_image()

# print(emg)
# print(label)

# cv2.imshow('test', image)
# cv2.waitKey(100000)
# cv2.destroyAllWindows()
#
# emg = loader.get_emg_datas(7)
#
# for i in range(500):
#     images = loader.get_images(7)
#     print(i, images.shape)
>>>>>>> a94da23a14f50c70778827e2c4f4372f011ca31d
