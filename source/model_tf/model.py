'''
        Author          : MagmaTart
        Last Modified   : 05/06/2018
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.rnn as rnn
import numpy as np

def leaky_relu(input, slope=0.2):
    return tf.nn.relu(input) - slope * tf.nn.relu(-input)

# TODO : Build Model, Make train and test phase

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

class Model:
    def __init__(self, mode='train', batch_size=1, labels=9, learning_rate=0.1, is_real_image=False):
        self.mode = mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.is_real_image = is_real_image

        self.label_num = labels

    def lstm(self, emg_data, reuse=False):
        print(emg_data.shape)

        with tf.variable_scope('lstm', reuse=reuse):
            cell = rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)
            initial_state = cell.zero_state(self.batch_size, tf.float32)
            outputs, _states = tf.nn.dynamic_rnn(cell, emg_data, initial_state=initial_state, dtype=np.float32)

            print(outputs.shape, 'A')
            # 300줄이 순차적으로 RNN에 들어가고, 최종 출력은 outputs의 마지막 row임
            outputs = outputs[:, -1]
            print(outputs.shape)
            # outputs = tflayers.fully_connected(inputs=outputs, num_outputs=80, activation_fn=None)
            outputs = tflayers.fully_connected(inputs=outputs, num_outputs=80)

            # 80을 Generator의 입력으로 쓰고, Labeling을 통한 학습을 위해 Label 개수에 맞춘 출력을 하나 더 놓음
            if self.mode == 'train':
                return outputs
            elif self.mode == 'pretrain':
                print(outputs.shape)
                outputs = tflayers.fully_connected(inputs=outputs, num_outputs=self.label_num)
                print(outputs.shape, 'K')
                return outputs

    def generator_edge(self, emg_data, z, reuse=False):
        # (None, 300, 16) -> (1, 100)
        # input = self.lstm(emg_data)
        _ = self.lstm(emg_data, reuse)
        input = z
        print(input.shape)
        # input = tf.concat([input, z],  1)
        # print(input.shape)

        # Generator
        with tf.variable_scope('generator_edge', reuse=reuse):
            net = slim.fully_connected(input, 64*8*8, activation_fn=tf.nn.relu,
                                          weights_initializer=tflayers.xavier_initializer(),
                                          reuse=reuse)
            print(net)
            net = tf.reshape(net, [-1, 8, 8, 64])
            net = slim.conv2d(net, num_outputs=128, kernel_size=1, stride=1)
            print(net)
            # with slim.arg_scope([slim.conv2d_transpose], padding='SAME', kernel_size=3, stride=2,
            #                     weights_initializer=tflayers.xavier_initializer()):
            #     with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
            #                         activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
            #         net = slim.conv2d_transpose(net, num_outputs=128)
            #         net = slim.batch_norm(net)
            #         print(net)
            #         net = slim.conv2d_transpose(net, num_outputs=256)       # output : 32 x 32
            #         net = slim.batch_norm(net)
            #         print(net)
            #         net = slim.conv2d_transpose(net, num_outputs=512)       # output : 64 x 64
            #         net = slim.batch_norm(net)
            #         print(net)
            #         net = slim.conv2d_transpose(net, num_outputs=256)       # output : 128 x 128
            #         net = slim.batch_norm(net)
            #         print(net)

            # 16 x 16
            net = slim.conv2d_transpose(net, num_outputs=128, kernel_size=3, stride=2,
                                        weights_initializer=tflayers.xavier_initializer())
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)
            print(net)

            # 32 x 32
            net = slim.conv2d_transpose(net, num_outputs=256, kernel_size=3, stride=2,
                                        weights_initializer=tflayers.xavier_initializer())
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)
            print(net)

            # 64 x 64
            net = slim.conv2d_transpose(net, num_outputs=512, kernel_size=3, stride=2,
                                        weights_initializer=tflayers.xavier_initializer())
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)
            print(net)

            #128 x 128
            net = slim.conv2d_transpose(net, num_outputs=256, kernel_size=3, stride=2,
                                        weights_initializer=tflayers.xavier_initializer())
            net = slim.batch_norm(net, activation_fn=tf.nn.relu)
            print(net)

            net = slim.conv2d_transpose(net, num_outputs=1, kernel_size=1, stride=1, padding='VALID',
                                        activation_fn=tf.nn.relu,
                                        weights_initializer=tflayers.xavier_initializer())
            print(net)

            return net

    def generator_real(self, emg_data, z, reuse=False):
        # (None, 300, 16) -> (1, 100)
        # input = self.lstm(emg_data)
        _ = self.lstm(emg_data, reuse)
        input = z
        print(input.shape)
        # input = tf.concat([input, z],  1)
        # print(input.shape)

        # Generator
        with tf.variable_scope('generator_real', reuse=reuse):
            net = slim.fully_connected(input, 256, activation_fn=tf.nn.relu,
                                       weights_initializer=tflayers.xavier_initializer(),
                                       reuse=reuse)
            print(net)
            net = tf.reshape(net, [-1, 16, 16, 1])
            net = slim.conv2d(net, num_outputs=128, kernel_size=1, stride=1)
            print(net)
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', kernel_size=2, stride=2,
                                weights_initializer=tflayers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu, is_training=(self.mode == 'train')):
                    net = slim.conv2d_transpose(net, num_outputs=256)  # output : 32 x 32
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d_transpose(net, num_outputs=512)  # output : 64 x 64
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d_transpose(net, num_outputs=256)  # output : 128 x 128
                    print(net)
                    net = slim.batch_norm(net)

            net = slim.conv2d_transpose(net, num_outputs=3, kernel_size=1, stride=1, padding='VALID',
                                        weights_initializer=tflayers.xavier_initializer())
            print(net)

            return net

    def discriminator_edge(self, input, reuse=False):
        net = input
        print(net)

        with tf.variable_scope("discriminator_edge", reuse=reuse):
            with slim.arg_scope([slim.conv2d], kernel_size=3, stride=2,
                                weights_initializer=tflayers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=leaky_relu, is_training=(self.mode=='train')):
                    net = slim.conv2d(net, num_outputs=256)     # 64 x 64
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=512)     # 32 x 32
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=256)     # 16 x 16
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=128)     # 8 x 8
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=128)     # 4 x 4
                    net = slim.batch_norm(net)
                    print(net)

            net = slim.conv2d(net, num_outputs=1, kernel_size=4, stride=1, activation_fn=leaky_relu,
                              weights_initializer=tflayers.xavier_initializer())
            print(net)
            net = slim.flatten(net)
            print(net)

        return net

    def discriminator_real(self, input, reuse=False):
        net = input
        print(net)

        with tf.variable_scope("discriminator_real", reuse=reuse):
            with slim.arg_scope([slim.conv2d], kernel_size=2, stride=2, padding='SAME',
                                weights_initializer=tflayers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=leaky_relu, is_training=(self.mode == 'train')):
                    net = slim.conv2d(net, num_outputs=256)  # 64 x 64
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=512)  # 32 x 32
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=256)  # 16 x 16
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=128)  # 8 x 8
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=128)  # 4 x 4
                    print(net)
                    net = slim.batch_norm(net)

            net = slim.conv2d(net, num_outputs=1, kernel_size=4, stride=1, activation_fn=leaky_relu,
                              weights_initializer=tflayers.xavier_initializer())
            print(net)
            net = slim.flatten(net)
            print(net)

        return net

    def build(self):

        if self.is_real_image:
            self.real_image = tf.placeholder(tf.float32, [None, 128, 128, 3])
        else:
            self.real_image = tf.placeholder(tf.float32, [None, 128, 128, 1])

        self.emg_data = tf.placeholder(tf.float32, [None, 300, 16])
        # self.z = tf.placeholder(tf.float32, [None, 20])
        self.z = tf.placeholder(tf.float32, [None, 1000])
        # self.c = tf.placeholder(tf.float32, [None, 20])
        self.y_label = tf.placeholder(tf.int64, [None])         # one-hot label

        if self.mode == 'train':
            if self.is_real_image:
                self.fake_image = self.generator_real(self.emg_data, self.z, reuse=False)
                self.fake_logits = self.discriminator_real(self.fake_image)
                self.real_logits = self.discriminator_real(self.real_image, reuse=True)
            else:
                self.fake_image = self.generator_edge(self.emg_data, self.z, reuse=False)
                self.fake_logits = self.discriminator_edge(self.fake_image)
                self.real_logits = self.discriminator_edge(self.real_image, reuse=True)

            self.d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.fake_logits), self.fake_logits)
            self.d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_logits), self.real_logits)
            self.d_loss = tf.reduce_mean(self.d_loss_fake + self.d_loss_real)

            self.g_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_logits), self.fake_logits))

            self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss)
            self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss)

        elif self.mode == 'pretrain':
            print('Before one-hot', self.y_label.shape)
            self.y_onehot = tf.one_hot(self.y_label, self.label_num)
            print('After one-hot :', self.y_onehot)

            self.lstm_logits = self.lstm(self.emg_data)
            self.lstm_prediction = tf.argmax(self.lstm_logits, 1)
            print('lstm_logits :', self.lstm_logits.shape)
            self.lstm_loss = tf.losses.softmax_cross_entropy(self.y_onehot, self.lstm_logits)
            # self.lstm_prediction = tf.one_hot(self.lstm_logits, self.label_num)
            # self.lstm_prediction = self.lstm_logits
            print('lstm_prediction :', self.lstm_prediction.shape)
            self.lstm_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.lstm_prediction, self.y_label), tf.float32))

            self.lstm_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.lstm_loss)