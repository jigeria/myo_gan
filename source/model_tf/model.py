'''
        Author          : MagmaTart
        Last Modified   : 06/22/2018
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

num_classes = 9

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
            if self.mode == 'train' or self.mode == 'myo-lstm-test':
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

        # False로 만들면 2의 배수 구조 (현재)
        # True로 만들고 구조 변경 실험
        test = True

        # Generator
        with tf.variable_scope('generator_edge', reuse=reuse):
            # with tf.device('/gpu:0'):
            net = slim.fully_connected(input, 64*8*8, activation_fn=tf.nn.relu,
                                          weights_initializer=tflayers.xavier_initializer(),
                                          reuse=reuse)
            print(net)
            net = tf.reshape(net, [-1, 8, 8, 64])
            net = slim.conv2d(net, num_outputs=128, kernel_size=1, stride=1)
            print(net)

            if not test:
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=3, stride=2,
                                    weights_initializer=tflayers.xavier_initializer()):
                    with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                        net = slim.conv2d_transpose(net, num_outputs=512)
                        net = slim.batch_norm(net)
                        print(net)
                        net = slim.conv2d_transpose(net, num_outputs=256)       # output : 32 x 32
                        net = slim.batch_norm(net)
                        print(net)
                        net = slim.conv2d_transpose(net, num_outputs=128)       # output : 64 x 64
                        net = slim.batch_norm(net)
                        print(net)
                        net = slim.conv2d_transpose(net, num_outputs=1)       # output : 128 x 128
                        net = slim.batch_norm(net)
                        print(net)

                # net = slim.conv2d_transpose(net, num_outputs=1, kernel_size=1, stride=1, padding='SAME',
                #                             activation_fn=tf.nn.relu,
                #                             weights_initializer=tflayers.xavier_initializer_conv2d())

            else:
                # 2x Upsampling -> Conv2D
                # Xavier -> Truncated normal
                # 16 x 16
                print('TEST : FALSE')
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=3, stride=1, padding='SAME',
                                    normalizer_fn=tflayers.batch_norm,
                                    weights_initializer=tf.random_normal_initializer(stddev=0.02)):
                    # with slim.arg_scope([slim.batch_norm], is_training=(self.mode=='train')):
                    net = tf.image.resize_images(net, [16, 16])
                    net = slim.conv2d_transpose(net, num_outputs=512)
                    # net = slim.batch_norm(net)
                    print(net)

                    # 32 x 32
                    net = tf.image.resize_images(net, [32, 32])
                    net = slim.conv2d_transpose(net, num_outputs=256)
                    # net = slim.batch_norm(net)
                    print(net)

                    # 64 x 64
                    net = tf.image.resize_images(net, [64, 64])
                    net = slim.conv2d_transpose(net, num_outputs=128)
                    # net = slim.batch_norm(net)
                    print(net)

                    # 128 x 128
                    net = tf.image.resize_images(net, [128, 128])
                    net = slim.conv2d_transpose(net, num_outputs=1)
                    # net = slim.batch_norm(net)
                    print(net)

                # net = slim.conv2d_transpose(net, num_outputs=1, kernel_size=1, stride=1, padding='SAME',
                #                             activation_fn=tf.nn.relu,
                #                             normalizer_fn=tflayers.batch_norm,
                #                             weights_initializer=tflayers.xavier_initializer_conv2d())

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

        test = True

        # Generator
        with tf.variable_scope('generator_real', reuse=reuse):
            net = slim.fully_connected(input, 64*8*8, activation_fn=tf.nn.relu,
                                       weights_initializer=tflayers.xavier_initializer(),
                                       reuse=reuse)
            print(net)
            net = tf.reshape(net, [-1, 8, 8, 64])
            net = slim.conv2d(net, num_outputs=128, kernel_size=1, stride=1)
            print(net)

            if not test:
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=3, stride=2,
                                    weights_initializer=tflayers.xavier_initializer()):
                    with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu, is_training=(self.mode == 'train')):
                        net = slim.conv2d_transpose(net, num_outputs=128)
                        net = slim.batch_norm(net)
                        print(net)
                        net = slim.conv2d_transpose(net, num_outputs=256)  # output : 32 x 32
                        net = slim.batch_norm(net)
                        print(net)
                        net = slim.conv2d_transpose(net, num_outputs=512)  # output : 64 x 64
                        net = slim.batch_norm(net)
                        print(net)
                        net = slim.conv2d_transpose(net, num_outputs=256)  # output : 128 x 128
                        net = slim.batch_norm(net)
                        print(net)

                net = slim.conv2d_transpose(net, num_outputs=3, kernel_size=1, stride=1,
                                            weights_initializer=tflayers.xavier_initializer())
                print(net)

            else:
                with slim.arg_scope([slim.conv2d_transpose], kernel_size=3, stride=1, padding='SAME',
                                    weights_initializer=tflayers.xavier_initializer_conv2d()):
                    with slim.arg_scope([slim.batch_norm], activation_fn=tf.nn.relu):
                        # 2x Upsampling -> Conv2D
                        # Xavier -> Truncated normal
                        # 16 x 16
                        net = tf.image.resize_images(net, [16, 16])
                        net = slim.conv2d_transpose(net, num_outputs=128)
                        net = slim.batch_norm(net)
                        print(net)

                        # 32 x 32
                        net = tf.image.resize_images(net, [32, 32])
                        net = slim.conv2d_transpose(net, num_outputs=256)
                        net = slim.batch_norm(net)
                        print(net)

                        # 64 x 64
                        net = tf.image.resize_images(net, [64, 64])
                        net = slim.conv2d_transpose(net, num_outputs=512)
                        net = slim.batch_norm(net)
                        print(net)

                        # 128 x 128
                        net = tf.image.resize_images(net, [128, 128])
                        net = slim.conv2d_transpose(net, num_outputs=256)
                        net = slim.batch_norm(net)
                        print(net)

                net = slim.conv2d_transpose(net, num_outputs=1, kernel_size=1, stride=1, padding='SAME',
                                            activation_fn=tf.nn.tanh,
                                            weights_initializer=tflayers.xavier_initializer_conv2d())
            print(net)

            return net

    def discriminator_edge(self, input, reuse=False):
        net = input
        ind_feature = net
        print(net)

        with tf.variable_scope("discriminator_edge", reuse=reuse):
            # with tf.device('/gpu:0'):
            with slim.arg_scope([slim.conv2d], kernel_size=3, stride=2, activation_fn=tf.nn.relu,
                                normalizer_fn=tflayers.batch_norm,
                                weights_initializer=tf.random_normal_initializer(stddev=0.02)):
                # with slim.arg_scope([slim.batch_norm], activation_fn=leaky_relu, is_training=(self.mode=='train')):
                net = slim.conv2d(net, num_outputs=1024)     # 64 x 64
                # net = slim.batch_norm(net)
                print(net)
                net = slim.conv2d(net, num_outputs=512)     # 32 x 32
                # net = slim.batch_norm(net)
                print(net)
                net = slim.conv2d(net, num_outputs=256)     # 16 x 16
                ind_feature = net
                # net = slim.batch_norm(net)
                print(net)
                net = slim.conv2d(net, num_outputs=128)     # 8 x 8
                # net = slim.batch_norm(net)
                print(net)
                net = slim.conv2d(net, num_outputs=1)     # 4 x 4
                # net = slim.batch_norm(net)
                print(net)

            # LeakyReLU -> Sigmoid
            # net = slim.conv2d(net, num_outputs=1, kernel_size=4, stride=1, activation_fn=leaky_relu,
            #                   weights_initializer=tflayers.xavier_initializer())
            print(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=1, activation_fn=tf.nn.sigmoid,
                                       weights_initializer=tf.random_normal_initializer(stddev=0.02))
            print(net)

        return net, ind_feature

    def discriminator_real(self, input, reuse=False):
        net = input
        print(net)

        with tf.variable_scope("discriminator_real", reuse=reuse):
            with slim.arg_scope([slim.conv2d], kernel_size=3, stride=2,
                                weights_initializer=tflayers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], activation_fn=leaky_relu, is_training=(self.mode == 'train')):
                    net = slim.conv2d(net, num_outputs=256)  # 64 x 64
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=512)  # 32 x 32
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=256)  # 16 x 16
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=128)  # 8 x 8
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=64)  # 4 x 4
                    net = slim.batch_norm(net)
                    print(net)

            net = slim.conv2d(net, num_outputs=1, kernel_size=4, stride=1, activation_fn=tf.nn.sigmoid,
                              weights_initializer=tflayers.xavier_initializer())
            print(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=10, activation_fn=tf.nn.sigmoid,
                                       weights_initializer=tflayers.xavier_initializer())
            print(net)

        return net

    def cond_maker(self, input, reuse=False):
        temp_len = 30

        net = input
        net = tf.reshape(net, [-1, temp_len*16])
        print(net)          # Should be [?, 30 x 16]

        with tf.variable_scope("cond_maker", reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                weights_initializer=tflayers.xavier_initializer()):
                if self.mode == 'myo-lstm-test':
                    net = slim.fully_connected(net, num_outputs=temp_len*16)
                    net = slim.fully_connected(net, num_outputs=temp_len*12)
                    net = slim.fully_connected(net, num_outputs=temp_len*8)
                    net = slim.fully_connected(net, num_outputs=temp_len*4)
                    net = slim.fully_connected(net, num_outputs=80)
                elif self.mode == 'dual-learning':
                    net = slim.fully_connected(net, num_outputs=256 * 2)
                    net = slim.fully_connected(net, num_outputs=256 * 4)
                    net = slim.fully_connected(net, num_outputs=256 * 8)
                    net = slim.fully_connected(net, num_outputs=256 * 4)
                    net = slim.fully_connected(net, num_outputs=256 * 2)
                    net = slim.fully_connected(net, num_outputs=256)

        print(net.shape)
        return net

    def myo_reconstructor(self, input, reuse=False):
        # (8, 80) -> (8, 300, 16)
        # net = tf.reshape(input, [-1, 80])

        # [?, 256] -> [?, 30 x 16]

        temp_len = 30

        net = input
        print(net)

        with tf.variable_scope("myo_reconstructor", reuse=reuse):
            with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                                     weights_initializer=tflayers.xavier_initializer()):
                if self.mode == 'myo-lstm-test':
                    net = slim.fully_connected(net, num_outputs=temp_len)
                    net = slim.fully_connected(net, num_outputs=temp_len*2)
                    net = slim.fully_connected(net, num_outputs=temp_len*4)
                    net = slim.fully_connected(net, num_outputs=temp_len*8)
                    net = slim.fully_connected(net, num_outputs=temp_len*16)
                elif self.mode == 'dual-learning':
                    net = slim.fully_connected(net, num_outputs=30 * 16 * 1)
                    net = slim.fully_connected(net, num_outputs=30 * 16 * 2)
                    net = slim.fully_connected(net, num_outputs=30 * 16 * 4)
                    net = slim.fully_connected(net, num_outputs=30 * 16 * 8)
                    net = slim.fully_connected(net, num_outputs=30 * 16 * 4)
                    net = slim.fully_connected(net, num_outputs=30 * 16 * 2)
                    net = slim.fully_connected(net, num_outputs=30 * 16 * 1)

        print(net.shape)
        return net
        # net = slim.fully_connected(net)

    def classifier(self, input):
        print(input.shape)
        net = tflayers.fully_connected(input, num_outputs=128, activation_fn=tf.nn.relu)
        print(net.shape)
        net = tflayers.fully_connected(net, num_outputs=256, activation_fn=tf.nn.relu)
        print(net.shape)
        net = tflayers.fully_connected(net, num_outputs=128, activation_fn=tf.nn.relu)
        print(net.shape)
        net = tflayers.fully_connected(net, num_outputs=num_classes, activation_fn=tf.nn.relu)
        print(net.shape)

        # net = tf.reshape(net, [-1, 1, 9])

        return net

    def recons(self, input, reuse=False):
        net = input
        print(net)

        with tf.variable_scope("recons", reuse=reuse):
            with slim.arg_scope([slim.conv2d], kernel_size=3, stride=2,
                                weights_initializer=tf.glorot_normal_initializer()):
                with slim.arg_scope([slim.batch_norm], activation_fn=leaky_relu, is_training=(self.mode == 'train')):
                    net = slim.conv2d(net, num_outputs=128)  # 64 x 64
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=256)  # 32 x 32
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=128)  # 16 x 16
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=64)  # 8 x 8
                    net = slim.batch_norm(net)
                    print(net)
                    net = slim.conv2d(net, num_outputs=32)  # 4 x 4
                    net = slim.batch_norm(net)
                    print(net)

            # LeakyReLU -> Sigmoid
            # net = slim.conv2d(net, num_outputs=1, kernel_size=4, stride=1, activation_fn=leaky_relu,
            #                   weights_initializer=tflayers.xavier_initializer())
            # print(net)
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=8,
                                       weights_initializer=tf.glorot_normal_initializer())
            print(net)

        return net

    def make_condition(self, image):
        net = image
        with tf.variable_scope('make_condition'):
            # with tf.device('/gpu:0'):
            # make 64 x 64
            net = slim.conv2d(net, num_outputs=512, kernel_size=3, stride=2,
                              activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                              weights_initializer=tf.random_normal_initializer(0, 0.02))
            # make 32 x 32
            net = slim.conv2d(net, num_outputs=512, kernel_size=3, stride=2,
                              activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                              weights_initializer=tf.random_normal_initializer(0, 0.02))
            # make 16 x 16
            net = slim.conv2d(net, num_outputs=512, kernel_size=3, stride=2,
                              activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                              weights_initializer=tf.random_normal_initializer(0, 0.02))
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=800, activation_fn=tf.nn.relu, normalizer_fn=tflayers.batch_norm,
                                       weights_initializer=tflayers.xavier_initializer())

        return net


    def build(self):

        if self.is_real_image:
            self.real_image = tf.placeholder(tf.float32, [None, 128, 128, 3])
        else:
            self.real_image = tf.placeholder(tf.float32, [None, 128, 128, 1])

        # self.emg_data = tf.placeholder(tf.float32, [None, 30, 16])
        # self.emg_data = tf.placeholder(tf.float32, [None, 8])
        self.emg_data = tf.placeholder(tf.float32, [None, 100, 8])
        self.image_flatten = tf.placeholder(tf.float32, [None, 16*16])
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
                self.fake_logits, self.fake_feature = self.discriminator_edge(self.fake_image)
                self.real_logits, self.real_feature = self.discriminator_edge(self.real_image, reuse=True)

            # self.d_loss_fake = tf.losses.sigmoid_cross_entropy(tf.zeros_like(self.fake_logits), self.fake_logits)
            # self.d_loss_real = tf.losses.sigmoid_cross_entropy(tf.ones_like(self.real_logits), self.real_logits)
            self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.fake_logits), logits=self.fake_logits)
            self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.real_logits), logits=self.real_logits)
            self.d_loss = tf.reduce_mean(self.d_loss_fake + self.d_loss_real)

            # self.g_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(tf.ones_like(self.fake_logits), self.fake_logits))
            self.feature_matching_loss = tf.reduce_mean(tf.square(self.fake_image - self.real_image))
            self.g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.fake_logits), logits=self.fake_logits))

            self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss)
            self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss)
            self.f_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.feature_matching_loss)

        elif self.mode == 'rectest':
            emg_flatten = slim.flatten(self.emg_data)
            self.recon = self.make_condition(self.real_image)
            self.loss = tf.reduce_mean(tf.square(self.recon - emg_flatten))
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        elif self.mode == 'pretrain':
            print('Before one-hot', self.y_label.shape)
            self.y_onehot = tf.one_hot(self.y_label, self.label_num)
            print('After one-hot :', self.y_onehot)

            self.class_logits = self.classifier(self.emg_data)
            self.class_prediction = tf.argmax(self.class_logits, 1)
            print('class logits :', self.class_logits.shape)
            self.class_loss = tf.losses.softmax_cross_entropy(self.y_onehot, self.class_logits)
            self.class_acccuracy = tf.reduce_mean(tf.cast(tf.equal(self.class_prediction, self.y_label), tf.float32))

            self.class_trainer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.class_loss)

            # self.lstm_logits = self.lstm(self.emg_data)
            # self.lstm_prediction = tf.argmax(self.lstm_logits, 1)
            # print('lstm_logits :', self.lstm_logits.shape)
            # self.lstm_loss = tf.losses.softmax_cross_entropy(self.y_onehot, self.lstm_logits)
            # # self.lstm_prediction = tf.one_hot(self.lstm_logits, self.label_num)
            # # self.lstm_prediction = self.lstm_logits
            # print('lstm_prediction :', self.lstm_prediction.shape)
            # self.lstm_accuracy = tf.reduce_mean(tf.cast(tf.equal(self.lstm_prediction, self.y_label), tf.float32))

            # self.lstm_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.lstm_loss)

        elif self.mode == 'myo-lstm-test':
            self.x = slim.flatten(self.emg_data)
            self.fx = self.cond_maker(self.x)
            self.gfx = self.myo_reconstructor(self.fx)
            # self.flatten_gfx = tf.reshape(self.gfx, [-1, 30, 16])
            # self.fgfx = self.lstm(self.flatten_gfx, reuse=True)
            self.fgfx = self.cond_maker(self.gfx, reuse=True)

            print('flatter gfx :', self.gfx)

            self.lstm_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.x, predictions=self.gfx))
            self.recon_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.fx, predictions=self.fgfx))
            # self.total_loss = self.lstm_loss + self.recon_loss
            # self.lstm_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
            self.lstm_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.lstm_loss)
            self.recon_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.recon_loss)

        elif self.mode == 'dual-learning':
            self.x = slim.flatten(self.emg_data)
            self.y = self.image_flatten

            self.a_fx = self.cond_maker(self.x)
            self.a_gfx = self.myo_reconstructor(self.a_fx)

            self.a_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.x, predictions=self.a_gfx))

            self.b_gy = self.myo_reconstructor(self.y, reuse=True)
            self.b_fgy = self.cond_maker(self.b_gy, reuse=True)

            self.b_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=self.b_fgy))

            self.a_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.a_loss)
            self.b_trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.b_loss)
            pass