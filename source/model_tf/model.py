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

# emg : (1, 300, 16)

class Model:
    def __init__(self, mode='train', batch_size=1, learning_rate=0.0002):
        self.mode = mode
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def lstm(self, emg_data):
        print(emg_data.shape)
        cell = rnn.BasicLSTMCell(num_units=256, state_is_tuple=True)
        initial_state = cell.zero_state(self.batch_size, tf.float32)
        outputs, _states = tf.nn.dynamic_rnn(cell, emg_data, initial_state=initial_state, dtype=np.float32)

        # 300줄이 순차적으로 RNN에 들어가고, 최종 출력은 outputs의 마지막 row임
        print(outputs[:, -1].shape)
        # outputs = tf.reshape(outputs, [-1, 256])
        outputs = outputs[:, -1]
        print(outputs.shape)
        outputs = tflayers.fully_connected(inputs=outputs, num_outputs=80, activation_fn=None)
        return outputs

    def generator(self, emg_data, z, reuse=False):
        # (None, 300, 16) -> (1, 100)
        input = self.lstm(emg_data)
        print(input.shape)
        input = tf.concat([input, z],  1)
        print(input.shape)

        # Generator
        net = slim.fully_connected(input, 256, activation_fn=tf.nn.relu,
                                      weights_initializer=tflayers.xavier_initializer(),
                                      biases_initializer=tflayers.xavier_initializer(),
                                      reuse=reuse)
        print(net)
        net = tf.reshape(net, [-1, 16, 16, 1])
        net = slim.conv2d(net, num_outputs=128, kernel_size=1, stride=1)
        print(net)
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope([slim.conv2d_transpose], padding='SAME', kernel_size=2, stride=2,
                                weights_initializer=tflayers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.relu, is_training=(self.mode=='train')):
                    net = slim.conv2d_transpose(net, num_outputs=256)       # output : 32 x 32
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d_transpose(net, num_outputs=512)       # output : 64 x 64
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d_transpose(net, num_outputs=256)       # output : 128 x 128
                    print(net)
                    net = slim.batch_norm(net)

            net = slim.conv2d_transpose(net, num_outputs=1, kernel_size=1, stride=1, padding='VALID',
                                        weights_initializer=tflayers.xavier_initializer())
            print(net)

        return net

    def discriminator(self, input, reuse=False):
        net = input
        print(net)

        with tf.variable_scope("discriminator", reuse=reuse):
            with slim.arg_scope([slim.conv2d], kernel_size=2, stride=2, padding='SAME',
                                weights_initializer=tflayers.xavier_initializer()):
                with slim.arg_scope([slim.batch_norm], decay=0.95, center=True, scale=True,
                                    activation_fn=tf.nn.leaky_relu, is_training=(self.mode=='train')):
                    net = slim.conv2d(net, num_outputs=256)     # 64 x 64
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=512)     # 32 x 32
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=256)     # 16 x 16
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=128)     # 8 x 8
                    print(net)
                    net = slim.batch_norm(net)
                    net = slim.conv2d(net, num_outputs=128)     # 4 x 4
                    print(net)
                    net = slim.batch_norm(net)

            net = slim.conv2d(net, num_outputs=1, kernel_size=4, stride=1, activation_fn=tf.nn.leaky_relu,
                              weights_initializer=tflayers.xavier_initializer())
            print(net)
            net = slim.flatten(net)
            print(net)

        return net

    def build(self):
        self.real_image = tf.placeholder(tf.float32, [None, 128, 128, 1])
        self.emg_data = tf.placeholder(tf.float32, [None, 300, 16])
        self.z = tf.placeholder(tf.float32, [None, 20])
        # self.c = tf.placeholder(tf.float32, [None, 20])

        # self.fake_image = self.generator(self.z, self.c)
        self.fake_image = self.generator(self.emg_data, self.z, reuse=False)
        self.fake_logits = self.discriminator(self.fake_image)
        self.real_logits = self.discriminator(self.real_image, reuse=True)

        self.d_loss_fake = tf.losses.sigmoid_cross_entropy(self.fake_logits, tf.zeros_like(self.fake_logits))
        self.d_loss_real = tf.losses.sigmoid_cross_entropy(self.real_logits, tf.ones_like(self.real_logits))
        self.d_loss = tf.reduce_mean(self.d_loss_fake + self.d_loss_real)

        self.g_loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(self.fake_logits, tf.ones_like(self.fake_logits)))

        self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.d_loss)
        self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.g_loss)