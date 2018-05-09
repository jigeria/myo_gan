'''
        Author          : MagmaTart
        Last Modified   : 05/06/2018
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.preprocessing import normalize
import cv2

from load_data import DataLoader
from model import Model

print(tf.__version__)

mode = 'train'
is_real_image = False

loader = DataLoader(data_path='./dataset_2018_05_06/', is_real_image=is_real_image)

batch_size = 8
label_num = 9

model = Model(mode=mode, batch_size=batch_size, labels=label_num, learning_rate=0.0001, is_real_image=is_real_image)
model.build()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

total_acc = 0

if mode == 'pretrain':
    # sess.run(tf.global_variables_initializer())
    # restorer = tf.train.Saver()
    # restorer.restore(sess, './pretrain/iter6000.ckpt')

    for i in range(30001):
        emgs = loader.get_emg_datas(batch_size)
        _, labels = loader.get_images(batch_size)

        # for k in range(len(emgs)):
        #     emgs[k] = normalize(emgs[k])

        _, loss, acc = sess.run([model.lstm_trainer, model.lstm_loss, model.lstm_accuracy], feed_dict={model.emg_data:emgs, model.y_label:labels})
        print('Iteration', i, ' -', 'Loss :', loss, 'Acc :', acc)

        if i % 500 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './pretrain/' + 'iter' + str(i) + '.ckpt')

        # print('Iteration', i)

elif mode == 'pretrain-test':
    # ValueError: No variables to save
    restorer = tf.train.Saver()
    restorer.restore(sess, './pretrain/iter6000.ckpt')

    total = 0

    for i in range(100):
        print('Iteration', i)
        emgs = loader.get_emg_datas(batch_size)
        _, labels = loader.get_images(batch_size)
        acc = sess.run(model.lstm_accuracy, feed_dict={model.emg_data: emgs, model.y_label: labels})
        total += acc

    print('Average accuracy :', total/100)

elif mode == 'train':
    # restorer = tf.train.Saver()
    # restorer.restore(sess, './pretrain/iter6000.ckpt')

    for i in range(10000):
        print('Iteration ', i)
        emgs = loader.get_emg_datas(batch_size)
        images, labels = loader.get_images(batch_size)

        images = images / 127.5

        for k in range(len(emgs)):
            emgs[k] = normalize(emgs[k])

        z = np.random.rand(batch_size, 1000)

        _, _, ld, lg = sess.run([model.d_optimizer, model.g_optimizer, model.d_loss, model.g_loss], feed_dict={model.real_image:images, model.emg_data:emgs, model.z:z})
        print(ld, lg)

        test = sess.run(model.fake_image, feed_dict={model.emg_data:emgs, model.z:z})
        print(test.shape)
        cv2.imwrite('./samples/' + str(i) + '.png', test[0]*127.5)