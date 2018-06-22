'''
        Author          : MagmaTart
        Last Modified   : 05/06/2018
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from sklearn.preprocessing import normalize
import cv2
import matplotlib.pyplot as plt
import pywt

from load_data import DataLoader_Continous
from model import Model

print(tf.__version__)

mode = 'train'
is_real_image = False

loader = DataLoader_Continous(data_path='./dataset_0516/', emg_length=200, is_real_image=is_real_image)

batch_size = 4
label_num = 9

model = Model(mode=mode, batch_size=batch_size, labels=label_num, learning_rate=0.0002, is_real_image=is_real_image)
model.build()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

total_acc = 0

if mode == 'pretrain':
    # sess.run(tf.global_variables_initializer())
    # restorer = tf.train.Saver()
    # restorer.restore(sess, './pretrain/iter6000.ckpt')

    for i in range(30001):
        emgs = loader.get_emg_datas(15)
        print(emgs)
        # emgs = normalize(emgs)
        # print(emgs)
        labels = []
        _, label = loader.load_image()
        for k in range(15):
            # print(label)
            labels.append(label)

        labels = np.array(labels)

        _, loss, acc, pred, logit= sess.run([model.class_trainer, model.class_loss, model.class_acccuracy, model.class_prediction, model.class_logits], feed_dict={model.emg_data:emgs, model.y_label:labels})
        print('Iteration', i, ' -', 'Loss :', loss, 'Acc :', acc)
        # print('Logit :', logit)
        print('Labels :', labels)
        print('Prediction :', pred)

        if i % 500 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './pretrain/' + 'iter' + str(i) + '.ckpt')

        # print('Iteration', i)

elif mode == 'rectest':

    for i in range(100001):
        emgs = loader.get_emg_datas(batch_size)
        images = loader.get_images(batch_size)

        for k in range(len(emgs)):
            emgs[k] = normalize(emgs[k])
        print(emgs)

        # _, loss, pred, acc = sess.run([model.trainer, model.loss, model.prediction, model.accuracy], feed_dict={model.emg_data:emgs, model.y_label:labels})
        # print(i, loss)
        # print(labels)
        # print(pred)
        # print(acc)

        _, loss = sess.run([model.trainer, model.loss], feed_dict={model.emg_data: emgs, model.real_image: images})
        print(i, loss)

elif mode == 'data':
    # emgs = loader.get_emg_datas(batch_size)
    emgs = loader.load_emg_data()
    print(emgs.shape)
    cA, cD = pywt.dwt(emgs, 'db1')
    print(cA)
    print(cD)
    print(len(cA), len(cD))
    plt.plot(emgs)
    # plt.plot(cA)
    # plt.plot(cD)
    plt.show()

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
        # emgs = loader.get_emg_datas(batch_size)
        # images, labels = loader.get_images(batch_size)
        images = loader.get_images(batch_size)

        # images = images / 127.5

        # for k in range(len(emgs)):
        #     emgs[k] = normalize(emgs[k])

        # print(np.random.normal(0, 0.1, 1000).shape)
        z = np.array([np.random.normal(0, 0.1, 1000) for i in range(batch_size)])

        # _, _, ld, lg = sess.run([model.d_optimizer, model.g_optimizer, model.d_loss, model.g_loss], feed_dict={model.real_image:images, model.emg_data:emgs, model.z:z})
        _ = sess.run(model.g_optimizer, feed_dict={model.real_image: images, model.z: z})
        _, _, ld, lg, lf = sess.run([model.d_optimizer, model.g_optimizer, model.d_loss, model.g_loss, model.feature_matching_loss], feed_dict={model.real_image: images, model.z: z})

        print(ld, lg, lf)

        test = sess.run(model.fake_image, feed_dict={model.z:z})
        print(test.shape)
        cv2.imwrite('./samples/' + str(i) + '.png', test[0]*127.5)

elif mode == 'myo-lstm-test':
    # emgs = loader.get_emg_datas(batch_size)
    # print(emgs.shape)
    # v = sess.run(model.cond_vec, feed_dict={model.emg_data:emgs})
    # print(v.shape)

    # saver = tf.train.Saver()
    # saver.restore(sess=sess, save_path='./pretrain/myo-batch2-15000.ckpt')

    emgs = loader.get_emg_datas(2)

    image = loader.get_images(500)[263]
    image = cv2.resize(image, (16, 16))
    cv2.imshow('test', image)
    cv2.waitKey(100000)
    cv2.destroyAllWindows()

    # emgs = loader.get_emg_datas(2)
    # for k in range(len(emgs)):
    #     emgs[k] = normalize(emgs[k])
    # _, __, ll, rl = sess.run([model.lstm_trainer, model.recon_trainer, model.lstm_loss, model.recon_loss],
    #                          feed_dict={model.emg_data: emgs})
    # print(ll, rl)

    # print(emgs.shape)
    #

    '''
    emgs = loader.get_emg_datas(2)
    fx = sess.run(model.fx, feed_dict={model.emg_data:emgs})
    print(fx)
    emgs = loader.get_emg_datas(255)
    emgs = loader.get_emg_datas(2)
    fx = sess.run(model.fx, feed_dict={model.emg_data:emgs})
    print(fx)
    '''

    '''
    for i in range(100000):
        # print('Iteration ', i)
        emgs = loader.get_emg_datas(4)

        # print(emgs[0])
        for k in range(len(emgs)):
            emgs[k] = normalize(emgs[k])
        # print(emgs[0])

        # print('EMG :', emgs[0])

        _, __, ll, rl = sess.run([model.lstm_trainer, model.recon_trainer, model.lstm_loss, model.recon_loss], feed_dict={model.emg_data:emgs})

        if i % 10 == 0:
            print(i, ll, rl)

        # fx = sess.run(model.fx, feed_dict={model.emg_data:emgs})
        # print('FX :', fx)
        # print(fx.shape)

        if i % 1000 == 0:
            saver.save(sess=sess, save_path='./pretrain/myo-batch2-' + str(i) + '.ckpt')
    '''

elif mode == 'dual-learning':

    emgs = loader.get_emg_datas(6)

    for i in range(len(emgs)):
        plt.plot(emgs[i])
        plt.savefig('./samples/emg-PA' + str(i) + '.png')
        plt.close()

    # emgs = loader.get_emg_datas(12)
    # emgs = loader.get_emg_datas(6)
    #
    # for i in range(len(emgs)):
    #     plt.plot(emgs[i])
    #     plt.savefig('./samples/emg-S' + str(i) + '.png')
    #     plt.close()

    # plt.savefig()

'''
    lst = [1, 18, 30, 39, 48, 61, 91, 102]
    cur = 0

    saver = tf.train.Saver()
    saver.restore(sess=sess, save_path='./pretrain/dual255-20000.ckpt')

    for i in range(len(lst)):
        emgs = loader.get_emg_datas(20)
        emgs = loader.get_emg_datas(2)
        cur = lst[i]+2
        for k in range(len(emgs)):
            emgs[k] = normalize(emgs[k])

        res = sess.run(model.a_fx, feed_dict={model.emg_data:emgs})
        # print(res.shape)
        res = np.reshape(res[0], [16, 16, 1]) * 255.0
        res = np.array(res, dtype=np.uint8)
        # print(res.shape)
        # print(res)
        cv2.imwrite('./samples/dual' + str(i) + '.png', res)
        # cv2.imshow('test', res)
        # cv2.waitKey(100000)
        # cv2.destroyAllWindows()
'''

'''
    for i in range(100000):
        images_list = []

        images = loader.get_images(8)

        for k in range(len(images)):
            images_list.append(cv2.resize(images[k], (16, 16)))

        images = np.array(images_list)
        # print(image.shape)

        # for y in range(0, 64, 2):
        #     for x in range(0, 64, 2):
        #         # print(y, x, image[y][x], image[y+1][x])
        #         n = (int(image[y][x]) + int(image[y+1][x]) + int(image[y][x+1]) + int(image[y+1][x+1])) / 4
        #         # print('n :', n)
        #         new[int(y/2)].append(n)
        #
        # new = np.array(new)
        # print(new.shape)
        # print(new)

        # cv2.imshow('Test', new)
        # cv2.waitKey(100000)
        # cv2.destroyAllWindows()

        # print(images)

        images = np.reshape(images, [-1, 16*16]) / 255.0
        # print(images[0])
        
        emgs = loader.get_emg_datas(8)

        for k in range(len(emgs)):
            emgs[k] = normalize(emgs[k])

        # print(images.shape, emgs.shape)

        _, _, la, lb = sess.run([model.a_trainer, model.b_trainer, model.a_loss, model.b_loss], feed_dict={model.emg_data:emgs, model.image_flatten:images})

        print(i, la, lb)

        if i % 1000 == 0:
            saver.save(sess=sess, save_path='./pretrain/dual255-' + str(i) + '.ckpt')
    '''