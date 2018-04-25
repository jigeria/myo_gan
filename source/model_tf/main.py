import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize
import cv2
# TODO : python-opencv install

from load_data import DataLoader
from model import Model

loader = DataLoader(emg_data_path='./Sample_data/emg.csv', image_path='./Sample_data/hand_images/')

# images, emgs = loader.get_next_batch(3)
#
# print(type(images), images.shape)
# print(type(emgs), emgs.shape)

batch_size = 8

sess = tf.Session()

model = Model(batch_size=8)
model.build()

sess.run(tf.global_variables_initializer())

# print('Images shape :', images.shape)
# print('EMGs shape :', emgs.shape)
# print('Z vector shape :', z.shape)

emgs = loader.get_next_second_emgs(batch_size)
images = loader.get_next_images(batch_size)
images = images/127.5

for i in range(10000):
    print('Iteration ', i)
    emgs = loader.get_next_second_emgs(batch_size)
    images = loader.get_next_images(batch_size)

    for i in range(len(emgs)):
        emgs[i] = normalize(emgs[i])

    z = np.random.rand(batch_size, 20)

    _, _, ld, lg = sess.run([model.d_optimizer, model.g_optimizer, model.d_loss, model.g_loss], feed_dict={model.real_image:images, model.emg_data:emgs, model.z:z})
    print(ld, lg)

    test = sess.run(model.fake_image, feed_dict={model.emg_data:emgs, model.z:z})
    print(test.shape)
    # cv2.imshow('Test', test[0])
    # cv2.waitKey(100000)