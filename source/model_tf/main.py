import tensorflow as tf
import numpy as np
import cv2
# TODO : python-opencv install

from load_data import DataLoader
from model import Model

loader = DataLoader(emg_data_path='./Sample_data/time_match.csv', image_path='./Sample_data/hand_images/')

# images, emgs = loader.get_next_batch(3)
#
# print(type(images), images.shape)
# print(type(emgs), emgs.shape)

sess = tf.Session()

model = Model()
model.build()

sess.run(tf.global_variables_initializer())

for i in range(10000):
    image, emg = loader.get_next_batch(1)
    z = np.random.rand(1, 80)
    c = np.random.rand(1, 20)
    # TODO : Check emg data shape

    print(z.shape, c.shape)

    _, _, ld, lg = sess.run([model.d_optimizer, model.g_optimizer, model.d_loss, model.g_loss], feed_dict={model.real_image:image, model.z:z, model.c:c})
    print(ld, lg)