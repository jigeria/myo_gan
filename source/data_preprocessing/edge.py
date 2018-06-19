'''

        Author          : MagmaTart
        Last Modified   : 05/01/2018

        __Under Construction__

'''

import numpy as np
import cv2

import os

def make_thick(edge, level=2):
    assert level != 1, 'Thick level 1 is not valid.'

    edge_thick = np.ndarray((550, 500))

    for i in range(len(edge)):
        for j in range(len(edge[i])):
            if i < 550 - (level - 1) and i > level - 2 and j < 500 - (level - 1) and j > level - 2:
                if edge[i][j] > 0:
                    edge_thick[i - (level - 1):i + level, j - (level - 1):j + level] = 255

    return edge_thick

#----- Blue background -----#

def get_hand(image):
    for i in range(len(image)):
        for j in range(len(image[i])):
            # if image[i][j][0] < 70 and image[i][j][1] < 70 and image[i][j][2] < 70:
            #     image[i][j] = [0, 0, 0]

            if image[i][j][0] > 150 and image[i][j][2] < 100:
                image[i][j] = [255, 255, 255]
            elif image[i][j][0] < 60 and image[i][j][1] < 60 and image[i][j][2] < 60:
                image[i][j] = [255, 255, 255]
            else:
                image[i][j] = [0, 0, 0]

    for i in range(100, 120):
        for j in range(100, 128):
            image[i][j] = [255, 255, 255]

    for i in range(120, 128):
        for j in range(85, 128):
            image[i][j] = [255, 255, 255]

for filenum in range(21, 22):
    thick = 5
    fps = 30

    cap = cv2.VideoCapture('./video/' + str(filenum) + '.mp4')

    index = 0

    hand_images = []
    if cap.isOpened():
        for i in range(1*fps):
            cap.read()

    while index < 30:
        if cap.isOpened():
            index += 1
            print(index)
            ret, frame = cap.read()
            hand_images.append(frame)
            for i in range(3*fps):
                cap.read()

    for i in range(len(hand_images)):
        hand_images[i] = np.array(hand_images[i])
        print(hand_images[i].shape, i)
        hand_images[i] = hand_images[i][100:650, 0:550, :]
        # print(hand_images[i][-1][0])
        # cv2.imshow('test', hand_images[i])
        # cv2.waitKey(500000)


        # hand_images[i] = make_thick(hand_images[i], level=5)

    for i in range(len(hand_images)):
        h, w = hand_images[i].shape[:2]
        center = (w/2, h/2)
        mat = cv2.getRotationMatrix2D(center, 90, 1.0)
        hand_images[i] = cv2.warpAffine(hand_images[i], mat, (w, h))

        print(i)
        hand_images[i] = cv2.resize(hand_images[i], (128, 128))

    if not os.path.exists('./' + str(filenum)):
        os.mkdir('./' + str(filenum))

    for i in range(len(hand_images)):
        # cv2.imshow('test', hand_images[i])
        # cv2.waitKey(500000)
        cv2.imwrite('./' + str(filenum) + '/hand-real' + str(i) + '.png', hand_images[i])
        get_hand(hand_images[i])
        # cv2.imshow('test', hand_images[i])
        # cv2.waitKey(500000)
        cv2.imwrite('./' + str(filenum) + '/hand-edge' + str(i) + '.png', hand_images[i])

# -------------------------------------------------------------------

    # def make_thick(edge, level=2):
#     assert level != 1, 'Thick level 1 is not valid.'
#
#     edge_thick = np.ndarray((550, 500))
#
#     for i in range(len(edge)):
#         for j in range(len(edge[i])):
#             if i < 550 - (level - 1) and i > level - 2 and j < 500 - (level - 1) and j > level - 2:
#                 if edge[i][j] > 0:
#                     edge_thick[i - (level - 1):i + level, j - (level - 1):j + level] = 255
#
#     return edge_thick
#

# ----- White background ----- #

# fps = 30
#
# cap = cv2.VideoCapture('./MAH06242.mp4')
#
# index = 0
#
# hand_images = []
# if cap.isOpened():
#     for i in range(1*fps):
#         cap.read()
#
# while index < 30:
#     if cap.isOpened():
#         index += 1
#         print(index)
#         ret, frame = cap.read()
#         # cv2.imshow('test', frame)
#         # cv2.waitKey(500000)
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#         # frame[:,:,0] -= 20
#         frame[:,:,2] += 60
#         # frame[:,:,1] += 50
#         frame = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
#         # cv2.imshow('test', frame)
#         # cv2.waitKey(500000)
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         # cv2.imshow('test', frame)
#         # cv2.waitKey(500000)
#         hand_images.append(frame)
#         for i in range(3*fps):
#             cap.read()
# #
# # # cv2.imshow('test', hand_images[0])
# # # cv2.waitKey(500000)
# #
# for i in range(len(hand_images)):
#     hand_images[i] = np.array(hand_images[i])
#     print(hand_images[i].shape, i)
#     hand_images[i] = hand_images[i][100:650, 0:800]
#     hand_images[i] = cv2.resize(hand_images[i], (500, 500))
#     # print(hand_images[i][-1][0])
#     # cv2.imshow('test', hand_images[i])
#     # cv2.waitKey(500000)
#
# # image = cv2.imread('./test2.jpg', cv2.IMREAD_GRAYSCALE)
# # image = cv2.resize(image, (500, 500))
# # # copy = image.copy()
# # print(image.shape)
# # cv2.imshow('test', image)
# # cv2.waitKey(500000)
# # image = cv2.Canny(image, 500, 30)
# # cv2.imshow('test', image)
# # cv2.waitKey(500000)
# # h, w = image.shape[:2]
# # mask = np.zeros((h + 2, w + 2), np.uint8)
# # cv2.floodFill(image, mask, (0, 0), 255)
# #
# # cv2.imshow('test', image)
# # cv2.waitKey(500000)
#
# # image = cv2.resize(image, (500, 500))
# #image = image[100:650, :]
#
# hand_real = []
#
# for i in range(len(hand_images)):
#     h, w = hand_images[i].shape[:2]
#     center = (w/2, h/2)
#     mat = cv2.getRotationMatrix2D(center, 90, 1.0)
#     hand_images[i] = cv2.warpAffine(hand_images[i], mat, (w, h))
#     hand_real.append(cv2.resize(hand_images[i], (128, 128)))
#     # print(len(hand_images[i][0]))
#     '''
#     # hand_images[i] = cv2.resize(hand_images[i], (800, 800))
#     # for k in range(len(hand_images[i][0])):
#     #     hand_images[i][-1][k] = 255
#     # hand_images[i][-1] = np.zeros((1, len(hand_images[0][0])), np.uint8)
#     '''
# for i in range(len(hand_images)):
#     # cv2.imshow('test', hand_images[i])
#     # cv2.waitKey(500000)
#     # hand_images[i] = cv2.Canny(hand_images[i], 100, 50) ***
#     # cv2.imshow('test', hand_images[i])
#     # cv2.waitKey(500000)
#     ret, thresh = cv2.threshold(hand_images[i], 168, 255, cv2.THRESH_TOZERO)
#     # cv2.imshow('test', thresh)
#     # cv2.imshow('test', cv2.Canny(thresh, 200, 50))
#     # cv2.waitKey(500000)
#     thresh = cv2.Canny(thresh, 100, 50)
#     # cv2.imshow('test', thresh)
#     # cv2.waitKey(500000)
#
#     # image, contours, hierachy = cv2.findContours(hand_images[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     image, contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     # cv2.imshow('test', contours)
#     # cv2.waitKey(500000)
#     hand_images[i] = cv2.drawContours(image, contours, -1, (255, 255, 255), thickness=2)
#     cv2.imwrite('./hands/edge' + str(i) + '.png', hand_images[i])
#     # cv2.fillPoly(hand_images[i], pts=contours, color=(255, 255, 255))
#     # cv2.floodFill(hand_images[i], mask, (0, 0), 255)
#     h, w = hand_images[i].shape[:2]
#     mask = np.zeros((h + 2, w + 2), np.uint8)
#     # cv2.floodFill(hand_images[i], mask, (0, 0), 255)
#     cv2.floodFill(hand_images[i], mask, (0, 0), 255)
#
#     hand_images[i] = cv2.resize(hand_images[i], (128, 128))
#
# for i in range(len(hand_images)):
#     h, w = hand_images[i].shape[:2]
#     center = (w/2, h/2)
#     mat = cv2.getRotationMatrix2D(center, 90, 1.0)
#     hand_images[i] = cv2.warpAffine(hand_images[i], mat, (w, h))
#     print(len(hand_images[i][0]))
#
# # h, w = hand_images[0].shape[:2]
#
#
#
#
#
#
#     # print(image.shape)
#
# assert len(hand_images) == len(hand_real), 'Array length not matching'
#
# if not os.path.exists('./' + str(4)):
#         os.mkdir('./' + str(4))
#
# for i in range(len(hand_images)):
#     # cv2.imshow('test', hand_images[i])
#     # cv2.waitKey(500000)
#     # get_hand(hand_images[i])
#     # cv2.imshow('test', hand_images[i])
#     # cv2.waitKey(500000)
#
#     cv2.imwrite('./' + str(4) + '/hand-real' + str(i) + '.png', hand_real[i])
#     cv2.imwrite('./' + str(4) + '/hand-edge' + str(i) + '.png', hand_images[i])


    # two = edge.copy()
# h, w = two.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)
# cv2.floodFill(two, mask, (0, 0), 255)
#
# edge_thick = make_thick(edge, level=5)
#edge = cv2.resize(edge, (64, 64))
#edge_thick = cv2.resize(edge_thick, (64, 64))
# cv2.imshow('test', two)
# cv2.waitKey(100000)
#cv2.imshow('test', edge_thick)
#cv2.waitKey(100000)