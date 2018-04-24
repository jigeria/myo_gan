import cv2

image = cv2.imread('./test5.jpg', cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (500, 800))
image = image[100:650, :]

edge1 = cv2.Canny(image, 70, 100)
print(image.shape)
cv2.imshow('test', edge1)
cv2.waitKey(100000)
