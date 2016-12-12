import cv2
import numpy as np
from matplotlib import pyplot as plt
# Imports

"""
Open CV is a library of programming functions mainly aimed at real-time computer vision
In this lab I will show some of the features of OpenCv such as Edge Detection
"""

img = cv2.imread('GMIT.jpg',) # Take in Image

original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscalre with BGR

gray_3x3 = cv2.GaussianBlur(gray_img, (3, 3), 0)  # Convert image to blurry image 3x3
gray_13x13 = cv2.GaussianBlur(gray_img, (13, 13), 0)  # Convert image to blurry image 13x13

sobelHorizontal = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)  # x dir
sobelVertical = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5)  # y dir
sobelHV = sobelHorizontal + sobelVertical

edges = cv2.Canny(img, 100, 200)
edges2 = cv2.Canny(img, 100, 300)
#  Plot original image
plt.subplot(3, 3, 1), plt.imshow(original_img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
#  Plot grayscale image
plt.subplot(3, 3, 2), plt.imshow(gray_img, cmap = 'gray')
plt.title('GrayScale'), plt.xticks([]), plt.yticks([])
# Plot blurred 3x3
plt.subplot(3, 3, 3), plt.imshow(gray_3x3, cmap = 'gray')
plt.title('3x3 Blur'), plt.xticks([]), plt.yticks([])
# Plot blurred 13x13
plt.subplot(3, 3, 4), plt.imshow(gray_13x13, cmap = 'gray')
plt.title('13x13 Blur'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 5), plt.imshow(sobelHorizontal, cmap = 'gray')
plt.title('Sobel Horizontal'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 6), plt.imshow(sobelVertical, cmap = 'gray')
# Plot Sobel Horizontal
plt.title('Sobel Vertical'), plt.xticks([]), plt.yticks([])
# Plot Sobel Vertical
plt.subplot(3, 3, 7),plt.imshow(sobelHV, cmap = 'gray')
plt.title('Sobel H+V'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 8), plt.imshow(edges, cmap = 'gray')
# Plot canny detected image
plt.title('Canny Edge'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 3, 9), plt.imshow(edges2, cmap = 'gray')
plt.title('Custom Canny Edge'), plt.xticks([]), plt.yticks([])

# Show out plots
plt.show()
cv2.waitKey(0)
