import cv2
import copy
import numpy as np
from matplotlib import pyplot as plt
# Imports


"""
In this lab we will show some other variant of openCV functionality.
Harris and Shi Tomasi feature matching
"""

img = cv2.imread('GMIT1.jpg',)

original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

dst = cv2.cornerHarris(gray_img, 2, 3, 0.04)
imgHarris = img.copy() # make a copy of img for our shiTomasi detection

corners = cv2.goodFeaturesToTrack(gray_img, 20, 0.01, 50)
imgShiTomasi = img.copy() # make a copy of img for our shiTomasi detection

threshold = 0.03; #number between 0 and 1
for i in range(len(dst)):
	for j in range(len(dst[i])):
		if dst[i][j] > (threshold*dst.max()):
			cv2.circle(imgHarris,(j,i), 10, (115, 0, 25), 5)

for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShiTomasi,(x,y),10,(255, 255, 0), 5)

# Initiate ORB-SIFT dectection variable
orb = cv2.ORB()
# find the keypoints and descriptors with ORB-SIFT detection
kp1, des1 = orb.detectAndCompute(gray_img,None)
# draw only keypoints location,not size and orientation
imgOrb = cv2.drawKeypoints(img,kp1,color=(255, 255, 0)) 
# Read in second image
customImg = cv2.imread('GMIT2.jpg',)

original_CustomImg = cv2.cvtColor(customImg, cv2.COLOR_BGR2RGB)	
gray_customImg = cv2.cvtColor(customImg, cv2.COLOR_BGR2GRAY) #Convert to grayscale

dst2 = cv2.cornerHarris(gray_customImg, 2, 3, 0.04)
imgHarris2 = customImg.copy() # make a copy of img for our shiTomasi detection

corners2 = cv2.goodFeaturesToTrack(gray_customImg, 20, 0.01, 50)
imgShiTomasi2 = customImg.copy() # make a copy of img for our shiTomasi detection

threshold = 0.03; #number between 0 and 1
for i in range(len(dst2)):
	for j in range(len(dst2[i])):
		if dst2[i][j] > (threshold*dst2.max()):
			cv2.circle(imgHarris2,(j,i), 10, (115, 0, 25), 5)

for i in corners:
	x,y = i.ravel()
	cv2.circle(imgShiTomasi2,(x,y),10,(255, 255, 0), 5)

# find the keypoints and descriptors with ORB-SIFT
kp2, des2 = orb.detectAndCompute(gray_customImg,None)
# draw only keypoints location,not size and orientation
imgOrb2 = cv2.drawKeypoints(customImg,kp2,color=(255, 255, 0))

"""
Plot our imgs and detection performed images from different algorithms 
"""

	
plt.subplot(3, 2, 1),plt.imshow(cv2.cvtColor(imgHarris, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Harris Corners'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 2),plt.imshow(cv2.cvtColor(imgHarris2, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Harris Corners'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 3),plt.imshow(cv2.cvtColor(imgShiTomasi, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('GFTT Corners'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 4),plt.imshow(cv2.cvtColor(imgShiTomasi2, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('GFTT Corners'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 5),plt.imshow(cv2.cvtColor(imgOrb, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('ORB-SIFT Corners'), plt.xticks([]), plt.yticks([])

plt.subplot(3, 2, 6),plt.imshow(cv2.cvtColor(imgOrb2, cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('ORB-SIFT Corners'), plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0); 