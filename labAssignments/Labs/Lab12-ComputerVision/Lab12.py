import numpy as np
import cv2
# Imports


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#set up with image
img = cv2.imread('img.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert image to gray scale
faces = face_cascade.detectMultiScale(gray, 1.3, 5)  

"""
 http://docs.opencv.org/3.0- beta/doc/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face- detection
"""
for (x,y,w,h) in faces: # Loop over each face detection and use its values
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #make a rectangle using the values
    detect_gray = gray[y:y+h, x:x+w]
    detect_colour = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(detect_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(detect_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) # Highlight detection on eyes
        
cv2.imshow('img',img)

    for image in range(0,2):
        imgFin[image] = img.copy()


for image in range(0,2):

    gray = cv2.cvtColor(img[image], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img[image],(x,y),(x+w,y+h),(255,0,0),2)
        detect_gray = gray[y:y+h, x:x+w]
        detect_colour = img2[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(detect_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(detect_colour,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow(image,img[image])        
cv2.waitKey(0)
cv2.destroyAllWindows() # Close when ready, good manners 
