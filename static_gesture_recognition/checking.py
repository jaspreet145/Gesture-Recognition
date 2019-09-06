# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:05:52 2019

@author: hp
"""


import cv2
import imutils
import numpy as np

from keras.models import load_model
path = r'my_model.h5'
model = load_model(path)


#main function
if __name__ == "__main__":
    
    # get the reference to the webcam
    camera = cv2.VideoCapture(0)

    # region of interest (ROI) coordinates
    top, right, bottom, left = 10, 350, 225, 590

    # keep looping, until interrupted
    while(True):
    # get the current frame
        (grabbed, frame) = camera.read()
        
        # flip the frame so that it is not the mirror view
        frame = cv2.flip(frame, 1)
        
        clone = frame.copy()
        # get the ROI
        roi = frame[top:bottom, right:left]
    
        # convert the roi to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dim = (28,28)
        gray = cv2.resize(gray , dim)
        gray = gray.reshape(28,28,1)
        label = model.predict(np.array([gray]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(clone,str(label.argmax()),(400,300), font, 4,(255,255,255),2,cv2.LINE_AA)
    
        # draw the roi
        cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
    
        # display the frame
        cv2.imshow("Video Feed", clone)

        # observe the keypress by the user
        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break
        # if the user pressed "q", then stop looping
    
# free up memory
camera.release()
cv2.destroyAllWindows()   
            