'''
  File name: detectFace.py
  Author: Kashish Gupta, Rajat Bhageria, Rajiv Patel-Oconnor
  Date created: 11/19/17
'''
import numpy as np
import cv2

'''
  File clarification:
    Detect or hand-label bounding box for all face regions
    - Input img: the first frame of video
    - Output bbox: the four corners of bounding boxes for all detected faces
'''

def detectFace(img):
    #TODO: Your code here
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

    #returns (x, y, w, h) for each face in the pic
    faces = faceCascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                       flags = cv2.CASCADE_SCALE_IMAGE)
    
    #If we want to detect side-profile faces, aka faces that are not facing forward
    #then uncomment the below lines
    #faceCascade_side = cv2.CascadeClassifier('haarcascade_profileface.xml')
    #faces_side = faceCascade_side.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
    #                                   flags = cv2.CASCADE_SCALE_IMAGE)
    #faces = faces + faces_side
    

    #if we want to visualize the faces
    #for (x, y, w, h) in faces:
    #    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow('image', img)
    #cv2.waitKey(0)


    #parse this output into the F x 4 x 2 matrix that we want
    bbox = np.zeros([np.size(faces,0),4,2], np.int)
    f = 0
    for (x, y, w, h) in faces:
        bbox[f, 0, 0] = x
        bbox[f, 0, 1] = y

        bbox[f, 1, 0] = x + w
        bbox[f, 1, 1] = y

        bbox[f, 2, 0] = x
        bbox[f, 2, 1] = y + h

        bbox[f, 3, 0] = x + w
        bbox[f, 3, 1] = y + h

        f+=1

    if np.array_equal(bbox, np.zeros([np.size(faces,0),4,2])) :
        return None
    else :
        return bbox