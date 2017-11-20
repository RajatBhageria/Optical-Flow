'''
  File name: getFeatures.py
  Author: Kashish Gupta, Rajat Bhageria, Rajiv Patel-Oconnor
  Date created: 11/20/17
'''
import numpy as np
import skimage
'''
  File clarification:
    Detect features within each detected bounding box
    - Input img: the first frame (in the grayscale) of video
    - Input bbox: the four corners of bounding boxes
    - Output x: the x coordinates of features
    - Output y: the y coordinates of features
'''

def getFeatures(img, bbox):
    #we only care about pixels that are within bounding boxes
    #the below for loop creates a new image called boxed_img that includes only the pixels in bounding boxes
    r,c = img.shape
    boxed_img = np.zeros(img.shape, np.uint8)
    for arr in bbox :
        x1 = arr[0,0]
        y1 = arr[0,1]
        x2 = arr[3,0]
        y2 = arr[3,1]
        boxed_img[y1:y2+1, x1:x2+1] = img[y1:y2+1, x1:x2+1]

    #now we do corner detection
    features_array = skimage.feature.corner_shi_tomasi(boxed_img, sigma=1)

    #suppress everything except for the top 1000 points
    features_sorted = np.sort(features_array, axis=None)
    thresh = features_sorted[-1000]
    features_array[features_array < thresh] = 0
    features_array[features_array > 0] = 1

    x,y = np.meshgrid(c,r)
    x = x[features_array]
    y = y[features_array]

    if x.size > 1000 :
        x = x[0:1000]
        y = y[0:1000]
    elif x.size < 1000 :
        x_pad = np.zeros([1000], np.int)
        y_pad = np.zeros([1000], np.int)
        x_pad[0 : x.size] = x
        y_pad[0 : y.size] = y
        x = x_pad
        y = y_pad

    return x, y