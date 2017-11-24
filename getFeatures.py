'''
  File name: getFeatures.py
  Author: Kashish Gupta, Rajat Bhageria, Rajiv Patel-Oconnor
  Date created: 11/20/17
'''
import numpy as np
from skimage import feature
#from helper import anms, thresholdInBBox
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
    [numFaces, numCorners, coords] = bbox.shape
    xOutput = np.zeros((250, numFaces), dtype=np.int_)
    yOutput = np.zeros((250, numFaces), dtype=np.int_)
    count = 0
    for arr in bbox :
        x1 = arr[0,0]
        y1 = arr[0,1]
        x2 = arr[3,0]
        y2 = arr[3,1]
        boxed_img[y1:y2+1, x1:x2+1] = img[y1:y2+1, x1:x2+1]

        #now we do corner detection
        features_array = feature.corner_shi_tomasi(boxed_img, sigma=1)

        #suppress everything except for the top 1000 points
        features_sorted = np.sort(features_array, axis=None)
        thresh = features_sorted[-250]
        features_array[features_array < thresh] = 0
        features_array[features_array > 0] = 1
        features_array = features_array.astype(bool)
        
        x,y = np.meshgrid(range(c), range(r))
        x = x[features_array]
        y = y[features_array]
    
        if x.size > 250 :
            x = x[0:250]
            y = y[0:250]
        #we pad the array with 0's so that we always have 250 points of interest no matter what
        elif x.size < 250 :
            x_pad = np.zeros([250], np.int)
            y_pad = np.zeros([250], np.int)
            x_pad[0 : x.size] = x
            y_pad[0 : y.size] = y
            x = x_pad
            y = y_pad 
        xOutput[: , count] = x
        yOutput[ :, count] = y
        count += 1
        
    '''
    #automatic thresholding
    minX = bbox[0][0][0]
    maxX = bbox[0][1][0]
    minY = bbox[0][0][1]
    maxY = bbox[0][2][1]
    thresholded_features = thresholdInBBox(features_array, minX, maxX, minY, maxY)
    x, y, rmax = anms(thresholded_features, 100)
    '''
    x = xOutput
    y = yOutput

    return x, y