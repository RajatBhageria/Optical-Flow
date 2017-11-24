'''
  File name: helper.py
  Author:
  Date created:
'''
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage import feature
import cv2
'''
  File clarification:
  Include any helper function you want for this project such as the 
  video frame extraction, video generation, drawing bounding box and so on.
'''

def rgb2gray(img):
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

def overlay_points(img, x, y, name):
    plt.figure()
    implot = plt.imshow(img)
    plt.scatter(x, y,color='red',marker='o', s=1)
    plt.savefig(name)
    plt.close("all")

def plotPoints(img, x, y):
    x = x.astype(int)
    y = y.astype(int)
    img[x, y] = [0, 255, 0]
    return img
    
def thresholdInBBox(features_array, minX, maxX, minY, maxY):
    cimg = features_array[minY: maxY + 1, minX: maxX + 1]
    mu = np.mean(cimg)
    s=np.std(cimg)
    threshold = mu+4*s
    # threshold image
    for i in range(minY, maxY + 1):
        for j in range(minX, maxX + 1):
            if features_array[i][j] < threshold:
                features_array[i][j] = 0.0
    return features_array
    
def distance(y1, x1, y2, x2):
    return math.sqrt((x2-x1)**2 + (y2- y1)**2)

def anms(cimg, max_pts):
    nonzero = np.nonzero(cimg)
    nz_rows = nonzero[0]
    nz_cols = nonzero[1]
    num_features = len(nonzero[0])
    threshold = 1
    store = np.zeros((num_features, 3))
    #implement a data structure to speed this up
    for i in range(num_features):
        minRadius = 100000
        comparativeIntensity = cimg[nz_rows[i]][nz_cols[i]]
        for j in range(num_features):
            if i == j:
                continue
            if cimg[nz_rows[j]][nz_cols[j]] >= comparativeIntensity*threshold:
                currRadius = distance(nz_rows[j], nz_cols[j], nz_rows[i], nz_cols[i])
                if currRadius < minRadius:
                    minRadius = currRadius
        store[i][0] = nz_rows[i]
        store[i][1] = nz_cols[i]
        store[i][2] = minRadius
        
    store = store[store[:, 2].argsort()][::-1]
    
    if max_pts > num_features:
        y = store[:, 0]
        x = store[:, 1]
        rmax = np.min(store[:,2])
        return x, y, rmax
    
    y = store[:, 0][:max_pts].astype(int)
    x = store[:, 1][:max_pts].astype(int)
    rmax  = store[:,2][max_pts + 1]
          
    return x, y, rmax

def videoToFrames(filepath):
    vidcap = cv2.VideoCapture(filepath)
    success,image = vidcap.read()
    count = 0
    success = True
    while success:
        success,image = vidcap.read()
        print('Read a new frame: ', success)
        cv2.imwrite("./data/medium/StrangerThings%d.jpg" % count, image)     # save frame as JPEG file
        count += 1