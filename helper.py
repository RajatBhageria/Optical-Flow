'''
  File name: helper.py
  Author:
  Date created:
'''
import numpy as np
import matplotlib.pyplot as plt
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