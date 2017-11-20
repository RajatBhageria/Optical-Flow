'''
  File name: helper.py
  Author:
  Date created:
'''
import numpy as np
'''
  File clarification:
  Include any helper function you want for this project such as the 
  video frame extraction, video generation, drawing bounding box and so on.
'''

def rgb2gray(img):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])