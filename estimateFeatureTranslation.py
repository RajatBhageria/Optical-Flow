'''
  File name: estimateFeatureTranslation.py
  Author: Rajiv Patel-O'Connor
  Date created: 11-20-2017
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from helper import rgb2gray
'''
  File clarification:
    Estimate the translation for single features 
    - Input startX: the x coordinate for single feature wrt the first frame
    - Input startY: the y coordinate for single feature wrt the first frame
    - Input Ix: the gradient along the x direction
    - Input Iy: the gradient along the y direction
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newX: the x coordinate for the feature wrt the second frame
    - Output newY: the y coordinate for the feature wrt the second frame
'''

def estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2):
  #TODO: Your code here
  
  #Convert from RGB to grayscale
  gray1 = rgb2gray(img1)
  gray2 = rgb2gray(img2)
  
  # Get size of image and allocate space for matrices in "culmination"
  [h, w] = gray1
  LHS_summation = np.zeros((2,2)) #make this an np.empty after shown to work
  RHS_summation = np.zeros((2, 1))
  
  # Get partial with respect to time
  It = gray2 - gray1
  
  # Calculate boundaries of window (normally 10x10 except when on boundary)
  windowMinX = startX - 4 if startX - 4 > 0 else startX
  windowMaxX = startX + 5 if startX + 5 < w else w - startX
  windowMinY = startY - 4 if startY - 4 > 0 else startY
  windowMaxY = startY + 5 if startY + 5 < h else h - startY
  
  # Get window points from Ix, Iy, and It
  Ix_window = Ix[windowMinY: windowMaxY, windowMinX: windowMaxX + 1]
  Iy_window = Iy[windowMinY: windowMaxY, windowMinX: windowMaxX]
  It_window = It[windowMinY: windowMaxY, windowMinX: windowMaxX]
  
  #Fill in culmination matrices
  LHS_summation[0][0] = np.sum(np.multiply(Ix_window, Ix_window))
  LHS_summation[0][1] = LHS_summation[1][0] = np.sum(np.multiply(Ix_window, Iy_window))
  LHS_summation[1][1] = np.sum(np.multiply(Iy_window, Iy_window))
  
  RHS_summation[0][0] = -1.0 * np.sum(np.multiply(Ix_window, It_window))
  RHS_summation[1][0] = -1.0 * np.sum(np.multiply(Iy_window, It_window))
  
  #Solve for newX and newY
  LHS_inverse = np.linalg.inv(LHS_summation)
  newX, newY = LHS_inverse.dot(RHS_summation).T[0]
  
  return newX, newY