from estimateFeatureTranslation import estimateFeatureTranslation
import numpy as np
import cv2
from helper import rgb2gray

'''
  File name: estimateAllTranslation.py
  Author: Rajat Bhageria, Rajiv Patel-O'Connor
  Date created:
'''

'''
  File clarification:
    Estimate the translation for all features for each bounding box as well as its four corners
    - Input startXs: all x coordinates for features wrt the first frame
    - Input startYs: all y coordinates for features wrt the first frame
    - Input img1: the first image frame
    - Input img2: the second image frame
    - Output newXs: all x coordinates for features wrt the second frame
    - Output newYs: all y coordinates for features wrt the second frame
'''

def estimateAllTranslation(startXs, startYs, img1, img2):

  #Convert from RGB to grayscale
  grey1 = rgb2gray(img1)

  #find the number of total features
  [numFeatures, numFaces] = startXs.shape

  #instantiate the output
  newXs = np.zeros((numFeatures,numFaces))
  newYs = np.zeros((numFeatures,numFaces))

  #Get the gradients of the first image
  #[Ix, Iy] = np.gradient(grey1)
  Ix = cv2.Sobel(grey1, cv2.CV_64F, 1, 0, ksize=5)
  Iy = cv2.Sobel(grey1, cv2.CV_64F, 0, 1, ksize=5)

  #Loop through all the faces
  for face in range(0, numFaces):
      #Loop through all the features
      for feature in range(0, numFeatures):

          #get the current feature for the current face
          startX = startXs[feature, face]
          startY = startYs[feature, face]

          # Estimate the translation for the current feature
          [newX, newY]=estimateFeatureTranslation(startX, startY, Ix, Iy, img1, img2)

          #append the newX and newY for the current feature to the list of new featuers
          newXs[feature, face] = newX
          newYs[feature, face] = newY

  return newXs, newYs