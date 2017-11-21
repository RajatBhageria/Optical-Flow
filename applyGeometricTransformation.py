import numpy as np
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
'''
  File name: applyGeometricTransformation.py
  Author:
  Date created:
'''

'''
  File clarification:
    Estimate the translation for bounding box
    - Input startXs: the x coordinates for all features wrt the first frame
    - Input startYs: the y coordinates for all features wrt the first frame
    - Input newXs: the x coordinates for all features wrt the second frame
    - Input newYs: the y coordinates for all features wrt the second frame
    - Input bbox: corner coordiantes of all detected bounding boxes
    
    - Output Xs: the x coordinates(after eliminating outliers) for all features wrt the second frame
    - Output Ys: the y coordinates(after eliminating outliers) for all features wrt the second frame
    - Output newbbox: corner coordiantes of all detected bounding boxes after transformation
'''

def applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox):
  #find the number of faces and number of features on each face
  [numFeatures, numFaces] = startXs.shape

  #instantiate the outputs
  Xs = []
  Ys = []
  newbbox = np.zeros((numFaces,4,2))

  #loop over the number of faces
  for face in range(0,numFaces):
    #find the distances between the points
    distances = np.linalg.norm((startXs[:,face] - newXs[:,face]) + (startYs[:,face] - newYs[:,face]))

    #set the maxDistance beyond which a feature is an outlier
    maxDistance = 4

    #Remove all the outlier points with distance between original and correspondance greater than maxDistance
    newXsWithoutOutliers = newXs[distances<maxDistance]
    newYsWithoutOutliers = newYs[distances<maxDistance]
    startXsWithoutOutliers = startXs[distances < maxDistance]
    startYsWithoutOutliers = startYs[distances < maxDistance]

    #get the current bounding box
    currentBbox = bbox[face, :, :]

    #find the similarity transform
    transform = SimilarityTransform.__init__()
    transformationWorked = transform.estimate([startXsWithoutOutliers,startYsWithoutOutliers],[newXsWithoutOutliers,newYsWithoutOutliers])

    #if the transformation was successful
    if (transformationWorked):
        #get the transformation matrix
        homoMatrix = transform.params
    else:
        homoMatrix = np.eye(3,3)

    #transform the image and add to newbbox
    currentNewBbox = matrix_transform(currentBbox,homoMatrix)
    newbbox[face,:,:] = currentNewBbox

    #add the new Xs and Ys to final Xs and Ys
    Xs[:,face] = newXsWithoutOutliers
    Ys[:,face] = newYsWithoutOutliers

  return Xs, Ys, newbbox