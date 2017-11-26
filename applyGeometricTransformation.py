import numpy as np
from skimage.transform import SimilarityTransform
from skimage.transform import matrix_transform
'''
  File name: applyGeometricTransformation.py
  Author: Rajat Bhageria, Rajiv O'Conner - Patel 
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
    #distances = np.linalg.norm((startXs[:,face] - newXs[:,face]) + (startYs[:,face] - newYs[:,face]))
    distances = ((startXs[:,face] - newXs[:,face])**2 + (startYs[:,face] - newYs[:,face])**2)**.5

    #set the maxDistance beyond which a feature is an outlier
    maxDistance = 4

    #Remove all the outlier points with distance between original and correspondance greater than maxDistance
    newXofFace= newXs[:,face]
    newXsWithoutOutliers = newXofFace[distances < maxDistance]
    newYofFace = newYs[:,face]
    newYsWithoutOutliers = newYofFace[distances < maxDistance]
    startXofFace = startXs[:,face]
    startXsWithoutOutliers = startXofFace[distances < maxDistance]
    startYofFace = startYs[:, face]
    startYsWithoutOutliers = startYofFace[distances < maxDistance]

    #get the current bounding box
    currentBbox = bbox[face, :, :]

    #find the similarity transform
    transform = SimilarityTransform()
    src = np.column_stack((startXsWithoutOutliers,startYsWithoutOutliers))
    dest = np.column_stack((newXsWithoutOutliers,newYsWithoutOutliers))
    transformationWorked = transform.estimate(src,dest)

    currentNewBbox = []

    #if the transformation was successful
    if (transformationWorked):
        #get the transformation matrix
        homoMatrix = transform.params
        #do the transform
        if (homoMatrix.shape==(3,3)):
            currentNewBbox = matrix_transform(currentBbox, homoMatrix)
    else:
        #set the old bbox to the current one
        currentNewBbox = currentBbox

    #add to newbbox
    newbbox[face,:,:] = currentNewBbox

    #HERE we need to modify Xs and Ys based on the newbbox that we just found
    #need to do
    #Xs = Xs[Xs > xstart and Xs < xend]
    #Ys = Ys[Xs > xstart and Xs < xend]
    #Xs = Xs[Ys > ystart and Ys < yend]
    #Ys = Ys[Ys > ystart and Ys < yend]
    #all 4 of these need to be done in order to delete Xs and Ys that are out of bounds, but I wasnt
    #sure how to access the start and end boundaries from bbox nor how to index into Xs and Ys if there are multiple faces

    #add the new Xs and Ys to final Xs and Ys
    #only once face or the first face
    if (len(Xs) ==0):
        Xs = newXsWithoutOutliers
    else: #multiple faces
        np.append(Xs,newXsWithoutOutliers)

    # only once face or the first face
    if (len(Ys) == 0):
        Ys = newYsWithoutOutliers
    else: #multiple faces
        np.append(Ys,newYsWithoutOutliers)

  return np.asarray(Xs), np.asarray(Ys), newbbox