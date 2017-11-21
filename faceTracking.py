'''
  File name: faceTracking.py
  Author: Kashish Gupta, Rajat Bhageria, Rajiv Patel-Oconnor
  Date created: 11/19/17
'''
import numpy as np
import cv2
import detectFace
import getFeatures
import estimateAllTranslation
import applyGeometricTransformation

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces
    - Output trackedVideo: the generated video with tracked features and bounding box for face regions
'''

def faceTracking(rawVideo):
    
    #process the video here, convert into frames of images
    #assuming that 'rawVideo' is a video path
    cap = cv2.VideoCapture(rawVideo)
    #ie, test with rawVideo = 'Data/Easy/TheMartian.mp4'
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    num_frames = int(cap.get(7))
    #create an array that holds all the frames in the video, format: frame_number x width x height
    frames = np.array([num_frames, frame_width, frame_height, 3])
    f = 0

    while(cap.isOpened()):
        ret, frame = cap.read()
        #cv2.imshow('fr ame')
        frames[f,:,:,:]= frame
        f+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    #need to detect face only on the first frame of the video
    #if "good" face not found, then try other frames, until a good face is found
    face_found = False
    f = 0
    face = None
    while ~face_found:
        #face is a Fx4x2 bounding box
        face = detectFace(frames[f,:,:,:]);
        f+=1
        if face :
            face_found = True
            f-=1

    #find the start features
    init_frame = f
    init_img = frames[init_frame,:,:,:]
    init_img_gray = cv2.cvtColor(init_img, cv2.COLOR_BGR2GRAY)
    #there will always be 1000 xy's, because we have padded with (0,0)
    #make srue to ignore the (0,0) points later in the code
    startXs, startYs = getFeatures(init_img_gray, face)

    ### STILL LOT TO DO FROM HERE ON!

    #cv2.rectangle can be used to draw rectangles if needed

    #initialize the the output matrix of tracked images
    outputMatrix = np.zeros(num_frames-f,frame_width,frame_height)

    #draw rectangles of all the faces on the current image
    initImgWithBBox = init_img
    [numFaces,_,_] = face.shape
    for face in range(0,numFaces):
        bboxOfCurrFace = face[face,:,:]
        initImgWithBBox = cv2.rect(initImgWithBBox,bboxOfCurrFace[0,:],bboxOfCurrFace[3,:])

    #add the initial image as the first image
    outputMatrix[0,:,:,:] = initImgWithBBox

    #actually do the transform and find the new bounding box
    for frame in range(f,num_frames-1): #this should probably not be -1
        img1 = frames[frame,:,:,:]
        img2 = frames[frame+1,:,:,:]

        [newXs, newYs] = estimateAllTranslation(startXs, startYs, img1, img2)
        [Xs, Ys, newbbox] = applyGeometricTransformation(startXs, startYs, newXs, newYs, face)

        #now add a rectangle of newbbox to img2
        img2WithBoundingBox = img2
        for face in range(0, numFaces):
            bboxOfCurrFace = newbbox[face, :, :]
            img2WithBoundingBox = cv2.rect(img2WithBoundingBox, bboxOfCurrFace[0, :], bboxOfCurrFace[3, :])

        #add img2 to the output matrix
        outputMatrix[frame,:,:,:] = img2WithBoundingBox

    #convert outputMatrix to a video and return as trackedVideo

    return trackedVideo