'''
  File name: faceTracking.py
  Author: Kashish Gupta, Rajat Bhageria, Rajiv Patel-Oconnor
  Date created: 11/19/17
'''
import numpy as np
import cv2
import detectFace

'''
  File clarification:
    Generate a video with tracking features and bounding box for face regions
    - Input rawVideo: the video contains one or more faces
    - Output trackedVideo: the generated video with tracked features and bounding box for face regions
'''

def faceTracking(rawVideo):
    
    #process the video here, convert into frames of images
    cap = cv2.VideoCapture(rawVideo)
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
    while ~face_found:
        face = detectFace(frames[f,:,:,:]);
        f+=1
        if face :
            face_found = True
            
    #step 2 and 3 TODO:
    
    return trackedVideo