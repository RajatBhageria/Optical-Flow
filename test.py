#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:57:25 2017

@author: rajivpatel-oconnor
"""

from detectFace import detectFace
from getFeatures import getFeatures
from estimateAllTranslation import estimateAllTranslation
from applyGeometricTransformation import applyGeometricTransformation
from helper import rgb2gray, overlay_points
import matplotlib.pyplot as plt
import cv2

#create all the images
color_img1 = plt.imread('./data/easy/TheMartian30.jpg')
img1 = rgb2gray(color_img1)

color_img2 = plt.imread('./data/easy/TheMartian31.jpg')
img2 = rgb2gray(color_img2)

#find the bounding boxes
bbox = detectFace(color_img1)

#find the positions of all the features
startXs, startYs = getFeatures(img1, bbox)
#overlay_points(img1, startXs, startYs, 'postGetFeatures_TheMartian1')

#draw the bounding boxes
first = bbox[0,0,:]
second = bbox[0,3,:]
cv2.rectangle(img1,(first[0],first[1]), (second[0],second[1]),color=(0,255,0))
plt.imshow(img1)

#estimate the translation
newXs, newYs = estimateAllTranslation(startXs, startYs, color_img1, color_img2)

Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)
#draw the bounding boxes
first = bbox[0,0,:]
second = bbox[0,3,:]
cv2.rectangle(img2,(first[0],first[1]), (second[0],second[1]),color=(0,255,0))
overlay_points(img2, Xs, Ys, 'postApplyGeomTransform_TheMartian31')


