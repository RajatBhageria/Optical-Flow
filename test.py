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

color_img1 = plt.imread('./data/easy/TheMartian1.jpg')
img1 = rgb2gray(color_img1)

color_img2 = plt.imread('./data/easy/TheMartian2.jpg')
img2 = rgb2gray(color_img2)

bbox = detectFace(color_img1)

startXs, startYs = getFeatures(img1, bbox)
overlay_points(img1, startXs, startYs, 'postGetFeatures_TheMartian1')

newXs, newYs = estimateAllTranslation(startXs, startYs, color_img1, color_img2)

Xs, Ys, newbbox = applyGeometricTransformation(startXs, startYs, newXs, newYs, bbox)

overlay_points(img2, Xs, Ys, 'postApplyGeomTransform_TheMartian2')


