#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:57:25 2017

@author: rajivpatel-oconnor
"""

from detectFace import detectFace
from getFeatures import getFeatures
from applyGeometricTransform import applyGeometricTransform
from helpers import rgb2gray, overlay_points, videoToFrames
import matplotlib.pyplot as plt
import numpy as np


color_img1 = plt.imread('./data/easy/the_martian.jpg')
img = rgb2gray(color_img)

bbox = detectFace(color_img)

startXs, startYs = getFeatures(img, bbox)

newXs, newYs = estimateAllTranslation(startXs, startYs, color_img1)
