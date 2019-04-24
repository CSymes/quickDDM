# -*- coding: utf-8 -*-
#readVideo.py
"""
Created on Tue Mar 26 14:43:01 2019
This class reads in a video and returns a 3d numpy array based on it containing
@author: Lionel
"""

import numpy as np
import cv2
import sys

"""
file: file path to read from, as string
RETURN: frames as (frame count, y position, x position)
"""

def readVideo(file):
    videoFile = cv2.VideoCapture(file)

    if not videoFile.isOpened():
        raise cv2.error("Malformed video file (" + file + ")")

    numFrames = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
    pxHeight = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
    pxWidth = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoArray = np.empty((numFrames,pxHeight,pxWidth),np.uint8)
    frameBuffer = np.empty((pxHeight, pxWidth))
    framesRead = 0
    readStatus = True
    while(framesRead < numFrames):
        #We only need one colour channel
        readStatus, frameBuffer = videoFile.read()
        if not readStatus:
            raise cv2.error('The video file has an error')
        videoArray[framesRead] = frameBuffer[:,:,0]
        framesRead += 1
    videoFile.release()
    return videoArray
