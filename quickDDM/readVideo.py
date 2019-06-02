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
        raise OSError("Malformed video file (" + file + ")")

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
            raise OSError('The video file has an error')
        videoArray[framesRead] = np.array(frameBuffer[:,:,0])
        framesRead += 1
    videoFile.release()
    #Now we need to see if it should be cast to a square
    dimensionDifference = pxHeight - pxWidth
    if dimensionDifference > 0: #Height greater than width
        leftDrop = int(np.ceil(dimensionDifference / 2))
        rightDrop = int(np.floor(dimensionDifference / 2))
        if rightDrop == 0:
            videoArray = videoArray[:,leftDrop:,:]
        else:
            videoArray = videoArray[:,leftDrop:-rightDrop,:]

    if dimensionDifference < 0: #Width greater than height
        dimensionDifference = np.abs(dimensionDifference)
        leftDrop = int(np.ceil(dimensionDifference / 2))
        rightDrop = int(np.floor(dimensionDifference / 2))
        if rightDrop == 0:
            videoArray = videoArray[:,:,leftDrop:]
        else:
            #negative slice index drops the last n elements
            videoArray = videoArray[:,:,leftDrop:-rightDrop]
    return videoArray

"""
file: file path to read from, as string
RETURN: framerate as single float
"""
def readFramerate(file):
    videoFile = cv2.VideoCapture(file)

    if not videoFile.isOpened():
        raise OSError("Malformed video file (" + file + ")")

    # Successfully opened the video
    fps = int(videoFile.get(cv2.CAP_PROP_FPS))
    return fps
