# -*- coding: utf-8 -*-
#frameDifferencer.py
"""
Created on Tue Mar 26 15:11:06 2019
Takes the differences between various frames, currently only between 
@author: Lionel
"""
import numpy as np
"""
videoFrames: a real or complex 3d array, where the dimensions map to 
(frame number, y position, x position)
spacings: a numpy array containing the different spacings to use
RETURN: list(array(frame order, y position, x position))
"""
def frameDifferencer(videoFrames, spacings):
    #TODO: try doing this without any loops, just numpyness
    differences = []
    for i in spacings:
        differences.append(videoFrames[i:videoFrames.shape[0],:,:] - videoFrames[0:videoFrames.shape[0] - i,:,:])
    return differences