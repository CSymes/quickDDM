# -*- coding: utf-8 -*-
#frameDifferencer.py
"""
Created on Tue Mar 26 15:11:06 2019
Takes the differences between various frames, currently only between the first pair
@author: Lionel
"""
import numpy as np
"""
videoFrames: a real or complex 3d array, where the dimensions map to 
(frame number, y position, x position)
spacing: a single integer specifying the spacing between frames to subtract
RETURN: list(array(frame order, y position, x position))
"""
def frameDifferencer(videoFrames, spacing):
    #TODO: try doing this without any loops, just numpyness
    differences = videoFrames[spacing:videoFrames.shape[0],:,:] - videoFrames[0:videoFrames.shape[0] - spacing,:,:]
    return differences