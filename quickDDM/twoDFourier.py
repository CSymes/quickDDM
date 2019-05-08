# -*- coding: utf-8 -*-
#twoDFourier.py
"""
Created on Wed Mar 27 12:28:43 2019
Takes in a list of 3d arrays of frames, and runs the 2d fourier transform on
each frame
As of hotfix01, all variants should be updated to have an fftshift
@author: Lionel
"""

"""
framesArray: a 3d array formatted as [frame order, y position, x position]
RETURN: [frameSquence, inverse y, inverse x]
"""
import numpy as np
def twoDFourier(framesArray):
    framesArray = np.fft.fftshift(np.fft.fft2(framesArray), axes = (1,2))
    return normaliseFourier(framesArray)

def normaliseFourier(frames):
    # Normalise the transform (based on size, move to real domain)
    scaling = (frames.shape[1] * frames.shape[2]) ^ 2
    normalised = np.square(np.absolute(frames))/scaling
    return normalised

    
"""
Attempt at decreasing memory demands by taking the average in the same action
as the absolute value, to never have a full complex array in memory
frames: a 3d array formatted as [frame order, y position, x position]
RETURN: [inverse y, inverse x], average normalised absolute value
"""
def cumulativeTransformAndAverage(frames):
    scaling = (frames.shape[1] * frames.shape[2]) ^ 2
    averages = np.zeros(frames.shape[1:3])#Same spatial shape, no time
    for i in range(0,frames.shape[0]):
        #Don't have to worry about the axes here, since there is no time here
        averages += np.square(np.absolute(np.fft.fftshift(np.fft.fft2(frames[i,:,:]))))
    #Taking the mean and normalising for size
    averages = (averages/scaling)/frames.shape[0]
    return averages

"""
This version havles memory requirements by using rfft. However, this changes
the dimensionality of the result, which will cause 
framesArray: a 3d array formatted as [frame order, y position, x position]
RETURN: [frameSquence, inverse y, inverse x]
Note that the inverse x axis is half the length of the original
Theory link:  https://stackoverflow.com/questions/52387673/what-is-the-difference-between-numpy-fft-fft-and-numpy-fft-rfft/52388007
"""
def realTwoDFourier(framesArray):
    
    scaling = (framesArray.shape[1] * framesArray.shape[2]) ^ 2
    #Can't use default normalise because it is now half size
    framesArray = np.fft.fftshift(np.fft.rfft2(framesArray), axes = (-2,))
    return np.square(np.absolute(framesArray))/scaling
