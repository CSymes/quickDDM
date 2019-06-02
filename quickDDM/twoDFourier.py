# -*- coding: utf-8 -*-
#twoDFourier.py
"""
Created on Wed Mar 27 12:28:43 2019
Takes in a list of 3d arrays of frames, and runs the 2d fourier transform on
each frame
As of hotfix01, all variants should be updated to have an fftshift
@author: Lionel
"""

import numpy as np

"""
Takes the 2-demensional Fourier transform of a list of (assumed square)
frames and returns the list of shifted and normalised outputs
Scales downwards in relation to frame size

framesArray: a 3d array formatted as [frame order, y position, x position]
RETURN: [frameSquence, inverse y, inverse x], complex
"""
def twoDFourierUnnormalized(framesArray):
    #Since the scaling is applied before the squaring, take the square root
    scaling = np.sqrt((framesArray.shape[-2] * framesArray.shape[-1]))
    #Can't use default normalise because it is now half size
    framesArray = np.fft.fftshift(np.fft.fft2(framesArray), axes = (-2,-1))
    return framesArray/scaling

"""
This version handles memory requirements by using rfft. However, this changes
the dimensionality of the result, which means that the "real" versions of
calculate q curves must be used as well after this. Also changes fftshift needs

framesArray: a 3d array formatted as [frame order, y position, x position]
RETURN: [frameSquence, inverse y, inverse x]

Note that the inverse x axis is half the length of the original
Theory link:  https://stackoverflow.com/questions/52387673/what-is-the-difference-between-numpy-fft-fft-and-numpy-fft-rfft/52388007
"""
def realTwoDFourierUnnormalized(framesArray):
    #Since the scaling is applied before the squaring, take the square root
    scaling = np.sqrt((framesArray.shape[-2] * framesArray.shape[-1]))
    #Can't use default normalise because it is now half size
    framesArray = np.fft.fftshift(np.fft.rfft2(framesArray), axes = (-2,))
    return framesArray/scaling

"""
Takes the real magnitude of a complex frame

framesArray: 2d array, [y position, x position], complex
RETURN: [y position, x position], float
"""
def castToReal(framesArray):
    return np.square(np.absolute(framesArray))
