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
RETURN: [frameSquence, inverse y, inverse x], complex
"""
import numpy as np
def twoDFourier(framesArray):
    #TODO: it is possible we can halve the size of the complex data by using rfft2
    #link: https://stackoverflow.com/questions/52387673/what-is-the-difference-between-numpy-fft-fft-and-numpy-fft-rfft/52388007
    #will need to test it thouroughly though
    framesArray = np.fft.fftshift(np.fft.fft2(framesArray))
    return normaliseFourier(framesArray)

def normaliseFourier(frames):
    # Normalise the transform (based on size, move to real domain)
    scaling = (frames.shape[1] * frames.shape[2]) ^ 2
    normalised = np.square(np.absolute(frames))/scaling
    return normalised
