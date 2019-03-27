# -*- coding: utf-8 -*-
#twoDFourier.py
"""
Created on Wed Mar 27 12:28:43 2019
Takes in a list of 3d arrays of frames, and runs the 2d fourier transform on
each frame
@author: Lionel
"""

"""
framesList: a list of 3d arrays formatted as
list([frame order, y position, x position])
RETURN: same format, but now list([frame order, inverse y, inverse x])
"""
def twoDFourier(framesList):
    dummyReturn = framesList
    return dummyReturn