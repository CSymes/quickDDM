# -*- coding: utf-8 -*-
#calculateRadiusGrid.py
"""
Created on Fri Apr 05 10:29:32 2019
Saves a grid of radii to a file, so that it may be read instead of generated in future
@author: Lionel
"""


import numpy as np


"""
size: integer side length of the square to be generated
"""
def createGrid(size):
    yRange = np.arange(-size/2, size/2, dtype = np.int32)
    xRange = np.arange(-size/2, size/2, dtype = np.int32)
    xGrid, yGrid = np.meshgrid(xRange,yRange)
    radiusGrid = np.around(np.sqrt(np.square(yGrid) + np.square(xGrid)),0).astype(np.int32)
    #TODO: test with different input dimensions (particularly odd lengths)
    with open("../data/radiusGrid.txt", "wb") as file:
        #%i marks it as an integer to save
        np.savetxt(file, radiusGrid, fmt='%i')
    return radiusGrid