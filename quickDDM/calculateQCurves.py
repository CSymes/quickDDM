# -*- coding: utf-8 -*-
#calculateQCurves.py
"""
Created on Wed Mar 27 11:06:47 2019
Takes in a list of transform arrays, does the averaging and squaring within
each time step, then computes the real Q curve for each time difference 
(equivalent to each list item)
@author: Lionel
"""


import numpy as np



"""
fourierDifferences: a list of complex arrays, each array being (time, y, x)
RETURN: a list of 1d intensity arrays, where the sequence of list elements maps
to larger and larger time differences
"""
def calculateQCurves(fourierDifferences):
    absolutes = np.square(np.absolute(fourierDifferences))/(fourierDifferences[0].shape[1]*fourierDifferences[0].shape[2])
    averages = []
    i = 0
    while(i < len(absolutes)):
        averages.append(np.mean(absolutes[i], axis = 0))
        absolutes[i] = 0 #Cleaning up after myself, freeing memory
        #TODO: consider supression of center values
        i += 1
    #TODO: for the averaging in a circle, look at this:
    #Link: https://stackoverflow.com/questions/8979214/iterate-over-2d-array-in-an-expanding-circular-spiral
    absolutes = 0 #just finishing the cleanup, just in case
    #For now, it is just mimicing the provided MATLAB, which isn't ideal, but sue me.
    yRange = np.arange(-averages[0].shape[0]/2.0,averages[0].shape[0]/2.0, dtype = np.int32)
    xRange = np.arange(-averages[0].shape[1]/2.0,averages[0].shape[1]/2.0, dtype = np.int32)
    xGrid, yGrid = np.meshgrid(xRange,yRange)
    #This hot mess gets the radial values as integers, with appropriate rounding
    radiusGrid = np.around(np.sqrt(np.square(yGrid) + np.square(xGrid)),0).astype(np.int16)
    yRange, xRange, xGrid, yGrid = 0, 0, 0, 0
    r = 0;
    i = 0;
    #TODO: test with different input dimensions (particularly odd lengths)
    qCurves = []
    while(i < len(averages)):
        qCurves.append(np.zeros(averages[i].shape[0]//2))
        #only while we get a full circle
        #TODO: try to rework this into numpy array format
        while(r < averages[i].shape[0]/2):
            pickGrid = radiusGrid == r
            qCurves[i][r] = (np.mean(averages[i][pickGrid]))
            r += 1
        i += 1
    return qCurves