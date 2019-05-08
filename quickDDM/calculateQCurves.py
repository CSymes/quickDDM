# -*- coding: utf-8 -*-
#calculateQCurves.py
"""
Created on Wed Mar 27 11:06:47 2019
Takes in atransform array, does the averaging and squaring within
each time step, then computes the real Q curve 
(equivalent to each list item)
@author: Lionel
"""


import numpy as np



"""
fourierDifferences: a complex array, (time, y, x)
RETURN: a 1d intensity array, at increasing radii
"""
def calculateQCurves(fourierDifferences):
    averages = np.mean(fourierDifferences, axis = 0)
    ySize = fourierDifferences.shape[1]
    xSize = fourierDifferences.shape[2]
    #For now, it is just mimicing the provided MATLAB, which isn't ideal, but sue me.
    yRange = np.arange(-ySize/2.0,ySize/2.0, dtype = np.int32)
    xRange = np.arange(-xSize/2.0,xSize/2.0, dtype = np.int32)
    xGrid, yGrid = np.meshgrid(xRange,yRange)
    #This hot mess gets the radial values as integers, with appropriate rounding
    radiusGrid = np.around(np.sqrt(np.square(yGrid) + np.square(xGrid)),0).astype(np.int16)
    yRange, xRange, xGrid, yGrid = 0, 0, 0, 0
    
    #Removing the centre values
    radiusGrid[:,xSize//2] = -1
    radiusGrid[ySize//2,:] = -1
    r = 1;
    #TODO: test with different input dimensions (particularly odd lengths)
    qCurve = np.zeros(averages.shape[0]//2)
    #only while we get a full circle
    #TODO: try to rework this into numpy array format
    while(r < xSize/2):
        pickGrid = radiusGrid == r
        qCurve[r] = (np.mean(averages[pickGrid]))
        r += 1
    return qCurve

def calculateWithCalls(fourierDifferences):
    # Normalising here no longer required - performed in the Fourier module
    if len(fourierDifferences.shape) == 3:
        averages = averagesLoop(fourierDifferences)
    else:
        averages = fourierDifferences
    radiusGrid = generateGrid(averages.shape)
    averagedCurves = takeCurves(averages, radiusGrid)
    return averagedCurves

def averagesLoop(absolutes):
    averages = np.mean(absolutes, axis = 0)
    return averages

def generateGrid(dims):
    yRange = np.arange(-dims[0]/2.0,dims[0]/2.0, dtype = np.int32)
    xRange = np.arange(-dims[1]/2.0,dims[1]/2.0, dtype = np.int32)
    xGrid, yGrid = np.meshgrid(xRange,yRange)
    #This hot mess gets the radial values as integers, with appropriate rounding
    radiusGrid = np.around(np.sqrt(np.square(yGrid) + np.square(xGrid)),0).astype(np.int16)
    #Removing the centre values
    radiusGrid[:,dims[0]//2] = -1
    radiusGrid[dims[1]//2,:] = -1
    
    return radiusGrid

def takeCurves(averages, radiusGrid):
    r = 1;
    qCurves = np.zeros(averages.shape[0]//2)
    while(r < averages.shape[0]/2):
        pickGrid = radiusGrid == r
        qCurves[r] = (np.mean(averages[pickGrid]))
        r += 1
    return qCurves