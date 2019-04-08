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
    #TODO: rearrange to avoid repeated division
    absolutes = np.square(np.absolute(fourierDifferences))/((fourierDifferences.shape[1]*fourierDifferences.shape[2])^2)
    ySize = fourierDifferences.shape[1]
    xSize = fourierDifferences.shape[2]
    averages = np.mean(absolutes, axis = 0)
    absolutes = 0 #Cleaning up after myself
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
    absolutes = absolutesInitial(fourierDifferences)
    averages = averagesLoop(absolutes)
    absolutes = 0
    radiusGrid = generateGrid(averages.shape)
    averagedCurves = takeCurves(averages, radiusGrid)
    return averagedCurves

def absolutesInitial(fourierDifferences):
    sizeCorrection = 1/fourierDifferences.shape[1]*fourierDifferences.shape[2]
    return np.square(np.absolute(fourierDifferences))*(sizeCorrection)

def averagesLoop(absolutes):
    averages = np.mean(absolutes, axis = 0)
    #TODO: consider supression of center values
    return averages

def generateGrid(dims):
    yRange = np.arange(-dims[0]/2.0,dims[0]/2.0, dtype = np.int32)
    xRange = np.arange(-dims[1]/2.0,dims[1]/2.0, dtype = np.int32)
    xGrid, yGrid = np.meshgrid(xRange,yRange)
    #This hot mess gets the radial values as integers, with appropriate rounding
    radiusGrid = np.around(np.sqrt(np.square(yGrid) + np.square(xGrid)),0).astype(np.int16)
    return radiusGrid

def takeCurves(averages, radiusGrid):
    r = 0;
    i = 0;
    qCurves = np.zeros(averages.shape[0]//2)
    while(r < averages.shape[0]/2):
        pickGrid = radiusGrid == r
        qCurves[r] = (np.mean(averages[pickGrid]))
        r += 1
    return qCurves