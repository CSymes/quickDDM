# -*- coding: utf-8 -*-
#calculateCorrelation.py
"""
Created on Wed Mar 27 11:48:12 2019
Takes in Q curves in the form of a list of arrays and turns them into
correlation curves at the various time spacings
@author: Lionel
"""
import numpy as np

"""
qCurves: a list of 1d arrays of intensity
RETURN: the array of all q curves, with matrix[n] getting the n-th q-curve and
matrix[:,n] getting a curve of intensity with respect to time difference at a
given inverse raduis i.e. real angle
"""
def calculateCorrelation(qCurves): 
    matrix = np.array(qCurves)
    #This seems to be all we need, just addressing it appropriately
    return matrix