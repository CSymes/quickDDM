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
fourierDifferences: a list of complex arrays, each array being (x, y, time)
RETURN: a list of 1d intensity arrays, where the sequence of list elements maps
to larger and larger time differences
"""
def calculateQCurves(fourierDifferences):
    
    dummyReturn = [np.arange(10),np.arange(10)]
    return dummyReturn