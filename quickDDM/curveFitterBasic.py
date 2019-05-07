# -*- coding: utf-8 -*-
#curveFitterBasic.py
"""
Created on Wed Apr 10 14:49:03 2019
Attempt at creating a basic curve-fitting tool that fits Q curves to the
various forms provided in the literature
Currently has to be modified in source to change fitting
@author: Lionel
"""
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

#A simple one expected for pure diffusion in some circumstances, from:
#Differential Dynamic Microscopy of Bacterial Motility
#Wilson et al
#Published 5-JAN-2011
#This one is useless with experiment01.avi
def diffusionFunction(tor, q, D):
    return np.exp(-D*(pow(q,2))*tor)

#Useable as an ultra-simple test case.
def linearFunction(tor, m, c):
    return tor*m + c

#For the formula used, please see:
#Differential Dynamic Microscopy of Bacterial Motility
#Wilson et al
#Published 5-JAN-2011
#This one is an absolute disaster with the provided file
def motilityFunction(tor, q, Z, alpha, vMean, D):
    theta = (q*tor*vMean)/(Z + 1)
    expTerm = np.exp(-D*(pow(q,2))*tor)
    velocityDistZTerm = (Z+1)/(Z*q*vMean*tor)
    velocitySineTerm = (np.sin(Z*np.arctan(theta)))/pow(1+theta,Z/2)
    return expTerm * ((alpha - 1) + alpha * velocityDistZTerm * velocitySineTerm)
    
#Mimicing what is used in "Characterizing Concentraded, Multiply Scattering..."
#This one matches the provided experiment01 file quite well
def risingExponential(deltaT, A, B, otherTor):
    g = np.exp(-deltaT/otherTor)
    return 2*A*(1-g)+B


    
"""
Extracts all fitting parameters from the correlations at the specified q
values, with the specified fitting.
correlations: The 2d matrix of intensity(dT,q)
qValues: list of q values at which to fit
fitting: string specifying which function to use
RETURN: tuple of (list(popt),list(curve),list(qIndex)) where popt is the tuple
of fitting parameters for a given q value, curve is the curve those parameters
produce, and qIndex is the index within the correlations list that maps to that
particular result's q value
"""
def fitCorrelationsToFunction(correlations, qValues, fitting):
    paramResults = []
    fittedCurves = []
    qIndexes = []
    for q in qValues:
        correctedQ = q*np.pi/1024.0
        torCurve = correlations[:,q]
        torVector = np.arange(1,len(torCurve)+1)
        #Extend this as required if more functions are wanted
        fittingFunctions ={
            "diffusion":diffusionFunction,
            #This sort of lambda is used to hide variables from the fitting,
            #mostly if the function is dependent on Q
            "motility":lambda tor, Z, alpha, vMean, D: motilityFunction(tor, correctedQ, Z, alpha, vMean, D),
            "rising exponential":risingExponential,
            "linear":linearFunction
        }
        popt, pcov = scipy.optimize.curve_fit(fittingFunctions[fitting], torVector, torCurve)
        paramResults.append(popt)
        #This * operator is a python-y thing that extracts a tuple into arguments
        currentCurve = fittingFunctions[fitting](torVector, *popt)
        fittedCurves.append(currentCurve)
    
    return (paramResults, fittedCurves, qIndexes)

def plotCurveComparisons(correlations, fittingResult, qIndicies):
    fitCurves = fittingResult[1]
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        torCurve = correlations[:,q]
        torVector = np.arange(1,len(torCurve)+1)
        plt.plot(torVector, torCurve, '-')
        plt.plot(torVector, fitCurves[i], '--')
    plt.show()
    
#Use this if you just want to run it in the console quickly, from a file.
def bootstrap(path, qValues, fitting):
    loadedCorrelations = np.loadtxt(path, delimiter = ' ')
    fittingResult = fitCorrelationsToFunction(loadedCorrelations, qValues, fitting)
    return fittingResult
