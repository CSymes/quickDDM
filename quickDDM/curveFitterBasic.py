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
    
#This will need to be adjusted for modularity. Perhaps taking the desired
#fitting style as a function pointer?
def fitCorrelationsToDiffusion(correlations, qValues):
    paramResults = []
    for q in qValues:
        #TODO: this needs to actual pull form the video size, should be
        #len * pi
        #It isn't required in all of the models either (?)
        correctedQ = q*np.pi/1024.0
        torCurve = correlations[:,q]
        torVector = np.arange(1,len(torCurve)+1)
        """
        TODO: consider defining upper and lower bands for parameters iwth know
        ranges in actual experiments
        uppers = ()
        lowers = ()
        """
        #Fit to motility
        #The ugly lambda structure is needed to hide q from the curve fitting
        #Use this structure whenever you need to include q as a function param
        #popt, pcov = scipy.optimize.curve_fit(lambda tor, Z, alpha, vMean, D: motilityFunction(tor, correctedQ, Z, alpha, vMean, D), torVector, torCurve)
        #If Q doesn't need to be hidden use this simpler form
        #popt, pcov = scipy.optimize.curve_fit(linearFunction, torVector, torCurve)
        popt, pcov = scipy.optimize.curve_fit(risingExponential, torVector, torCurve)
        paramResults.append(popt)
    return paramResults

#Use this if you just want to run it in the console quickly, from a file.
def bootstrap(path, qValues):
    loadedCorrelations = np.loadtxt(path, delimiter = ' ')
    fittingResult = fitCorrelationsToDiffusion(loadedCorrelations, qValues)
    return fittingResult
