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

"""
These functions are the various functions to which to fit the provided data
They must take parameters:
(torVector, *fitParams)
where torVector is the sequence of time differences and fitParams are the 
parameters to adjust to fit the function. If additional fixed parameters are 
required, create a lambda using the function with an appropriate signature
They must return a single floating point value.
"""


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
    
#Mimicing what is used in "Characterizing Concentraded, Multiply Scattering..."
#This one matches the provided experiment01 file quite well
#It is an exponential curve that decays from B up toward 2A+B
def risingExponential(deltaT, A, B, otherTor):
    g = np.exp(-deltaT/otherTor)
    return 2*A*(1-g)+B
   
"""
This is the dictionary from which to access fitting functions. Modify this here
whenever a new weighting scheme is added. DO NOT modify this during execution.
"""
fittingFunctions ={
    "diffusion":diffusionFunction,
    "rising exponential":risingExponential,
    "linear":linearFunction
}
   
"""
These functions are used to create tor vectors to downsample the provided
correlation curve. They take parameters of 
(correlationCurve, sampleParams)
where correlationCurve is a single vector from the correlation function and 
sampleParams is a tuple of floats that serve as the parameters of the function
RETURN: a vector of integers from 1 to some values less than 
len(correlationCurve), that may be used to index correlationCurve.
available weighting schemes:
    linear: takes linearly spaced samples. One param, the spacing desired,
    should be >= 1 to avoid repeated elements.
    log: baises towards early values by using exponentially spaced samples.
    One param how many samples should be taken per decade. All available
    samples are taken in the first 10, as these are the most important for
    fitting.
    percentile: Excludes values after the first element that exceeds the nth
    percentile of the curve. One param, the percentile to cut off at. 
    This gives a much shorter vector, and needs more understanding of the 
    expected curve than the others.
    
Not yet implemented:
    %rise: cut off after the first element above a certain % of the steady
    section's mean
    hardcutoff: exclude samples past the given number
    averagingexp: uses a moving average filter for points past the 
    first ten to mitigate noise, then samples exponentially
"""

def linearSpacing(correlationCurve, sampleParams):
    spacing = sampleParams[0]
    return np.arange(1,len(correlationCurve), spacing).astype(int)

def expSpacing(correlationCurve, sampleParams):
    perDecade = sampleParams[0]
    #Because we always want the first ten values
    oneToNine = np.arange(1,10)
    #exponential spacings from 10 to the end
    expTorVector = pow(10, np.arange(1,np.log10(len(correlationCurve)),1/perDecade))
    #Cast to integer so it can be used to index
    expTorVector = np.concatenate((oneToNine, expTorVector)).astype(int)
    return expTorVector
   
def percentileSpacing(correlationCurve, sampleParams):
    percent = sampleParams[0]
    medianIntensity = np.percentile(correlationCurve, percent)
    exceedingElements = np.where(correlationCurve > medianIntensity)
    lastIndex = exceedingElements[0][0]
    torVector = np.arange(1,lastIndex).astype(int)
    return torVector
    
"""
This is the dictionary from which to access weighting schemes. Modify this here
whenever a new weighting scheme is added. DO NOT modify this during execution.
"""
WEIGHTING_SCHEMES ={
    #More can be added here as required
    "linear": linearSpacing,
    "log": expSpacing,
    "percentile": percentileSpacing
    #These can be added later if desired
    #"%rise": ,
    #"hardcutoff":,
    #"averagingexp":
}

"""
Extracts all fitting parameters from the correlations at the specified q
values, with the specified fitting.
correlations: The 2d matrix of intensity(dT,q)
qValues: list of q values at which to fit
fitting: string specifying which function to use
weighting: tuple(String,float) how to bias the sampling (log, linear, etc)
RETURN: tuple of (list(popt),list(qIndex), function) where popt is
the tuple of fitting parameters for a given q value, qIndex is the index within
the correlations list that maps to that particular result's q value, and 
function is the function it fitted to.
Values are populated to the indicies matching the Q values, all others are None
The exception being the q indicies themselves
#TODO: named tuples
"""
def fitCorrelationsToFunction(correlations, qValues, fitting, weighting = ("linear", (1,))):
    paramResults = [None] * correlations.shape[1]
    #TODO: correct to use real dimensions
    #TODO: make a new tool for estimating initial values
    if fitting == "rising exponential":
        guess = (np.max(correlations) - np.min(correlations), np.min(correlations), 0.3)
    else :
        guess = None
    
    scheme = WEIGHTING_SCHEMES[weighting[0]]
    for q in qValues:
        torVector = scheme(correlations[:,q], weighting[1])
        torCurve = correlations[torVector,q]
        popt, pcov = scipy.optimize.curve_fit(fittingFunctions[fitting], torVector, torCurve, p0 = guess)
        paramResults[q] = popt
    
    return (paramResults, qValues, fittingFunctions[fitting])

def plotCurveComparisonsLinear(correlations, fittingResult, qIndicies):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        if fitParams[q] is None:
            continue
        #Drops the first element, as it is useless and not used in fitting
        torCurve = correlations[1:,q]
        torVector = np.arange(1,len(torCurve)+1)
        fitCurve = fitFunction(torVector, *fitParams[q])
        plt.plot(torVector, torCurve, '-', color = plt.cm.viridis(i/len(qIndicies)), label = 'Actual data at q=%d inverse pixels' % q)
        plt.plot(torVector, fitCurve, '--', color = plt.cm.viridis(i/len(qIndicies)), label = 'Fitted curve at q=%d inverse pixels' % q)
    plt.ylabel('\u0394(\u03B4t)')
    plt.xlabel('\u03B4t (Frames)')
    plt.legend()
    plt.title("Linearly Scaled Correlations")
    plt.show()
    
def plotCurveComparisonsLog(correlations, fittingResult, qIndicies):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        if fitParams[q] is None:
            continue
        #Drops the first element, as it is useless and not used in fitting
        torCurve = correlations[1:,q]
        linTorVector = np.arange(1,len(torCurve)+1)
        #Generates a log-spaced vector with the same number of elements, for neat plotting
        logTorVector = np.logspace(0, np.log10(len(torCurve)), num = len(torCurve))
        #Have to use scatter for the data points, as it looks quite badly 
        #distorted using a line at low delta T
        plt.scatter(linTorVector, torCurve, color = plt.cm.viridis(i/len(qIndicies)), label = 'Actual data at q=%d inverse pixels' % q)
        plt.plot(logTorVector, fitFunction(logTorVector, *fitParams[q]), '--', color = plt.cm.viridis(i/len(qIndicies)), label = 'Fitted curve at q=%d inverse pixels' % q)
    plt.xscale('symlog')
    plt.ylabel('\u0394(\u03B4t)')
    plt.xlabel('\u03B4t (Frames)')
    #Adjust the axes so that it starts at 0.01
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([0.01, xmax, ymin, ymax])
    plt.legend()
    plt.title("Logarithmically Scaled Correlations")
    plt.show()

def generateFittedCurves(fittingResult, startTor, endTor, stepTor):
    fitParams = fittingResult[0]
    qValues = fittingResult[1]
    fitFunction = fittingResult[2]
    torVector = np.arange(startTor, endTor + stepTor, stepTor)
    curves = []
    for q in qValues:
        if fitParams[q] is None:
            continue
        fitCurve = fitFunction(torVector, *fitParams[q])
        curves.append(fitCurve)
    return curves
    
#Use this if you just want to run it in the console quickly, from a file.
def bootstrap(path, qValues, fitting, weighting = ("linear",(1,))):
    loadedCorrelations = np.loadtxt(path)
    fittingResult = fitCorrelationsToFunction(loadedCorrelations, qValues, fitting, weighting)
    return (fittingResult, loadedCorrelations)
