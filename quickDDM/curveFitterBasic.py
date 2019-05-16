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
(torVector, *fitParams, *, q)
where torVector is the sequence of time differences and fitParams are the 
parameters to adjust to fit the function. The *,q marks q as a named argument,
which must be supplied. This must be present whether or not q is part of the
function, to ensure consistency.
They must return a single floating point value.
"""


#A simple one expected for pure diffusion in some circumstances, from:
#Differential Dynamic Microscopy of Bacterial Motility
#Wilson et al
#Published 5-JAN-2011
#Note that this uses q as a parameter, meaning a lambda must be used
def diffusionFunction(tor, D, *, q):
    return np.exp(-D*(pow(q,2))*tor)

#Useable as an ultra-simple test case.
def linearFunction(tor, m, c, *, q):
    return tor*m + c
    
#Mimicing what is used in "Characterizing Concentraded, Multiply Scattering..."
#This one matches the provided experiment01 file quite well
#It is an exponential curve that decays from B up toward 2A+B
def risingExponential(deltaT, A, B, D, *, q):
    g = np.exp(-deltaT/D)
    return 2*A*(1-g)+B
   
def risingExponentialWithQ(deltaT, A, B, D, *, q):
    g = np.exp(-deltaT * D * q^2)
    return 2*A*(1-g)+B
   
"""
This is the dictionary from which to access fitting functions. Modify this here
whenever a new weighting scheme is added. DO NOT modify this during execution.
"""
FITTING_FUNCTIONS ={
    "diffusion":diffusionFunction,
    "rising exponential":risingExponential,
    "linear":linearFunction,
    "rising with q":risingExponentialWithQ
}
   
   
"""
These functions serve to generate initial estimates and bounds for the fitting
functions, to get a nice, accurate fit. Largely based on the provided MATLAB.
The odd return signature is to best interface with the curve fitting function
Each one has a signature:
correlation: one correlation curve at fixed q for which to produce intial 
estimates and bounds.
RETURN: tuple (estimatesArray, (lowerBoundsArray, upperBoundsArray))
"""

#These are taken from the provided MATLAB
def risingExponentialGuess(correlation):
    Astart = np.max(correlation) - np.min(correlation)
    Bstart = np.min(correlation)
    Dstart = 0.3
    Amin = Astart / 2
    Amax = Astart * 2
    Bmin = Bstart / 2
    Bmax = Bstart * 2
    Dmin = 0.01
    Dmax = 80
    estimates = np.array([Astart, Bstart, Dstart])
    lowerBounds = np.array([Amin, Bmin, Dmin])
    upperBounds = np.array([Amax, Bmax, Dmax])
    return (estimates,(lowerBounds, upperBounds))
    
"""
This is the dictionary from which to access intial guess functions. Modify this
here whenever a new weighting scheme is added. DO NOT modify this during
execution.
"""
GUESS_FUNCTIONS ={
    "diffusion":None,#To do if required
    "rising exponential":risingExponentialGuess,
    "rising with q":risingExponentialGuess,
    "linear":None
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
zoom: parameter used to determine the size of a pixel. Typically 0.7, 1.42, 4.1
RETURN: tuple of (list(popt),list(qIndex), function) where popt is
the tuple of fitting parameters for a given q value, qIndex is the index within
the correlations list that maps to that particular result's q value, and 
function is the function it fitted to.
Values are populated to the indicies matching the Q values, all others are None
The exception being the q indicies themselves
#TODO: named tuples
#TODO: this is currently doing to q correction itself; consider where that should be done
"""
def fitCorrelationsToFunction(correlations, qValues, fitting, weighting = ("linear", (1,)), zoom = 0.71):
    paramResults = [None] * correlations.shape[1]
    #TODO: correct to use real dimensions, distribute this correction throughout
    
    scheme = WEIGHTING_SCHEMES[weighting[0]]
    guessGenerator = GUESS_FUNCTIONS[fitting]
    fittingFunction = FITTING_FUNCTIONS[fitting]
    #size of one frame in pixels, calculated from maximum q
    frameSize = (correlations.shape[1]-1)*2
    #converts pixel q to q in um^-1, for use in fitting
    qCorrection = (2*np.pi*zoom/(frameSize))
    for q in qValues:
        torVector = scheme(correlations[:,q], weighting[1])
        #correcting tor to use micro seconds
        torCurve = correlations[torVector,q]
        realQ = q * qCorrection
        #Used to fix q on each iteration, so that it can be a part of the
        #fittign function
        fitLambda = lambda tor, *args: fittingFunction(tor, *args, q=realQ)
        
        if guessGenerator is not None:
            (initials, paramBounds) = guessGenerator(correlations[:,q])
        else:
            (initials, paramBounds) = (None, None)
        popt, pcov = scipy.optimize.curve_fit(fitLambda, torVector, torCurve, p0 = initials, bounds = paramBounds)
        paramResults[q] = popt
    
    return (paramResults, qValues * qCorrection, fittingFunction, qCorrection)

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
        fitCurve = fitFunction(torVector, *fitParams[q], q = q)
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
        plt.plot(logTorVector, fitFunction(logTorVector, *fitParams[q], q = q), '--', color = plt.cm.viridis(i/len(qIndicies)), label = 'Fitted curve at q=%d inverse pixels' % q)
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
        fitCurve = fitFunction(torVector, *fitParams[q], q=q)
        curves.append(fitCurve)
    return curves
    
#Use this if you just want to run it in the console quickly, from a file.
def bootstrap(path, qValues, fitting, weighting = ("linear",(1,))):
    loadedCorrelations = np.loadtxt(path)
    fittingResult = fitCorrelationsToFunction(loadedCorrelations, qValues, fitting, weighting = weighting)
    return (fittingResult, loadedCorrelations)
