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
(tauVector, *fitParams, *, q)
where tauVector is the sequence of time differences and fitParams are the 
parameters to adjust to fit the function. The *,q marks q as a named argument,
which must be supplied. This must be present whether or not q is part of the
function, to ensure consistency.
They must return a single floating point value.
"""


#A simple one expected for pure diffusion in some circumstances, from:
#Differential Dynamic Microscopy of Bacterial Motility
#Wilson et al
#Published 5-JAN-2011
def diffusionFunction(tau, D, *, q):
    return np.exp(-D*(pow(q,2))*tau)

#Useable as an ultra-simple test case.
def linearFunction(tau, m, c, *, q):
    return tau*m + c
    
#Mimicing what is used in "Characterizing Concentraded, Multiply Scattering..."
#This one matches the provided experiment01 file quite well
#It is an exponential curve that decays from B up toward A+B
def risingExponential(deltaT, A, B, D, *, q):
    g = np.exp(-deltaT * D * (q**2))
    return A*(1-g)+B
   
"""
This is the dictionary from which to access fitting functions. Modify this here
whenever a new weighting scheme is added. DO NOT modify this during execution.
"""
FITTING_FUNCTIONS ={
    "diffusion":diffusionFunction,
    "rising exponential":risingExponential,
    "linear":linearFunction
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
    "linear":None
}
   
"""
These functions are used to create tau vectors to downsample the provided
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
    return np.arange(0,len(correlationCurve), spacing).astype(int)

def expSpacing(correlationCurve, sampleParams):
    perDecade = sampleParams[0]
    #Because we always want the first ten values
    zeroToNine = np.arange(0,10)
    #exponential spacings from 10 to the end
    expTauVector = pow(10, np.arange(1,np.log10(len(correlationCurve)),1/perDecade))
    #Cast to integer so it can be used to index
    expTauVector = np.concatenate((oneToNine, expTauVector)).astype(int)
    return expTauVector
   
def percentileSpacing(correlationCurve, sampleParams):
    percent = sampleParams[0]
    medianIntensity = np.percentile(correlationCurve, percent)
    exceedingElements = np.where(correlationCurve > medianIntensity)
    lastIndex = exceedingElements[0][0]
    tauVector = np.arange(0,lastIndex).astype(int)
    return tauVector
    
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
#TODO: figure out if there are any off-by-one errors in indexing correlations
#TODO: remove weighting scheme system, probably. It is more trouble than it's
#worth, because it means we'd have to include a time spacing vector at each q
"""
def fitCorrelationsToFunction(correlations, qValues, fitting, *, weighting = ("linear", (1,)), qCorrection = 1, timeSpacings = None, frameRate = 100):
    paramResults = [None] * correlations.shape[1]
    
    scheme = WEIGHTING_SCHEMES[weighting[0]]
    guessGenerator = GUESS_FUNCTIONS[fitting]
    fittingFunction = FITTING_FUNCTIONS[fitting]
    #size of one frame in pixels, calculated from maximum q
    frameSize = correlations.shape[1]*2
    #converts pixel q to q in um^-1, for use in fitting
    for q in qValues:
        #Choosing which samples to take, defaults to all
        tauVector = scheme(correlations[:,q], weighting[1])
        tauCurve = correlations[tauVector,q]
        if timeSpacings is None:
            #TODO: this is going to have an off-by-one error at the moment, needs to be taken care of
            tauVector = tauVector/frameRate
        else : #if it has been provided
            tauVector = timeSpacings
        #moving from pixels to um^-1
        realQ = q * qCorrection
        #Used to fix q on each iteration, so that it can be a part of the
        #fitting function without curve_fit trying to adjust it
        fitLambda = lambda tau, *args: fittingFunction(tau, *args, q=realQ)
        #Getting initial values for fitting, if they can be generated
        if guessGenerator is not None:
            (initials, paramBounds) = guessGenerator(correlations[:,q])
        else:#If they can't be, it uses no bounds and the default initials
            (initials, paramBounds) = (None, None)
        popt, pcov = scipy.optimize.curve_fit(fitLambda, tauVector, tauCurve, p0 = initials, bounds = paramBounds)
        paramResults[q] = popt
    
    return (paramResults, np.array(qValues) * qCorrection, fittingFunction)

def plotCurveComparisonsLinear(correlations, fittingResult, qIndicies, *, qCorrection = 1, timeSpacings = None, frameRate = 100):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        realQ = q * qCorrection
        if fitParams[q] is None:
            continue
        
        tauCurve = correlations[:,q]
        
        if timeSpacings is None:
            tauVector = np.arange(1,len(tauCurve)+1) / frameRate
            fitTauVector = np.arange(1,len(tauCurve)+1) / frameRate
        else:
            tauVector = timeSpacings
            #In the same range, but linearly spaced
            fitTauVector = np.arange(1, len(tauCurve) + 1) / len(timeSpacings) * np.max(timeSpacings)
        fitCurve = fitFunction(fitTauVector, *fitParams[q], q = realQ)
        
        plt.plot(tauVector, tauCurve, '-', color = plt.cm.viridis(i/len(qIndicies)), label = 'Actual data at q=%d inverse pixels' % q)
        plt.plot(fitTauVector, fitCurve, '--', color = plt.cm.viridis(i/len(qIndicies)), label = 'Fitted curve at q=%d inverse pixels' % q)
    plt.ylabel('\u0394(\u03B4t)')
    plt.xlabel('\u03B4t (Frames)')
    plt.legend()
    plt.title("Linearly Scaled Correlations")
    plt.show()
    

def plotCurveComparisonsLog(correlations, fittingResult, qIndicies, *, qCorrection = 1, timeSpacings = None, frameRate = 100):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        if fitParams[q] is None:
            continue
        realQ = q * qCorrection
        tauCurve = correlations[:,q]
        linTauVector = np.arange(1,len(tauCurve)+1)
        #Matches the spacings of the provided data, for plotting the given curve
        if timeSpacings is None:
            dataTauVector = linTauVector / frameRate
        else:
            dataTauVector = timeSpacings
        
        #Makes a log-spaced vector with the same range, for plotting fit
        logTauVector = np.logspace(np.log10(np.min(dataTauVector)), np.log10(np.max(dataTauVector)), num = len(tauCurve))
        #Have to use scatter for the data points, as it looks quite badly 
        #distorted using a line at low delta T
        plt.scatter(dataTauVector, tauCurve, color = plt.cm.viridis(i/len(qIndicies)), label = 'Actual data at q=%d inverse pixels' % q)
        plt.plot(logTauVector, fitFunction(logTauVector, *fitParams[q], q = realQ), '--', color = plt.cm.viridis(i/len(qIndicies)), label = 'Fitted curve at q=%d inverse pixels' % q)
    plt.xscale('symlog')
    plt.ylabel('\u0394(\u03B4t)')
    plt.xlabel('\u03B4t (Frames)')
    #Adjust the axes so that it starts at 0.01
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([0.01, xmax, ymin, ymax])
    plt.legend()
    plt.title("Logarithmically Scaled Correlations")
    plt.show()

"""
Returns a list of numpy arrays representing the fitted points generated by the
provided fittingResult tuple at the given time spacings or frame rate + number
Defaults to linear spacing from 10ms to 10s.
Ignores the frame rate and number of frames if given a time spacings vector
"""
def generateFittedCurves(fittingResult, qIndicies, *, timeSpacings = None, frameRate = 100, numFrames = 1000, qCorrection = 1):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    if timeSpacings is None:
            tauVector = np.arange(1/frameRate, (numFrames + 1) / frameRate, 1/frameRate)
    else:
        tauVector = timeSpacings
    
    curves = []
    for q in qIndicies:
        if fitParams[q] is None:
            continue
        realQ = q * qCorrection
        fitCurve = fitFunction(tauVector, *fitParams[q], q=realQ)
        curves.append(fitCurve)
    return np.array(curves)
    

"""
This is primarily just test code, but you might find a use for it. For each q,
it plots the data and the curves generated by both given fittings using the
fitting function for fitA. fitB is only used for its fit params, other entries
may be zero. Results are ploted by matplotlib
"""
def compareFittings(fitA, fitB, correlations, qIndicies, *, qCorrection = 1, timeSpacings = None, frameRate = 100):
    fitParamsA = fitA[0]
    fitParamsB = fitB[0]
    fitFunction = fitA[2]
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        realQ = q * qCorrection
        tauCurve = correlations[:,q]
        if timeSpacings is None:
            tauVector = np.arange(1,len(tauCurve)+1) / frameRate
        else:
            tauVector = timeSpacings
        fitCurveA = fitFunction(tauVector, *fitParamsA[q], q = realQ)
        fitCurveB = fitFunction(tauVector, *fitParamsB[q], q = realQ)
        plt.plot(tauVector,  tauCurve,  '-', color = plt.cm.viridis(i/len(qIndicies)), label = 'Actual data at q=%d inverse pixels' % q)
        plt.plot(tauVector, fitCurveA, '--', color = plt.cm.viridis(i/len(qIndicies)), label = 'Fitted curve A at q=%d inverse pixels' % q)
        plt.plot(tauVector, fitCurveB, '-.', color = plt.cm.viridis(i/len(qIndicies)), label = 'Fitted curve B at q=%d inverse pixels' % q)
    plt.ylabel('\u0394(\u03B4t)')
    plt.xlabel('\u03B4t (Frames)')
    plt.legend()
    plt.title("Linearly Scaled Comparison")
    plt.show()
    
#Use this if you just want to run it in the console quickly, from a file.
def bootstrap(path, qValues, fitting, *, weighting = ("linear",(1,)), zoom = 0.71):
    loadedData = np.loadtxt(path)
    #Assumes that it has been given data with time spacings at the start
    loadedCorrelations = loadedData[:,1:]
    timeSpacings = loadedData[:,0]
    qCorrection = (2*np.pi*zoom/((loadedCorrelations.shape[1])*2))
    fittingResult = fitCorrelationsToFunction(loadedCorrelations, qValues, 
        fitting, weighting = weighting, qCorrection = qCorrection, 
        timeSpacings = timeSpacings)
    plotCurveComparisonsLog(loadedCorrelations, fittingResult, (100,300,500), qCorrection= qCorrection, timeSpacings = timeSpacings)
    DList = []
    for fitTuple in fittingResult[0]:
        if fitTuple is not None:
            DList.append(fitTuple[2])
    DList = np.array(DList)
    plt.plot(np.array(qValues) * qCorrection, DList)
    plt.ylabel('Diffusion coefficient in um^2/s')
    plt.xlabel('q (um^-1)')
    plt.title("D as a function of q")
    plt.show()
    return (fittingResult, loadedCorrelations)
