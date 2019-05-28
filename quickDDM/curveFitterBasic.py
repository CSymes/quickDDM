# -*- coding: utf-8 -*-
#curveFitterBasic.py
"""
Created on Wed Apr 10 14:49:03 2019
This file handles all of the curve fitting. Any new models that are wanted must
be added here. Contains fitting logic, fitting models, fit parameter initial
estimate and bounds models for fitting, and some useful ways to display results
These display functions are NOT what the UI uses, they are for testing and
adjustment, not normal usage.
@author: Lionel
"""
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from inspect import signature

"""
These functions are the various functions to which to fit the provided data
They must take parameters:
(tauVector, *fitParams, *, q)
where tauVector is the sequence of time differences and fitParams are the 
parameters to adjust to fit the function.
The *,q marks q as a named argument, which must be supplied. This must be 
present whether or not q is part of the function, to ensure consistency.
The fitParams must be individual floating point numbers.
They must return a single floating point value.
"""


"""
Useable as an ultra-simple test case. Note that m and c are declared separately
and q is not used.
"""
def linearFunction(tau, m, c, *, q):
    return tau*m + c

"""    
This is a standard diffusion/brownian motion function.
It is an exponential curve that decays from B up toward A+B as tor increases.
The most relevant parameter to extract from fitting results here is D,
which is the diffusivity of the sample.
"""
def risingExponential(deltaT, A, B, D, *, q):
    g = np.exp(-deltaT * D * (q**2))
    return A*(1-g)+B
   
"""
This is the dictionary from which to access fitting functions. Modify this here
whenever a new weighting scheme is added. DO NOT modify this during execution.
Each entry should be addressed by a string, and have a value that is a fitting
function.
"""
FITTING_FUNCTIONS ={
    "rising exponential":risingExponential,
    "linear":linearFunction
}
   
   
"""
These functions serve to generate initial estimates and bounds for the fitting
functions, to get a nice, accurate fit.
The odd return signature is to best interface with the curve fitting function
Infinity is a valid bound, but if one intial guess or bound is returned, all
must have values. For example ((1,2), None, None) is a valid return, but not
((1,None), None, None) or ((1,2), (1,2,3), None)
Each one has a signature:
correlation: one correlation curve at fixed q for which to produce intial 
estimates and bounds.
RETURN: tuple (estimatesArray, (lowerBoundsArray, upperBoundsArray))
estimatesArray: The initial estimates for each fitting parameter, in order
lowerBoundsArray: The lower bounds for fitting, in order
upperBoundsArray: The upper bounds for fitting, in order
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
There MUST be an entry for each fitting model; if estimates are not desired,
use None.
"""
GUESS_FUNCTIONS ={
    "rising exponential":risingExponentialGuess,
    "linear":None
}
   
"""
These functions are used to create tau vectors to downsample the provided
correlation curve. They take parameters of 
(correlationCurve, sampleParams)
where correlationCurve is a single vector from the correlation function and 
sampleParams is a tuple of floats that serve as the parameters of the function
The sampleParams are a tuple so that different schemes may use different 
parameters
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
"""

def linearSpacing(correlationCurve, sampleParams):
    spacing = sampleParams[0]
    return np.arange(0,len(correlationCurve), spacing).astype(int)

def expSpacing(correlationCurve, sampleParams):
    perDecade = sampleParams[0]
    #Because we always want the first ten values
    zeroToNine = np.arange(0,10)
    #exponential spacings from 10 to the end
    expTauVector = pow(10, np.arange(1,np.log10(len(correlationCurve)),
            1/perDecade))
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
}

"""
Calculates all fitting parameters from the correlations at the specified q
values, with the specified fitting.

correlations: The 2d matrix of intensity[dT,q]
qValues: list of int, q values at which to fit, in pixels, used to index
    correlation matrix
fitting: string specifying which function to use as the model, see 
    FITTING_FUNCTIONS aboves
weighting: tuple(String,(*float)) how to bias the sampling (log, linear, etc)
    The floating point tuple is the parameter for the weighting, e.g how many
    samples per decade to take in exponential spacing
qCorrection: float, size of pixel in um. Calculated elsewhere.
timeSpacings: array(float), optional parameter, manually sets the time spacing
    between frames (can be nonlinear). If not provided, frames area ssumed to
    be linearly spaced, and the frame rate is used.
frameRate: float, optional parameter, used if no timeSpacings are provided.
    Measured in frames per second, assumes that frames are linearly spaced.
RETURN: tuple of (array(popt),array(realQ), function) where popt is
the 1D array of fitting parameters for a given q value, realQ is the q value 
in um^-1 that corresponds to that set of fitting parameters (equivalent to 
qValues * qCorrection), and function is a pointer to the function it used as 
a model.
The popt array has the appropriate set at each index in qValues, and is None 
elsewhere

A future project would do well to use named tuples for the output if any futher
functionality is required; this is unweildy as it is now.

It is very possible that the weighting component is unneded. Consider dropping
it if it isn't useful. Alternatively, just don't pass anything and it will use
the default, which is normally appropriate
"""
def fitCorrelationsToFunction(correlations, qValues, fitting, *, 
        weighting = ("linear", (1,)), qCorrection = 1, 
        timeSpacings = None, frameRate = 100):
    paramResults = [None] * correlations.shape[1]
    
    scheme = WEIGHTING_SCHEMES[weighting[0]]
    guessGenerator = GUESS_FUNCTIONS[fitting]
    fittingFunction = FITTING_FUNCTIONS[fitting]
    for q in qValues:
        #Choosing which samples to take, defaults to all
        tauVector = scheme(correlations[:,q], weighting[1])
        #The correlations at this q value
        tauCurve = correlations[tauVector,q]
        if timeSpacings is None:
            #default to using just the frame rate, assume linear spacing
            tauVector = tauVector/frameRate
        else : #if it has been provided, use it
            tauVector = timeSpacings
        #moving from pixels to um^-1
        realQ = q * qCorrection
        #Used to fix q on each iteration, so that it can be a part of the
        #fitting function without curve_fit trying to adjust it
        #The produced lambda has the same signature as the fitting function,
        #except for not taking the named parameter q
        fitLambda = lambda tau, *args: fittingFunction(tau, *args, q=realQ)
        #Getting initial values for fitting, if they can be generated
        if guessGenerator is not None:
            (initials, paramBounds) = guessGenerator(correlations[:,q])
            popt, pcov = scipy.optimize.curve_fit(fitLambda, tauVector,
                    tauCurve, p0 = initials, bounds = paramBounds)
        else:#If they can't be, it uses no bounds and the default initials
            #This needs to set up so that the number of parameters can be read,
            #because they are hidden from the fitting by the lambda
            #The -2 removes the q and x parameters, as it does not fit on these
            initials = (1,) * (len(signature(fittingFunction).parameters) - 2)
            popt, pcov = scipy.optimize.curve_fit(fitLambda, tauVector,
                    tauCurve, p0 = initials)
        #Calculating the actual fitting parameters
        
        paramResults[q] = popt
    
    return (paramResults, np.array(qValues) * qCorrection, fittingFunction)



"""
Uses matplotlib to display a number of fitted curves and the underlying data.
This doesn't play nice with the UI, and is primarily useful as test code.

correlations: The 2d matrix of intensity[dT,q]
fittingResult: tuple of (array(popt),array(realQ), function), as generated
    by fitCorrelationsToFunction()
qIndicies: list of int, q values at which to calculate, in pixels.
frameRate: float, optional, used if no timeSpacings are provided.
    Measured in frames per second, assumes that frames are linearly spaced.
timeSpacings: array(float), optional, manually sets the time spacing
    between frames (can be nonlinear). If not provided, frames area assumed to
    be linearly spaced, and the frame rate is used.
qCorrection: float, size of pixel in um.

Does not return a value.
"""
def plotCurveComparisonsLinear(correlations, fittingResult, qIndicies,
        *, qCorrection = 1, timeSpacings = None, frameRate = 100):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        realQ = q * qCorrection
        if fitParams[q] is None:
            continue
        
        tauCurve = correlations[:,q]
        
        if timeSpacings is None:
            #If not given, both the vector for plotting and the vector for
            #calculating the curve should be simple linearly spaced vectors
            tauVector = np.arange(1,len(tauCurve)+1) / frameRate
            fitTauVector = np.arange(1,len(tauCurve)+1) / frameRate
        else:
            tauVector = timeSpacings
            #Otherwise, the vector for caculating the fitting should be linear
            #across the same range as the provided data
            fitTauVector = np.arange(1, len(tauCurve) + 1)
                    / len(timeSpacings) * np.max(timeSpacings)
        fitCurve = fitFunction(fitTauVector, *fitParams[q], q = realQ)
        
        plt.plot(tauVector, tauCurve, '-', 
                color = plt.cm.viridis(i/len(qIndicies)), 
                label = 'Actual data at q=%d inverse pixels' % q)
        plt.plot(fitTauVector, fitCurve, '--', 
                color = plt.cm.viridis(i/len(qIndicies)), 
                label = 'Fitted curve at q=%d inverse pixels' % q)
    plt.ylabel('\u0394(\u03B4t)')
    plt.xlabel('\u03B4t (Frames)')
    plt.legend()
    plt.title("Linearly Scaled Correlations")
    plt.show()
    

    
"""
Please see description of plotCurveComparisonsLinear above. This serves the
same purpose, but instead plots on a log scale
"""
def plotCurveComparisonsLog(correlations, fittingResult, qIndicies,
        *, qCorrection = 1, timeSpacings = None, frameRate = 100):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    
    for i in range(0,len(qIndicies)):
        q = qIndicies[i]
        if fitParams[q] is None:
            continue
        realQ = q * qCorrection
        tauCurve = correlations[:,q]
        linTauVector = np.arange(1,len(tauCurve)+1)
        #Matches the spacing of the provided data, for plotting the given curve
        if timeSpacings is None:
            dataTauVector = linTauVector / frameRate
        else:
            dataTauVector = timeSpacings
        
        #Makes a log-spaced vector with the same range, for plotting fit
        logTauVector = np.logspace(np.log10(np.min(dataTauVector)), 
                np.log10(np.max(dataTauVector)), num = len(tauCurve))
        #Have to use scatter for the data points, as it looks quite badly 
        #distorted using a line at low delta T
        #viridis is the colour map used, set up to evenly space across the map
        #for maximum clarity
        plt.scatter(dataTauVector, tauCurve, 
                color = plt.cm.viridis(i/len(qIndicies)), 
                label = 'Actual data at q=%d inverse pixels' % q)
        plt.plot(logTauVector, 
                fitFunction(logTauVector, *fitParams[q], q = realQ), '--', 
                color = plt.cm.viridis(i/len(qIndicies)), 
                label = 'Fitted curve at q=%d inverse pixels' % q)
    plt.xscale('symlog')
    plt.ylabel('\u0394(\u03B4t)')
    plt.xlabel('\u03B4t (Frames)')
    #Adjust the axes so that it starts at 0.01, makes graphs cleaner
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([0.01, xmax, ymin, ymax])
    plt.legend()
    plt.title("Logarithmically Scaled Correlations")
    plt.show()

"""
Returns an array of numpy arrays representing the fitted points generated by 
the provided fittingResult tuple at the given time spacings or 
frame rate + number of frames
Defaults to linear spacing from 10ms to 10s.
Ignores the frame rate and number of frames if given a time spacings vector

fittingResult: tuple of (array(popt),array(realQ), function), as generated
    by fitCorrelationsToFunction()
qIndicies: list of int, q values at which to calculate, in pixels.
timeSpacings: array(float), optional, manually sets the time spacing
    between frames (can be nonlinear). If not provided, frames area assumed to
    be linearly spaced, and the frame rate is used.
frameRate: float, optional, used if no timeSpacings are provided.
    Measured in frames per second, assumes that frames are linearly spaced.
numFrames: int, optional, number of time steps to plot across.
qCorrection: float, size of pixel in um.
RETURN: 2D array, indexed by [frame number, q within qIndicies], holding the
curves, the first curve being return[:, 0]
"""
def generateFittedCurves(fittingResult, qIndicies, *, timeSpacings = None, 
        frameRate = 100, numFrames = 1000, qCorrection = 1):
    fitParams = fittingResult[0]
    fitFunction = fittingResult[2]
    #Setting up the vector of time differences to plot agains
    if timeSpacings is None:#Default using frame rate and num frames
            tauVector = np.arange(1/frameRate, (numFrames + 1) / frameRate, 1/frameRate)
    else:#Or the manually provided values
        tauVector = timeSpacings
    
    curves = []
    for q in qIndicies:
        #Ignores values with no given fit parameters
        if fitParams[q] is None:
            continue
        realQ = q * qCorrection
        fitCurve = fitFunction(tauVector, *fitParams[q], q=realQ)
        curves.append(fitCurve)
    #Casts to numpy array for usability elsewhere
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
 
"""
THIS IS NOT PRODUCTION CODE, A FINISHED SYSTEM SHOULD NEVER RUN THIS
Use this if you just want to run this process in the console quickly, from a 
file. This is incredibly useful for debugging. It loads in correlations, treats
the first slice as the time spacings (what we expect from the main process),
calculates the fittings, and displays the result, including a diffusivity curve
It returns the correlations and the fitting result tuple. 
""" 
def bootstrap(path, qValues, fitting, *, weighting = ("linear",(1,)), 
        zoom = 0.71):
    loadedData = np.loadtxt(path)
    #Assumes that it has been given data with time spacings at the start
    loadedCorrelations = loadedData[:,1:]
    timeSpacings = loadedData[:,0]
    qCorrection = (2*np.pi*zoom/((loadedCorrelations.shape[1])*2))
    fittingResult = fitCorrelationsToFunction(loadedCorrelations, qValues, 
        fitting, weighting = weighting, qCorrection = qCorrection, 
        timeSpacings = timeSpacings)
    plotCurveComparisonsLog(loadedCorrelations, fittingResult, (100,300,500), 
            qCorrection= qCorrection, timeSpacings = timeSpacings)
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
