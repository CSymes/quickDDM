#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#basicMain.py
"""
Created on Tue Mar 26 14:26:28 2019
This is the most basic way of running the process. It will call each part
of the process in turn, chaining them together.
@author: Lionel
"""
import numpy as np
import readVideo as rV
import frameDifferencer as fD
import twoDFourier as tDF
import calculateQCurves as cQC
import calculateCorrelation as cC
from collections import deque

#starting with the simplest case, consecutive frame differences
def differenceFirstMain(videoPath, spacings, outputPath = None):
    #spacings = np.array((13,14,15))
    correlations = []
    videoInput = rV.readVideo(videoPath)
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(videoInput, spacing)
        fourierSections = tDF.twoDFourier(frameDifferences)
        qCurve = cQC.calculateQCurves(fourierSections)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    
    frameRate = rV.readFramerate(videoPath)
    timeSpacings = np.array(spacings) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]
    if outputPath is not None:
        np.savetxt(outputPath, outputMatrix)
    
    return outputMatrix

def transformFirstMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    fourierSections = np.fft.fftshift(np.fft.fft2(videoInput), axes = (1,2))
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(fourierSections, spacing)
        frameDifferences = tDF.normaliseFourier(frameDifferences)
        qCurve = cQC.calculateQCurves(frameDifferences)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    frameRate = rV.readFramerate(videoPath)
    timeSpacings = np.array(spacings) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]
    if outputPath is not None:
        np.savetxt(outputPath, outputMatrix)
    return outputMatrix

def cumulativeDifferenceMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(videoInput, spacing)
        fourierMeans = tDF.cumulativeTransformAndAverage(frameDifferences)
        qCurve = cQC.calculateWithCalls(fourierMeans)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    frameRate = rV.readFramerate(videoPath)
    timeSpacings = np.array(spacings) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]
    if outputPath is not None:
        np.savetxt(outputPath, outputMatrix)
    return outputMatrix

def realDifferenceMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(videoInput, spacing)
        fourierSections = tDF.realTwoDFourier(frameDifferences)
        qCurve = cQC.calculateRealQCurves(fourierSections)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    frameRate = rV.readFramerate(videoPath)
    timeSpacings = np.array(spacings) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]
    if outputPath is not None:
        np.savetxt(outputPath, outputMatrix)
    return outputMatrix

def realTransformMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    scaling = (videoInput.shape[1] * videoInput.shape[2]) ** 2
    fourierSections = np.fft.fftshift(np.fft.rfft2(videoInput), axes = (1,))
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(fourierSections, spacing)
        #At the moment this will normalise incorrectly, but that is ok for timing tests
        frameDifferences = np.square(np.absolute(frameDifferences))/scaling
        qCurve = cQC.calculateRealQCurves(frameDifferences)
        correlations.append(qCurve)
    frameRate = rV.readFramerate(videoPath)
    timeSpacings = np.array(spacings) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]
    if outputPath is not None:
        np.savetxt(outputPath, outputMatrix)
    return outputMatrix

def realAccumulateMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(videoInput, spacing)
        fourierMeans = tDF.cumulativeTransformAndAverageReal(frameDifferences)
        qCurve = cQC.calculateRealQCurves(fourierMeans)
        correlations.append(qCurve)
    frameRate = rV.readFramerate(videoPath)
    timeSpacings = np.array(spacings) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]
    if outputPath is not None:
        np.savetxt(outputPath, outputMatrix)
    return outputMatrix

#defaults to 1GB
#does not currently use spacings, simplest possible version
#based on figure 2 in the technical note
def sequentialChunkerMain(videoPath, spacings, outputPath = None, RAMGB = 1):
    RAMBytes = RAMGB * np.power(2.0,30.0)
    videoInput = rV.readVideo(videoPath)
    numFrames = videoInput.shape[0]
    correlations = [None] * (numFrames - 1)
    #Number of pixels per frame, times 128 for the size of a complex float,
    #but halved because the real transform is used
    complexFrameByteSize = videoInput.shape[1] * videoInput.shape[2] * 128 / 2
    #TODO: adjust for the other RAM using variables
    #one frame's RAM in reserve for the head
    framesPerSlice = int((RAMBytes // complexFrameByteSize) - 1)
    #The number of different slice intervals that must be take
    numSpacingSets = int(np.ceil((numFrames -1) / framesPerSlice))
    print('numSpacingSets')
    print(numSpacingSets)
    print('framesPerSlice')
    print(framesPerSlice)
    print('complexFrameByteSize')
    print(complexFrameByteSize)
    #For each diagonal section
    for sliceSpacing in range(0, numSpacingSets):
        #A double ended queue, more efficient than a list for queue operations
        currentSlice = deque()
        #The index by which new frames are grabbed for the slice
        baseIndex = 0
        #Finding the expected shape of the transform results
        if videoInput.shape[2] % 2 == 0:
            transformShape = (videoInput.shape[1],videoInput.shape[2]//2 + 1)
        else:
            transformShape = (videoInput.shape[1],(videoInput.shape[2]+1)//2)
        totalDifferencesShape = (framesPerSlice, transformShape[0], transformShape[1])
        #Preparing the destination of the frame differences
        totalDifferences = np.zeros(totalDifferencesShape)
        numDifferences = np.zeros((framesPerSlice,))
        #For each head
        for headIndex in range((sliceSpacing * framesPerSlice) + 1, numFrames):
            #If the queue is full, remove the oldest element
            if len(currentSlice) == framesPerSlice:
                currentSlice.popleft()
            #Get a new value into the slice queue
            currentSlice.append(tDF.realTwoDFourierUnnormalized(videoInput[baseIndex,:,:]))
            baseIndex += 1
            
            head = videoInput[headIndex,:,:]
            head = tDF.realTwoDFourierUnnormalized(head)
            #time difference between this frame and the first in the queue
            relativeDifference = 0
            #iterating backwards through the list, oldest element first
            for sliceFrameIndex in range(len(currentSlice) - 1, -1, -1):
                difference = head - currentSlice[sliceFrameIndex]
                totalDifferences[relativeDifference,:,:] += tDF.castToReal(difference)
                numDifferences[relativeDifference] += 1
                relativeDifference += 1
            
        for relativeDifference in range(0,len(currentSlice)):
            meanDifference = (totalDifferences[relativeDifference,:,:] / numDifferences[relativeDifference])
            timeDifference = relativeDifference + sliceSpacing * framesPerSlice
            correlations[timeDifference] = cQC.calculateRealQCurves(meanDifference)
    correlations = cC.calculateCorrelation(correlations)
    
    
    frameRate = rV.readFramerate(videoPath)
    #TODO: if spacings are introduced, revise this to use them
    timeSpacings = np.array(np.arange(1,len(correlations) + 1)) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]
    if outputPath is not None:
        np.savetxt(outputPath, outputMatrix)
    return outputMatrix
    
if __name__ == '__main__':
    differenceFirstMain(sys.argv[1], [1, 2, 3], sys.argv[2] if len(sys.argv) == 3 else None)
