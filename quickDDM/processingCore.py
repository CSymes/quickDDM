#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#processingCore.py
"""
Created on Tue Mar 26 14:26:28 2019
This is the most basic way of running the process. It will call each part
of the process in turn, chaining them together.
@author: Lionel
"""

import sys
import numpy as np
import readVideo as rV
import twoDFourier as tDF
import calculateQCurves as cQC
import calculateCorrelation as cC
from collections import deque

"""
Defaults to 1GB
Does not currently use spacings, simplest possible version
Based on the process outlined in figure 2 in the technical note
"""
def sequentialChunkerMain(videoPath, spacings, outputPath = None, RAMGB = 1, progress=None, abortFlag=None):
    if progress is not None:
        progress.setText('Reading Video from Disk')
        progress.cycle()

    videoInput = rV.readVideo(videoPath)
    numFrames = videoInput.shape[0]
    correlations = [None] * (numFrames - 1)

    #Number of pixels per frame, times 128 for the size of a complex float,
    #but halved because the real transform is used
    RAMBytes = RAMGB * np.power(2.0,30.0)
    complexFrameByteSize = videoInput.shape[1] * videoInput.shape[2] * 128 / 2
    #TODO: adjust for the other RAM using variables
    #one frame's RAM in reserve for the head
    framesPerSlice = int((RAMBytes // complexFrameByteSize) - 1)

    #The number of different slice intervals that must be take
    numSpacingSets = int(np.ceil((numFrames -1) / framesPerSlice))

    print('numSpacingSets:', numSpacingSets)
    print('framesPerSlice:', framesPerSlice)
    print('complexFrameByteSize:', complexFrameByteSize)

    # Used to show progress in the UI
    framesProcessed = 0
    target = numFrames * (numFrames - 1) / 2 # algorithm complexity
    # allow 10% extra time to calculate the q curves
    qProgress = target * 0.1 / numSpacingSets # per-slice q-curve allowance
    target += qProgress * numSpacingSets

    #For each diagonal section
    for sliceSpacing in range(0, numSpacingSets):
        if progress is not None: progress.setText(f'Working on Slice {sliceSpacing+1}/{numSpacingSets}')

        #A double ended queue, more efficient than a list for queue operations
        currentSlice = deque()
        #The index by which new frames are grabbed for the slice
        baseIndex = 0
        #Finding the expected shape of the transform results

        #trying something new, dropping a couple of samples to match matlab (1 in each dimension)
        if (videoInput.shape[2] - 1) % 2 == 0:
            transformShape = (videoInput.shape[1] - 1, (videoInput.shape[2] - 1)//2 + 1)
        else:
            #+1 for the real transform correction, -1 to drop a sample based on MATLAB
            transformShape = (videoInput.shape[1] - 1,(videoInput.shape[2]+1-1)//2)
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
            #Also drops a row and column
            currentSlice.append(tDF.realTwoDFourierUnnormalized(videoInput[baseIndex,:-1,:-1]))
            baseIndex += 1
            #Drops a row and column
            head = videoInput[headIndex,:-1,:-1]
            head = tDF.realTwoDFourierUnnormalized(head)
            #time difference between this frame and the first in the queue
            relativeDifference = 0
            #iterating backwards through the list, oldest element first
            for sliceFrameIndex in range(len(currentSlice) - 1, -1, -1):
                # Update progress tracker
                if progress is not None:
                    framesProcessed += 1
                    progress.setProgress(framesProcessed, target)

                difference = head - currentSlice[sliceFrameIndex]
                totalDifferences[relativeDifference,:,:] += tDF.castToReal(difference)
                numDifferences[relativeDifference] += 1
                relativeDifference += 1

                if abortFlag: return None

        for relativeDifference in range(0,len(currentSlice)):
            if progress is not None:
                framesProcessed += qProgress / len(currentSlice)
                progress.setProgress(framesProcessed, target)

            meanDifference = (totalDifferences[relativeDifference,:,:] / numDifferences[relativeDifference])
            timeDifference = relativeDifference + sliceSpacing * framesPerSlice
            correlations[timeDifference] = cQC.calculateRealQCurves(meanDifference)

            if abortFlag: return None

    if progress is not None:
        progress.cycle()
        progress.setText('Calculating Correlation Curves')

    correlations = cC.calculateCorrelation(correlations)


    frameRate = rV.readFramerate(videoPath)
    timeSpacings = np.array(np.arange(1,len(correlations) + 1)) / frameRate
    #This is the way to stack arrays in numpy
    outputMatrix = np.c_[timeSpacings, correlations]

    if abortFlag: return None

    if outputPath is not None:
        if progress is not None: progress.setText('Saving to Disk')

        np.savetxt(outputPath, outputMatrix)

    if progress is not None:
        progress.setPercentage(100)
        progress.setText('Done!')

    return outputMatrix

if __name__ == '__main__':
    sequentialChunkerMain('..\\tests\\data\\10frames.avi', None)
