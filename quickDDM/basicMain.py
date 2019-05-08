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
    if outputPath is not None:
        with open(outputPath, "ab") as file:
            np.savetxt(file, correlations, delimiter = ' ')
    return correlations

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
    if outputPath is not None:
        with open(outputPath, "ab") as file:
            np.savetxt(file, correlations, delimiter = ' ')
    return correlations

def cumulativeDifferenceMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(videoInput, spacing)
        fourierMeans = tDF.cumulativeTransformAndAverage(frameDifferences)
        qCurve = cQC.calculateWithCalls(fourierMeans)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    if outputPath is not None:
        with open(outputPath, "ab") as file:
            np.savetxt(file, correlations, delimiter = ' ')
    return correlations

def realDifferenceMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(videoInput, spacing)
        fourierSections = tDF.realTwoDFourier(frameDifferences)
        qCurve = cQC.calculateRealQCurves(fourierSections)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    return correlations

def realTransformMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    scaling = (videoInput.shape[1] * videoInput.shape[2]) ^ 2
    fourierSections = np.fft.fftshift(np.fft.rfft2(videoInput), axes = (1,))
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(fourierSections, spacing)
        #At the moment this will normalise incorrectly, but that is ok for timing tests
        frameDifferences = np.square(np.absolute(frameDifferences))/scaling
        qCurve = cQC.calculateRealQCurves(frameDifferences)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    if outputPath is not None:
        with open(outputPath, "ab") as file:
            np.savetxt(file, correlations, delimiter = ' ')
    return correlations

def realAccumulateMain(videoPath, spacings, outputPath = None):
    correlations = []
    videoInput = rV.readVideo(videoPath)
    for spacing in spacings:
        frameDifferences = fD.frameDifferencer(videoInput, spacing)
        fourierMeans = tDF.cumulativeTransformAndAverageReal(frameDifferences)
        qCurve = cQC.calculateRealQCurves(fourierMeans)
        correlations.append(qCurve)
    correlations = cC.calculateCorrelation(correlations)
    if outputPath is not None:
        with open(outputPath, "ab") as file:
            np.savetxt(file, correlations, delimiter = ' ')
    return correlations

if __name__ == '__main__':
    differenceFirstMain(sys.argv[1], [1, 2, 3], sys.argv[2] if len(sys.argv) == 3 else None)
