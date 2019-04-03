# -*- coding: utf-8 -*-
#basicMain.py
"""
Created on Tue Mar 26 14:26:28 2019
This is the most basic way of running the process. It will call each part
of the process in turn, chaining them together.
@author: Lionel
"""
import sys
import numpy as np
import readVideo as rV
import frameDifferencer as fD
import twoDFourier as tDF
import calculateQCurves as cQC
import calculateCorrelation as cC

#starting with the simplest case, consecutive frame differences
spacings = np.array((2,))
videoInput = rV.readVideo(sys.argv[1])
frameDifferences = fD.frameDifferencer(videoInput, spacings)
fourierSections = tDF.twoDFourier(frameDifferences)
qCurves = cQC.calculateQCurves(fourierSections)
correlations = cC.calculateCorrelation(qCurves)
if len(sys.argv) == 3:
    with open(sys.argv[2], "ab") as file:
        np.savetxt(file, correlations, delimiter = ',')