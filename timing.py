#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#testTiming.py
"""
Created 2019-04-24
Runs a bunch of timers over sections of the main program for comparison purposes
@author: Cary
"""

import quickDDM.readVideo as readVideo
import quickDDM.frameDifferencer as frameDifferencer
import quickDDM.twoDFourier as twoDFourier
import quickDDM.calculateQCurves as calculateQCurves
import quickDDM.calculateCorrelation as calculateCorrelation

from timeit import default_timer as time
from collections import defaultdict
import sys
import cv2
import os




"""
Wraps all the module methods in timing calls and stores the wrapped
functions in the system import list
"""
def createWrappers(order, times):
	def wrap(module, smod, sfunc, tname):
		func = getattr(module, sfunc)
		def wrapped_func(*args): # create a wrapper function
			startTime = time() # take initial timestamp
			r = func(*args) # call the real function with all given args
			times[tname] += time() - startTime # store total time consumed
			return r # transparently return
		setattr(module, sfunc, wrapped_func) # overwrite the function in its module
		sys.modules[smod] = module # and pre-"import" it for processingCore.py
		# this is important, since processingCore doesn't know it's in the quickDDM namespace
		order.append(tname) # keep track of function order

	wrap(readVideo, 'readVideo', 'readVideo', 'Video Read')
	wrap(frameDifferencer, 'frameDifferencer', 'frameDifferencer', 'Differencing')
	wrap(twoDFourier, 'twoDFourier', 'twoDFourier', '2D Fourier (basic)')
	wrap(twoDFourier, 'twoDFourier', 'cumulativeTransformAndAverage', '2D Fourier (cumulative)')
	wrap(calculateQCurves, 'calculateQCurves', 'calculateQCurves', 'Q Curves')
	wrap(calculateQCurves, 'calculateQCurves', 'calculateWithCalls', 'Q Curves (c with c)')
	wrap(calculateCorrelation, 'calculateCorrelation', 'calculateCorrelation', 'Correlation')

"""
Calls the main function with wrapped module methods
"""
def testTiming(probe, filepath, maxF):
	# Run the main program
	exec(f'quickDDM.processingCore.{probe}("{filepath}", range(1, {maxF}))')

"""
Prints the stored times for the looped-over functions
"""
def printResults(order, times):
	for k in order:
		print(f'{k: <25}: {times[k]:.5}')



if __name__ == '__main__':
	# You'll need to copy or link your choice of file for this to run on
	filepath = './tests/data/timing.avi'
	if not os.path.isfile(filepath):
		err = '\n\nCopy a file to time against to ./tests/data/timing.avi'
		raise FileNotFoundError(err)

	# List of functions to time
	mains = [
		'transformFirstMain',
		'differenceFirstMain',
		'cumulativeDifferenceMain',
		'sequentialChunkerMain'
	]

	print('All times given in seconds')

	# Store times in a dict. This way it's interoperable with the loops
	# in processingCore, as well as order agnostic
	times = defaultdict(float)
	order = []

	# Wrap module functions in timers and inject into system cache
	createWrappers(order, times)
	# Then import the processors
	import quickDDM.processingCore

	# Blindly open the video file and read the frame count
	videoFile = cv2.VideoCapture(filepath)
	frames = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))

	for f in mains:
		try:
			times.clear() # delete records from last run-through
			print(f'  Timing {f}()')
			a = time()

			testTiming(f, filepath, frames)

			b = time()
			printResults(order, times) # Output results
			print(f'Total Time: {b-a:.5}') # And total time taken
		except AttributeError: # the exec() borked
			print(f'Invalid function to test, "{f}"')
			continue