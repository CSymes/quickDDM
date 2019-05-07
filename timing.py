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
		sys.modules[smod] = module # and pre-"import" it for basicMain.py
		# this is important, since basicMain doesn't know it's in the quickDDM namespace
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
def testTiming(probe):
	# You'll need to copy or link your choice of file for this to run on
	filepath = 'tests/data/timing.avi'

	if not os.path.isfile(filepath):
		err = '\n\nCopy a file to time against to ./tests/data/time.avi'
		raise FileNotFoundError(err)

	import quickDDM.basicMain

	# Run the main program
	exec(f'quickDDM.basicMain.{probe}("{filepath}", range(1, 5))')

"""
Prints the stored times for the looped-over functions
"""
def printResults(order, times):
	for k in order:
		print(f'{k: <25}: {times[k]:.5}')



if __name__ == '__main__':
	mains = [
		'transformFirstMain',
		'differenceFirstMain',
		'cumulativeDifferenceMain'
	]

	print('All times given in seconds')

	# Store times in a dict. This way it's interoperable with the loops
	# in basicMain, as well as order agnostic
	times = defaultdict(float)
	order = []
	createWrappers(order, times)

	for f in mains:
		times.clear()
		print(f'  Timing {f}()')
		testTiming(f)
		printResults(order, times) # Output remaining results
