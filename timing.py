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
Inserts a bunch of timing calls into basicMain.py and outputs the results
"""
def testTiming():
	# You'll need to copy or link your choice of file for this to run on
	filepath = 'tests/data/timing.avi'

	if not os.path.isfile(filepath):
		err = '\n\nCopy a file to time against to ./tests/data/time.avi'
		raise FileNotFoundError(err)

	print('All times given in seconds')
	# Store times in a dict. This way it's interoperable with the loop in basicMain
	times = defaultdict(float)



	# TODO make a way to genericly perform this wrapping

	#==========================================================#
	r_rv = readVideo.readVideo # store reference to original function (real)
	def m_rv(*args): # create a wrapper func (mocked)
		startTime = time() # take initial timestamp
		r = r_rv(*args) # call the real readVideo with all given args
		times['read'] = time() - startTime # store total time consumed
		print(f'readVideo:                {times["read"]:.5}') # print it as well
		return r # transparently return
	readVideo.readVideo = m_rv # overwrite the function in readVideo.py
	sys.modules['readVideo'] = readVideo # and pre-"import" it for basicMain.py
	# this is important, since basicMain doesn't know it's in the quickDDM namespace
	#==========================================================#



	#==========================================================#
	r_fd = frameDifferencer.frameDifferencer
	def m_fd(*args):
		startTime = time()
		r = r_fd(*args)
		times['differencer'] += time() - startTime # ADD the time
		return r
	frameDifferencer.frameDifferencer = m_fd
	sys.modules['frameDifferencer'] = frameDifferencer
	#==========================================================#
	r_2df = twoDFourier.twoDFourier
	def m_2df(*args):
		startTime = time()
		r = r_2df(*args)
		times['fourier'] += time() - startTime
		return r
	twoDFourier.twoDFourier = m_2df
	sys.modules['twoDFourier'] = twoDFourier
	#==========================================================#
	r_qc = calculateQCurves.calculateQCurves
	def m_qc(*args):
		startTime = time()
		r = r_qc(*args)
		times['qcurves'] += time() - startTime
		return r
	calculateQCurves.calculateQCurves = m_qc
	sys.modules['calculateQCurves'] = calculateQCurves
	#==========================================================#



	#==========================================================#
	r_cc = calculateCorrelation.calculateCorrelation
	def m_cc(*args):
		printCumulativeResults(times) # Output results from timing the previous modules

		startTime = time()
		r = r_cc(*args)
		times['correlate'] = time() - startTime
		print(f'calculateCorrelation:     {times["correlate"]:.5}')
		return r
	calculateCorrelation.calculateCorrelation = m_cc
	sys.modules['calculateCorrelation'] = calculateCorrelation
	#==========================================================#



	# Run the main program
	sys.argv = ['basicMain.py', filepath]
	import quickDDM.basicMain

"""
Prints the stored times for the looped-over functions
"""
def printCumulativeResults(times):
	print(f'frameDifferencer (total): {times["differencer"]:.5}')
	print(f'twoDFourier (total):      {times["fourier"]:.5}')
	print(f'calculateQCurves (total): {times["qcurves"]:.5}')



if __name__ == '__main__':
	testTiming()
