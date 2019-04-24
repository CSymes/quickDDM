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
import unittest
import sys
import os

class Timing(unittest.TestCase):
	def testTiming(self):
		# You'll need to copy your choice of file for this to run on
		# I use 10frames.avi atm, in future it'll be larger source videos
		filepath = 'tests/data/timing.avi'

		if not os.path.isfile(filepath):
			err = '\n\nCopy a file to time against to ./tests/data/time.avi'
			self.fail(err)

		print('All times given in seconds')
		# Blank object to store timing in
		# This way it's interoperable with the loop in basicMain
		times = type('', (), {})()

		# TODO make a way to genericly perform this wrapping

		#==========================================================#
		r_rv = readVideo.readVideo # store reference to original function (real)
		def m_rv(*args): # create a wrapper func (mocked)
			startTime = time() # take initial timestamp
			r = r_rv(*args) # call the real readVideo with all given args
			times.read = time() - startTime # store total time consumed
			print(f'readVideo:                {times.read:.5}') # print it as well
			return r # transparently return
		readVideo.readVideo = m_rv # overwrite the function in readVideo.py
		sys.modules['readVideo'] = readVideo # and pre-"import" it for basicMain.py
		# this is important, since basicMain doesn't know it's in the quickDDM namespace
		#==========================================================#



		# predefine cumulative counters
		times.differencer = 0; times.fourier = 0; times.qcurves = 0;
		#==========================================================#
		r_fd = frameDifferencer.frameDifferencer
		def m_fd(*args):
			startTime = time()
			r = r_fd(*args)
			times.differencer += time() - startTime # ADD the time
			return r
		frameDifferencer.frameDifferencer = m_fd
		sys.modules['frameDifferencer'] = frameDifferencer
		#==========================================================#
		r_2df = twoDFourier.twoDFourier
		def m_2df(*args):
			startTime = time()
			r = r_2df(*args)
			times.fourier += time() - startTime
			return r
		twoDFourier.twoDFourier = m_2df
		sys.modules['twoDFourier'] = twoDFourier
		#==========================================================#
		r_qc = calculateQCurves.calculateQCurves
		def m_qc(*args):
			startTime = time()
			r = r_qc(*args)
			times.qcurves += time() - startTime
			return r
		calculateQCurves.calculateQCurves = m_qc
		sys.modules['calculateQCurves'] = calculateQCurves
		#==========================================================#
		def printCumulativeResults():
			print(f'frameDifferencer (total): {times.differencer:.5}')
			print(f'twoDFourier (total):      {times.fourier:.5}')
			print(f'calculateQCurves (total): {times.qcurves:.5}')



		#==========================================================#
		r_cc = calculateCorrelation.calculateCorrelation
		def m_cc(*args):
			printCumulativeResults() # Output results from timing the previous modules

			startTime = time()
			r = r_cc(*args)
			times.correlate = time() - startTime
			print(f'calculateCorrelation:     {times.correlate:.5}')
			return r
		calculateCorrelation.calculateCorrelation = m_cc
		sys.modules['calculateCorrelation'] = calculateCorrelation
		#==========================================================#
		


		# Run the main program				
		sys.argv = ['basicMain.py', filepath]
		import quickDDM.basicMain

		print('') # Drop test counter onto its own line