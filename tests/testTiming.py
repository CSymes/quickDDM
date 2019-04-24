# -*- coding: utf-8 -*-
#testTiming.py
"""
Created 2019-04-24
Runs a bunch of timers over sections of the main program for comparison purposes
@author: Cary
"""

from quickDDM.readVideo import readVideo
from quickDDM.frameDifferencer import frameDifferencer
from quickDDM.twoDFourier import twoDFourier
from quickDDM.calculateQCurves import calculateQCurves
from quickDDM.calculateCorrelation import calculateCorrelation
import unittest
import numpy

class Timing(unittest.TestCase):
	def testTiming(self):
		# You'll need to set your choice of file for this to run on
		# I use 10frames.avi atm, in future it'll be larger source videos
		filepath = 'tests/data/timing.avi'
		print('') # Drop test counter onto its own line