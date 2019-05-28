# -*- coding: utf-8 -*-
#testFrameDifferencer.py
"""
Created 2019-04-04
Series of tests for the frame subtraction functionality
@author: Cary
"""

from quickDDM.readVideo import readVideo
from quickDDM.frameDifferencer import frameDifferencer
import unittest
import numpy, numpy.testing

class FrameDifferenceTestCases(unittest.TestCase):
    def testSubtractingIdenticalFrames(self):
        frames = readVideo('tests/data/black.avi')
        diff = frameDifferencer(frames, 1) # subtract every set of frames with a distance of 1
        # Should be 4 frames of 0's, as all `frames` are identical

        gold = [[[0 for i in range(1024)] for j in range(1024)] for k in range(5 - 1)]

        self.assertTrue(numpy.array_equal(diff, gold)) # Check first subtraction (first two frames) is zero

    def testSubtractingUniqueFrames(self):
        # consists of a frame of white followed by a 70% grey frame
        frames = readVideo('tests/data/alternating.avi')
        diff = frameDifferencer(frames, 1)

        gold = [[[int(255*0.3) for i in range(1024)] for j in range(1024)] for k in range(2-1)]

        self.assertTrue(numpy.array_equal(diff, gold)) # compare first subtraction

    def testCorrectNumberOfFramesReturned(self):
        frames = readVideo('tests/data/10frames.avi')
        correctNum = lambda delta: (len(frames)-delta)

        for delta in range(len(frames)):
            diff = frameDifferencer(frames, delta)
            self.assertEqual(len(diff), correctNum(delta))

    def testThisVsMatlab(self):
        frames = readVideo('tests/data/small.avi')
        diff = frameDifferencer(frames, 1)[0]

        with open('tests/data/diff_matlab_2-1.csv', 'rb') as fm:
            gold = numpy.loadtxt(fm, delimiter=',')
            numpy.testing.assert_array_equal(gold, diff)


    def testOverlyWideSpacingFails(self):
        frames = readVideo('tests/data/10frames.avi')
        self.assertRaises(ValueError, frameDifferencer, frames, 10)

    def testSingleFrameVideoGetsNoSoup(self):
        frames = readVideo('tests/data/short.avi')
        self.assertRaises(ValueError, frameDifferencer, frames, 1)
