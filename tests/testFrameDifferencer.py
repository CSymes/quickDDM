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
import numpy

class FrameDifferenceTestCases(unittest.TestCase):
    def testSubtractingIdenticalFrames(self):
        frames = readVideo('tests/data/black.avi')
        diff = frameDifferencer(frames, 1) # subtract every set of frames with a distance of 1
        # Should be 4 frames of 0's, as all `frames` are identical

        gold = [[[0 for i in range(1024)] for j in range(1024)] for k in range(5 - 1)]

        self.assertTrue(numpy.array_equal(diff, gold)) # Check first subtraction (first two frames) is zero

    @unittest.skip("Fails - only calculates the first frame diff per spacing for now")
    def testSubtractingUniqueFrames(self):
        # consists of a frame of white followed by a 70% grey frame
        frames = readVideo('tests/data/alternating.avi')
        diff = frameDifferencer(frames, 1)

        gold = [[[int(255*0.3) for i in range(1024)] for j in range(1024)] for k in range(5 - 1)]

        self.assertTrue(numpy.array_equal(diff, gold)) # compare first subtraction

    @unittest.skip("Need to rework spacings")
    def testCorrectNumberOfSpacingsCalculated(self):
        frames = readVideo('tests/data/black.avi')
        diff = frameDifferencer(frames, [1, 2, 3, 4, 5])

        self.assertEqual([4, 3, 2, 1, 0], [len(d) for d in diff])

    def testOverlyWideSpacingFails(self):
        frames = readVideo('tests/data/black.avi')
        diff = frameDifferencer(frames, 10)

        # Behaviour unspecified for the moment - can either return an empty set of subtractions
        # or maybe throw an error? TODO - need to decide whether the function should prevent this
        # itself or expect the calling function to prevent it
        self.assertEqual(len(diff), 0)

    def testSingleFrameVideoGetsNoSoup(self):
        frames = readVideo('tests/data/short.avi')
        diff = frameDifferencer(frames, 1)
        
        self.assertEqual(len(diff), 0)
