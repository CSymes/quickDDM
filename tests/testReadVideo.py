# -*- coding: utf-8 -*-
#testReadVideo.py
"""
Created 2019-04-04
Series of tests for the video reading/frame splitting functionality
@author: Cary
"""

import unittest
import cv2
import numpy
import contextlib, io
from quickDDM.readVideo import readVideo

class ReadVideoTestCases(unittest.TestCase):
    def testValidVideoOpens(self):
        # TODO maybe split readVideo into readVideo(IntoFrames) and read(Single)Frame?
        readVideo('tests/data/short.avi') # single black frame

    def testReadFramesAreCorrect(self):
        frames = readVideo('tests/data/black.avi') # 5x frames of straight black
        self.assertEqual(len(frames), 5)

        gold = [[[0 for i in range(1024)] for j in range(1024)] for k in range(5)]
        self.assertTrue(numpy.array_equal(frames, gold))

    def testCorruptedVideoFails(self):
        # Unfortunately this invalid read creates a semi-unblockable write to stderr
        # Workaround: https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
        # Just take a lot of extra code, probably not worth it
        with self.assertRaises(cv2.error):
            print('\nExpecting ioctl error... ', end='')
            readVideo('tests/data/corrupt.avi')
            # corrupt.avi is literally just 1MB of /dev/random

    @unittest.skip("Unsure how to mock this for now")
    def testBadPermissionsReadFails(self):
        self.fail("Unable to read")

    def testEmptyVideoFails(self):
        with self.assertRaises(cv2.error):
            print('\nExpecting ioctl error... ', end='')
            frames = readVideo('tests/data/empty.avi')
