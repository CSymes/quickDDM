# -*- coding: utf-8 -*-
#testReadVideo.py
"""
Created 2019-04-04
Series of tests for the video reading/frame splitting functionality
@author: Cary
"""

from quickDDM.readVideo import readVideo, readFramerate
import os, sys, shutil
import numpy
import unittest

"""
Suite of unit tests to run against quickddm.readVideo
"""
class ReadVideoTestCases(unittest.TestCase):
    def testValidVideoOpens(self):
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
        # TODO think about implementing it anyway
        with self.assertRaises(OSError):
            print('\nExpecting IO error... ', end='')
            readVideo('tests/data/corrupt.avi')
            # corrupt.avi is literally just 1MB of /dev/random

    def testBadPermissionsReadFails(self):
        if sys.platform == 'win32':
            return # Windows doesn't do this perms stuff, so I guess this can't fail?

        fpat = '/tmp/quickDDM-noread.avi'

        # copy valid file to /tmp if it's not still there from a previous run of the tests
        if not os.path.isfile(fpat):
            shutil.copyfile('tests/data/black.avi', fpat)

        os.chmod(fpat, 0o000) # Set permissions so it's no longer readable
        self.assertRaises(OSError, readVideo, fpat)


    def testEmptyVideoFails(self):
        with self.assertRaises(OSError):
            print('\nExpecting ioctl error... ', end='')
            frames = readVideo('tests/data/empty.avi')

    def testReadFramerate(self):
        fps = readFramerate('tests/data/black.avi')

        self.assertEqual(fps, 100)
