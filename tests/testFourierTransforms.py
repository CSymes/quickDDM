# -*- coding: utf-8 -*-
#testFourierTransforms.py
"""
Created 2019-04-06
Series of tests for the 2D Fourier calculation functionality
@author: Cary
"""

from quickDDM.readVideo import readVideo
from quickDDM.frameDifferencer import frameDifferencer
from quickDDM.twoDFourier import twoDFourier as fft, normaliseFourier as n_fft

import numpy, numpy.testing
import unittest
import cv2

class FourierTransformTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # a (10-frame) 128x128 crop of the large data file
        self.frames = readVideo('tests/data/small.avi')
        self.diff = frameDifferencer(self.frames, 1)

    def testBasicFourier(self):
        q = fft([self.frames[0]])[0]

        with open('tests/data/fft_matlab_f1.csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')

            # Need to be more permissive to account for the extreme centre of
            # the transform not being negated by the subtraction
            numpy.testing.assert_allclose(q, m, rtol=1e-3)

    def testSubtractThenFourier(self):
        q = fft([self.diff[0]])[0]

        with open('tests/data/fft_matlab_f(2-1).csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')
            # print('Total difference:', sum(sum(abs(q-m))))

            # Test equality, allowing for small differences in FFT calcs
            numpy.testing.assert_allclose(q, m, rtol=1e-3)

    def testFourierThenSubtract(self):
        f = fft(self.frames[0:2], normalise=False)
        q = n_fft(numpy.asarray([f[1] - f[0]]))[0]

        with open('tests/data/fft_matlab_f2-f1.csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')

            numpy.testing.assert_allclose(q, m, rtol=1e-3)

    def testFFTLinearity(self):
        a = fft([self.diff[0]])[0]

        f = fft(self.frames[0:2], normalise=False)
        b = n_fft(numpy.asarray([f[1] - f[0]]))[0]

        # Order should be very nearly immaterial
        numpy.testing.assert_allclose(a, b, rtol=1e-12)
