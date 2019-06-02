# -*- coding: utf-8 -*-
#testFourierTransforms.py
"""
Created 2019-04-06
Series of tests for the 2D Fourier calculation functionality
@author: Cary
"""

from quickDDM.readVideo import readVideo
from quickDDM.twoDFourier import twoDFourierUnnormalized as fft2
from quickDDM.twoDFourier import castToReal as n_fft2

import numpy, numpy.testing
import unittest
import cv2

class FourierTransformTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # a (10-frame) 128x128 crop of the large data file
        self.frames = readVideo('tests/data/small.avi').astype(numpy.int16)
        self.firstDiff = self.frames[1] - self.frames[0]


    def testBasicFourier(self):
        frame = self.frames[0:1]
        ft = fft2(frame)[0]
        q = n_fft2(ft)

        with open('tests/data/fft_matlab_f1.csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')

            # Need to be more permissive to account for the extreme centre of
            # the transform not being negated by the subtraction
            numpy.testing.assert_allclose(q, m, rtol=1e-3)

    def testSubtractThenFourier(self):
        frame = numpy.asarray([self.firstDiff])
        ft = fft2(frame)[0]
        q = n_fft2(ft)

        with open('tests/data/fft_matlab_f(2-1).csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')
            # print('Total difference:', sum(sum(abs(q-m))))

            # Test equality, allowing for small differences in FFT calcs
            numpy.testing.assert_allclose(q, m, rtol=1e-3)

    def testFourierThenSubtract(self):
        frames = self.frames[0:2]
        ft = fft2(frames)
        q = n_fft2(ft[1] - ft[0])

        with open('tests/data/fft_matlab_f2-f1.csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')

            numpy.testing.assert_allclose(q, m, rtol=1e-3)

    def testFFTLinearity(self):
        aft = fft2(numpy.asarray([self.firstDiff]))[0]
        a = n_fft2(aft)

        bft = fft2(self.frames[0:2])
        b = n_fft2(bft[1] - bft[0])

        # Order should be very nearly immaterial
        numpy.testing.assert_allclose(a, b, rtol=1e-12)
