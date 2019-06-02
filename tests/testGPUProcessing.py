# -*- coding: utf-8 -*-
#testGPUProcessing.py
"""
Created 2019-05-19
Quick tests for checking that GPU processing results match
@author: Cary
"""

from pyopencl._cl import LogicError
import reikna
from reikna.fft import FFT, FFTShift
from reikna.transformations import norm_const, div_const

from quickDDM.readVideo import readVideo
from quickDDM.twoDFourier import twoDFourierUnnormalized, castToReal

import numpy, numpy.testing
import unittest
import sys

class GPUTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Cache test data
        self.frames = readVideo('tests/data/small.avi').astype(numpy.int16)
        self.firstDiff = self.frames[1] - self.frames[0]

        api = reikna.cluda.ocl_api()
        self.thread = api.Thread.create()

        # One array style for the complex data (FFT out) and floats for post-normalisation
        footprint = self.thread.array(self.frames[0].shape, dtype=numpy.complex)
        footprint_out = self.thread.array(self.frames[0].shape, dtype=numpy.float)

        self.fft = FFT(footprint).compile(self.thread) # FFT Computation object

        fftshift = FFTShift(footprint)
        div = div_const(footprint, numpy.sqrt(numpy.prod(self.frames[0].shape))) # divide by frame size
        norm = norm_const(footprint, 2) # abs (reduce to real mag.) and square

        # attach transformations to fftshift computation
        fftshift.parameter.output.connect(div, div.input, output_prime=div.output)
        fftshift.parameter.output_prime.connect(norm, norm.input, output_prime_2=norm.output)

        self.normalise = fftshift.compile(self.thread) # Compile FFTShift with normalisation Transformations

    def testSimpleFourierMatchesCPU(self):
        devFr = self.thread.to_device(self.firstDiff.astype(numpy.complex))
        self.fft(devFr, devFr)

        local = devFr.get()
        cpu = numpy.fft.fft2(self.firstDiff)

        numpy.testing.assert_allclose(local, cpu)

    def testFourierWithNormalisationMatchesCPU(self):
        res = self.thread.array(self.frames[0].shape, dtype=numpy.float64)
        devFr = self.thread.to_device(self.firstDiff.astype(numpy.complex))
        self.fft(devFr, devFr)
        self.normalise(res, devFr)

        local = res.get()

        ftframe = numpy.asarray([self.firstDiff])
        cpu = castToReal(twoDFourierUnnormalized(ftframe)[0])

        numpy.testing.assert_allclose(local, cpu)

    def testFourierWithNormalisationMatchesMatlab(self):
        res = self.thread.array(self.frames[0].shape, dtype=numpy.float64)
        devFr = self.thread.to_device(self.firstDiff.astype(numpy.complex))
        self.fft(devFr, devFr)
        self.normalise(res, devFr)

        local = res.get()

        with open('tests/data/fft_matlab_f2-f1.csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')

            numpy.testing.assert_allclose(local, m)
