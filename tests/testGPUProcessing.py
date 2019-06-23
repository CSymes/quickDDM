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
from quickDDM.gpuCore import (createComplexFFTKernel,
    createNormalisationKernel, runKernelOperation)

import numpy, numpy.testing
import unittest
import sys

class GPUTestCases(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Cache test data
        self.frames = readVideo('tests/data/small.avi').astype(numpy.int16)
        self.firstDiff = self.frames[1] - self.frames[0]

        self.thread = reikna.cluda.ocl_api().Thread.create()

        # Get the OpenCL kernels from the 2DF module
        self.fft = createComplexFFTKernel(self.thread, self.frames[0].shape)
        self.normalise = createNormalisationKernel(self.thread, self.frames[0].shape)

    def testSimpleFourierMatchesCPU(self):
        local = runKernelOperation(self.thread, self.fft, self.firstDiff).get()

        # Need to un-shift since twoDFourierUnnormalized does it already, but
        # the OCL kernel doesn't
        ftframe = numpy.asarray([self.firstDiff])
        cpu = numpy.fft.fftshift(twoDFourierUnnormalized(ftframe)[0])

        numpy.testing.assert_allclose(local, cpu)

    def testFourierWithNormalisationMatchesCPU(self):
        local = runKernelOperation(self.thread, self.fft, self.firstDiff)
        local = runKernelOperation(self.thread, self.normalise, local,
            outType=numpy.float64).get()

        ftframe = numpy.asarray([self.firstDiff])
        cpu = castToReal(twoDFourierUnnormalized(ftframe)[0])

        numpy.testing.assert_allclose(local, cpu)

    def testFourierWithNormalisationMatchesMatlab(self):
        local = runKernelOperation(self.thread, self.fft, self.firstDiff)
        local = runKernelOperation(self.thread, self.normalise, local,
            outType=numpy.float64).get()

        with open('tests/data/fft_matlab_f2-f1.csv', 'rb') as mf:
            m = numpy.loadtxt(mf, delimiter=',')

            numpy.testing.assert_allclose(local, m)
