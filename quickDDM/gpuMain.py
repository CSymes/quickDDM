#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#gpuMain.py

"""
Main function, running on a GPU using Reikna/PyOpenCL
@author: Cary
@created 2019-04-27
"""

import sys
import numpy

from pyopencl._cl import LogicError
import reikna
from reikna.fft import FFT, FFTShift
from reikna.transformations import norm_const, div_const

from timeit import default_timer as time

from readVideo import readVideo
from frameDifferencer import frameDifferencer
from twoDFourier import twoDFourier
from calculateQCurves import calculateQCurves
from calculateCorrelation import calculateCorrelation

cpuCheck = False


def createFFTKernel(thread, shape):
    footprint = thread.array(shape, dtype=numpy.complex)
    fft = FFT(footprint).compile(thread)
    return fft

def createNormalisationKernel(thread, shape):
    footprint = thread.array(shape, dtype=numpy.complex)
    fftshift = FFTShift(footprint)

    div = div_const(footprint, numpy.sqrt(numpy.prod(shape)))
    norm = norm_const(footprint, 2)

    fftshift.parameter.output.connect(div, div.input, output_prime=div.output)
    fftshift.parameter.output_prime.connect(norm, norm.input, output_prime_2=norm.output)

    normalise = fftshift.compile(thread)
    return normalise


"""
TODO

 * Retain GPU mem references
 * Managed GPU mem properly

 * Process more on GPU
 * Formulate integration with UI (progress bars etc.)
"""

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print('Invalid args')
        exit()

    spacings = range(3)
    correlations = []

    frames = readVideo(sys.argv[1]) # Read frames in
    # rffts = [] # Store VRAM pointers
    ffts = [] # Store RAM pointers

    a = time()
    t_fft = 0
    t_get = 0

    # Create access node for OpenCL
    api = reikna.cluda.ocl_api()
    try:
        thr = api.Thread.create()
    except LogicError:
        print('No OpenCL-compatible devices (e.g. GPUs) found', file=sys.stderr)
        exit()

    # Display accessible devices
    for plat in api.get_platforms():
        for cld in plat.get_devices():
            print(f'Using {cld.name} with {cld.global_mem_size/1024**3:.1f}GB VRAM')
            # print('Has extensions:', cld.extensions)

    # need to compile an OpenCL kernel to calculate FFTs with
    fft = createFFTKernel(thr, frames[0].shape)
    # and one to shift and normalise
    normalise = createNormalisationKernel(thr, frames[0].shape)

    # Calculate and store FFT for each frame in global mem on the GPU
    for frame in frames:
        a = time()

        res = thr.array(frames[0].shape, dtype=numpy.float64)
        devFr = thr.to_device(frame) # Send frame to device

        fft(devFr, devFr) # find transform, store back into same memory
        normalise(res, devFr)

        b = time()
        t_fft += b-a

        # rffts.append(devFr) # keep transform in VRAM
        ffts.append(res.get()) # store transform in main RAM

        t_get += time()-b

    c = time()
    ffts = numpy.asarray(ffts, dtype=numpy.int16)
    d = time()

    print(f'FFT Time: {t_fft:.5f}')
    print(f'Copy Time: {t_get:.5f}')
    print(f'Conversion Time: {d-c:.5f}')

    if cpuCheck:
        # Perform the same operation on the CPU to compare time consumed
        tc1 = time()
        og_fft = numpy.fft.fftshift(numpy.fft.fft2(frames), axes = (1,2))
        tc2 = time()
        print(f'CPU FFT Time: {tc2-tc1:.5f}')


    d = time()
    for spacing in spacings:
        frameDifferences = frameDifferencer(ffts, spacing)
        fourierSections = twoDFourier(frameDifferences)
        qCurve = calculateQCurves(fourierSections)
        correlations.append(qCurve)
    correlations = calculateCorrelation(correlations)

    print(f'Other Time: {time()-d:.5}')
