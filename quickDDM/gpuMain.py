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

import pyopencl as cl
import reikna.cluda as cluda
from reikna.fft import FFT
import arrayfire as af

from timeit import default_timer as time

from readVideo import readVideo
from frameDifferencer import frameDifferencer
from twoDFourier import twoDFourier
from calculateQCurves import calculateQCurves
from calculateCorrelation import calculateCorrelation

if __name__ == '__main__':
    spacings = (1, 2)
    correlations = []

    frames = readVideo(sys.argv[1]) # Read frames in
    print(frames.shape)
    # rffts = [] # Store VRAM pointers
    ffts = [] # Store RAM pointers

    a = time()
    t_fft = 0
    t_get = 0

    # Create access node for OpenCL
    api = cluda.ocl_api()
    thr = api.Thread.create()

    # Display accessible devices
    for plat in api.get_platforms():
        for cld in plat.get_devices():
            print(f'Using {cld.name} with {cld.global_mem_size/1024**3:.1f}GB VRAM')
            # print('Has extensions:', cld.extensions)

    # need to compile an OpenCL kernel to run FFTs with
    size = (len(frames[0]), len(frames[0][0]))
    fft = FFT(thr.array(size, dtype=complex)).compile(thr)

    # Calculate and store FFT for each frame in global mem on the GPU
    for frame in frames:
        a = time()

        devFr = thr.to_device(frame) # Send frame to device
        fft(devFr, devFr) # find transform, store back into same memory

        b = time()
        t_fft += b-a

        # rffts.append(devFr) # keep transform in VRAM
        ffts.append(devFr.get()) # store transform in main RAM

        t_get += time()-b

    c = time()
    ffts = numpy.asarray(ffts, dtype=numpy.int16)
    d = time()

    print(f'FFT Time: {t_fft:.5f}')
    print(f'Copy Time: {t_get:.5f}')
    print(f'Conversion Time: {d-c:.5f}')

    # Perform the same operation on the CPU to compare time consumed
    o1 = time()
    og_fft = numpy.fft.fftshift(numpy.fft.fft2(frames), axes = (1,2))
    o2 = time()
    print(f'CPU FFT Time: {o2-o1:.5f}')

    for spacing in spacings:
        break
        frameDifferences = frameDifferencer(ffts, spacing)
        fourierSections = twoDFourier(frameDifferences)
        qCurve = calculateQCurves(fourierSections)
        correlations.append(qCurve)
    correlations = calculateCorrelation(correlations)

    print(f'Other Time: {time()-o2:.5f}')
