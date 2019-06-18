#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#gpuCore.py

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

from collections import deque

from .readVideo import readVideo, readFramerate
from .calculateQCurves import calculateWithCalls
from .calculateCorrelation import calculateCorrelation



"""
Compiles a kernel to calculate the 2D FFT of a frame
Also divides by scaling factor, but doesn't move to real domain or fftshift
"""
def createComplexFFTKernel(thread, shape):
    scaling = numpy.sqrt(shape[-2] * shape[-1])
    footprint = thread.array(shape, dtype=numpy.complex128)
    fft = FFT(footprint)

    div = div_const(footprint, scaling)
    fft.parameter.output.connect(div, div.input, output_prime=div.output)

    return fft.compile(thread)

"""
Creates a kernel to make a frame real, as well as fftshift
"""
def createNormalisationKernel(thread, shape):
    footprint = thread.array(shape, dtype=numpy.complex)
    fftshift = FFTShift(footprint)

    norm = norm_const(footprint, 2)
    fftshift.parameter.output.connect(norm, norm.input, output_prime=norm.output)

    normalise = fftshift.compile(thread)
    return normalise


"""
Runs a Reikna kernel over a data frame and gives the result (in VRAM)

Parameters:
    thread: Reikna CLUDA thread object
    kernel: compiled GPU kernel - assumed to have compiled signature (out, in)
    frame: data to operate on
Returns:
    On-GPU buffer containing the results (same dimensions as `frame`)
"""
def runKernelOperation(thread, kernel, frame):
    frame = numpy.ascontiguousarray(frame).astype(numpy.complex128)

    devFr = thread.to_device(frame) # Send frame to device
    fBuffer = thread.array(frame.shape, dtype=numpy.complex128)

    kernel(fBuffer, devFr)
    return fBuffer



# TODO - Process more stages on GPU

def sequentialGPUChunker(filename, spacings, RAMGB = 4, progress=None, abortFlag=None):
    if progress is not None:
        progress.setText('Reading Video from Disk')
        progress.cycle()

    videoInput = readVideo(filename)
    numFrames = videoInput.shape[0]
    correlations = [None] * (numFrames - 1)

    if abortFlag: return None



    showProgress = progress is not None
    if showProgress: progress.setText('Getting OpenCL Context')

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

    if showProgress: progress.setText('Creating OpenCL Kernels')

    # need to compile an OpenCL kernel to calculate FFTs with
    size = [d-1 for d in videoInput[0].shape]
    fftComplex = createComplexFFTKernel(thr, size)
    fftNorm = createNormalisationKernel(thr, size)



    # Number of pixels per frame, multiplied by 128 for the size of a complex
    # float, but halved because the real transform is used
    RAMBytes = RAMGB * numpy.power(2.0, 30.0)
    complexFrameByteSize = videoInput.shape[1] * videoInput.shape[2] * 128 / 2

    # One frame's RAM in reserve for the head
    framesPerSlice = int((RAMBytes // complexFrameByteSize) - 1)

    # The number of different slice intervals that must be taken
    numSpacingSets = int(numpy.ceil((numFrames -1) / framesPerSlice))

    # Used to show progress in the UI
    framesProcessed = 0
    target = numFrames * (numFrames - 1) / 2 # algorithm complexity
    # Allow 10% extra time to calculate the q curves
    qProgress = target * 0.1 / numSpacingSets # per-slice q-curve allowance
    target += qProgress * numSpacingSets

    if abortFlag: return None

    # For each diagonal section
    for sliceSpacing in range(0, numSpacingSets):
        if progress is not None:
            progress.setText(f'Working on Slice {sliceSpacing+1}/{numSpacingSets}')

        #A double ended queue, more efficient than a list for queue operations
        currentSlice = deque()
        #The index by which new frames are grabbed for the slice
        baseIndex = 0
        #Finding the expected shape of the transform results



        transformShape = (videoInput.shape[1] - 1, videoInput.shape[2] - 1)
        totalDifferencesShape = (framesPerSlice, transformShape[0], transformShape[1])
        #Preparing the destination of the frame differences
        totalDifferences = numpy.zeros(totalDifferencesShape)
        numDifferences = numpy.zeros((framesPerSlice,))

        #For each head
        for headIndex in range((sliceSpacing * framesPerSlice) + 1, numFrames):
            #If the queue is full, remove the oldest element
            if len(currentSlice) == framesPerSlice:
                currentSlice.popleft()
            #Get a new value into the slice queue
            #Also drops a row and column
            currentSlice.append(runKernelOperation(thr, fftComplex, videoInput[baseIndex, :-1, :-1]))
            baseIndex += 1
            #Drops a row and column
            head = videoInput[headIndex, :-1, :-1]
            head = runKernelOperation(thr, fftComplex, head)

            #time difference between this frame and the first in the queue
            relativeDifference = 0
            #iterating backwards through the list, oldest element first
            for sliceFrameIndex in range(len(currentSlice) - 1, -1, -1):
                # Update progress tracker
                if progress is not None:
                    framesProcessed += 1
                    progress.setProgress(framesProcessed, target)

                difference = head - currentSlice[sliceFrameIndex]
                normalFrame = thr.array(size, dtype=numpy.float64)
                fftNorm(normalFrame, difference)

                totalDifferences[relativeDifference, :, :] += normalFrame.get()

                # TODO - Need to make this ^^^ run on the GPU
                # allocate list of empty buffers, add into?

                numDifferences[relativeDifference] += 1
                relativeDifference += 1

                if abortFlag: return None

        for relativeDifference in range(0, len(currentSlice)):
            if progress is not None:
                framesProcessed += qProgress / len(currentSlice)
                progress.setProgress(framesProcessed, target)

            meanDifference = (totalDifferences[relativeDifference, :, :] / numDifferences[relativeDifference])
            timeDifference = relativeDifference + sliceSpacing * framesPerSlice
            correlations[timeDifference] = calculateWithCalls(meanDifference)

            if abortFlag: return None

    if progress is not None:
        progress.cycle()
        progress.setText('Calculating Correlation Curves')

    correlations = calculateCorrelation(correlations)


    frameRate = readFramerate(filename)
    timeSpacings = numpy.array(numpy.arange(1, len(correlations) + 1)) / frameRate
    # This is how you stack arrays in numpy, apparently 🙃
    outputMatrix = numpy.c_[timeSpacings, correlations]

    if abortFlag: return None

    if progress is not None:
        progress.setPercentage(100)
        progress.setText('Done!')

    return outputMatrix



if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Invalid args')
        exit()

    sequentialGPUChunker(sys.argv[1])
