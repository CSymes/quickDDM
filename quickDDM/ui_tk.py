#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#ui_tk.py
'''
Creates a UI to interface with the program, using the tkinter framework
@created: 2019-04-25
@author: Cary
'''

from readVideo import readVideo
from frameDifferencer import frameDifferencer
from twoDFourier import twoDFourier
from calculateQCurves import calculateQCurves
from calculateCorrelation import calculateCorrelation
from curveFitterBasic import fitCorrelationsToFunction, generateFittedCurves

from tkinter import *
from tkinter.ttk import Frame, Progressbar, Scrollbar, Entry
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.messagebox import askokcancel

from PIL import Image
from PIL.ImageTk import PhotoImage

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.pyplot import cm

import cv2
import numpy
import re
import random

# from pathos.multiprocessing import Pool as ProcPool
import concurrent.futures as futures
# from queue import Queue

# measure (peak) ram usage - 'nix only, sorry
try: import resource; rRam = True
except ModuleNotFoundError: rRam = False
from timeit import default_timer as time


WINDOW_PADDING = 10 # pixels
# LoadFrame constants
ADDR_PLACEHOLDER = 'Choose a file...' # Placeholder text for filepath entry box
SCRUB_STEPS = 10 # previewable frames
PREVIEW_DIM = 256 # size of the preview image (pixels, square)
DEFAULT_SCALING = 0.71 # default scale factor

# Intermodule/function constants
FITTING_BROWNIAN = 'rising exponential'
FITTING_DIRECTIONAL = 'TODO'
FITTING_NONE = ''

BACKEND_CPU = 'CPU'
BACKEND_GPU = 'GPU'
BACKEND_LOAD = 'FromDisk' # TODO frontend integration

# Processes vs Threads - essentially - bypasses the GIL vs ease of memory sharing
# stackoverflow.com/questions/3044580/multiprocessing-vs-threading-python
# processPool = ProcPool(processes=10)
threadPool = futures.ThreadPoolExecutor(max_workers=10)
threadResults = []
threadKill = False



class LoadFrame(Frame):
    def __init__(self, parent):
        super().__init__(parent, padding=WINDOW_PADDING)
        # prevent AttributeError when checking if it needs clearing before anything's been cleared
        self.video_file = None

        self.populateL() # video load/data elements
        self.columnconfigure(3, minsize=WINDOW_PADDING) # Gap between left and right sections
        self.populateR() # video preview elements



    ### Input Handlers


    """Click handler for the video selection button"""
    def triggerVideoSelect(self):
        # Open a file selection dialogue
        filename = askopenfilename(initialdir = '../tests/data',
                                   title = 'Select video file',
                                   filetypes = (('avi files', '*.avi'),
                                                ('all files','*.*')))
        if not filename:
            return # file selector aborted

        self.loadVideo(filename)


    """Select a video after being called by one of the above trigger methods"""
    def loadVideo(self, filename):
        # OpenCV sees much that is not clear to mere mortals...
        videoFile = cv2.VideoCapture(filename)

        # Successfully opened the video
        if videoFile.isOpened():
            # ...including all its metadata, thankfully
            frames = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(videoFile.get(cv2.CAP_PROP_FPS))
            width = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
            height = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.maxDelta = frames - 1

            # Update the labels in the UI that show the metadata
            self.FPS.set(f'{fps}')
            self.dim.set(f'{width}x{height} pixels')
            self.numFrames.set(f'{frames}')

            # Calculate the video length
            length = frames/fps
            lu = 's'
            if (length > 60*60): # on the order of hours
                length = length/60/60
                lu = 'hours'
            elif (length > 60*2): # more than 120 seconds long, might as well show as minutes
                length = length/60
                lu = 'min'
            self.length.set(f'{length:.2f} {lu}')

            self.address.set(filename) # update the displayed filename
            self.load_button['state'] = 'normal' # activate processing button

            self.spacing['state'] = 'normal' # activate metadata inputs
            self.scaling['state'] = 'normal'
            self.spacing.delete(0, 'end')
            self.scaling.delete(0, 'end')
            self.spacing.insert(0, self.maxDelta)
            self.spacingHelp.set(f' (max {self.maxDelta})')
            self.scaling.insert(0, DEFAULT_SCALING)

            self.video_file = videoFile
            self.handleScroll('moveto', '0.0') # Activate scrollbar, show preview

            loader.load_button.invoke() # TODO remove
            # Leave videoFile open for fetching of previews

        # File didn't open - not a video, corrupt, read error, or whatever else it could be
        else:
            self.clearPreviews() # filepath no longer valid - clear any previous preview data

            messagebox.showerror('File Error',
                                 'There was an issue with loading or accessing the '
                                 'specified video file. Double check it\'s a video and that '
                                 'you have the necessary permissions')
            videoFile.release() # Release anyway

    """Delete all metadata and image preview data"""
    def clearPreviews(self):
        # Metadata removal
        self.FPS.set('')
        self.dim.set('')
        self.numFrames.set('')
        self.length.set('')
        self.spacing.set('')
        self.spacingHelp.set('')
        self.scaling.set('')

        # Input/buttion disabling
        self.load_button['state'] = 'disabled'
        self.spacing['state'] = 'disabled'
        self.scaling['state'] = 'disabled'

        self.scrub.set('0.0', '1.0') # Reset/disable scrollbar
        self.img_preview.delete('all') # Delete the image preview
        if self.video_file: # if necessary
            self.video_file.release() # release the previous video stream
        self.video_file = None # and allow deallocation

    """
    Click handler for analysis button
    Basically throws away the current window and creates a new Frame for the processing UI
    """
    def triggerAnalysis(self):
        # Preserve values before destruction of this Frame
        win = self._nametowidget(self.winfo_parent())
        filename = self.address.get()
        spacing = self.spacing.get()
        scaling = self.scaling.get()
        fps = float(self.FPS.get())
        frames = int(self.numFrames.get())
        exp = self.deltaExponential.get()
        backend = self.backendChoice.get()

        # Final stage of validation for user inputs
        inputErrs = []
        # Check spacing > 0
        try:
            spacing = int(spacing)
            if spacing == 0: raise ValueError()
            if spacing >= int(self.numFrames.get()): raise ValueError()
        except ValueError:
            inputErrs.append('Invalid maximum spacing size')

        # Check scaling > 0
        try:
            scaling = float(scaling)
            if scaling == 0: raise ValueError()
        except ValueError:
            inputErrs.append('Invalid physical scale')

        # Show all errors
        if inputErrs:
            messagebox.showerror('Input Error', '\n'.join(inputErrs))
            return # Break until they're corrected

        self.video_file.release()
        self.destroy()
        # The old has passed away...

        # ...behold, the new has come!
        pframe = ProcessingFrame(win, backend, filename, fps, frames, spacing, scaling, exp)
        pframe.grid()

    """Rips the frame at `index` out of the chosen video and shows in the preview pane"""
    def setPreview(self, index):
        self.video_file.set(cv2.CAP_PROP_POS_FRAMES, index)
        status, frame = self.video_file.read()

        if status:
            thumSize = self.img_preview.winfo_width()
            pimg = Image.fromarray(frame) # convert from OpenCV to PIL
            pimg.thumbnail((thumSize, thumSize)) # downsize image
            tkimg = PhotoImage(image=pimg) # and convert to TK

            # Insert image onto the canvas
            self.img_preview.create_image((0, 0), image=tkimg, anchor='nw')

            # prevent garbage collection
            # see http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
            self.img_preview.img = tkimg
        else:
            print('Video could not be previewed - good luck processing it')

    """Deals with movement of the scrollbar for the preview pane"""
    def handleScroll(self, msg, x, units=''):
        if not self.numFrames.get():
            return # No scrolling until a video's been selected, mate

        x = float(x) # It's a string for some reason

        if msg == 'moveto': # Truck was dragged
            pass
        elif msg == 'scroll':
            if units == 'pages': # Scrollbar was clicked on
                pass
            elif units == 'units': # Arrow clicked on
                x = self.scrub.get()[0] + x/SCRUB_STEPS # budge the truck one index

        x = min(max(float(x), 0), 1-1/SCRUB_STEPS) # Bound its movement
        x = float(f'{x:.1f}') # Show discrete steps # this will be an issue for steps other than 0.1
        self.scrub.set(str(x), str(x+1/SCRUB_STEPS)) # Update scrollbar position

        # Note this will never show the final frame.
        self.setPreview(int(int(self.numFrames.get())*x))
        # TODO some sort of caching - this "wastes" a lot of CPU/IO redoing frames when scrubbing



    ### UI Configuration


    """Add elements to allow picking of a video file and for metadata preview"""
    def populateL(self):
        # StringVars for dynamic label updating
        self.address = StringVar(); self.address.set(ADDR_PLACEHOLDER)
        self.FPS = StringVar()
        self.dim = StringVar()
        self.numFrames = StringVar()
        self.length = StringVar()


        # file address label
        lAdd = Entry(self, textvariable=self.address)
        lAdd.grid(row=0, column=0, columnspan=2, sticky=[E, W], padx=(0, WINDOW_PADDING))
        # Juggling to clear/restore placeholder text
        lAdd.bind('<FocusIn>', lambda e: self.address.set('') if
                                    (self.address.get() == ADDR_PLACEHOLDER) else None)
        lAdd.bind('<FocusOut>', lambda e: (self.address.set(ADDR_PLACEHOLDER) or self.clearPreviews())
                  if (self.address.get() == '') else self.loadVideo(self.address.get()))

        # file chooser button
        lChoose = Button(self, text='Choose Source', command=self.triggerVideoSelect)
        lChoose.grid(row=0, column=2)

        # metadata pane - contains all metadata labels added in the below block
        lMetadata = Frame(self, borderwidth=1, relief='solid')
        lMetadata.grid(row=1, column=0,
                       rowspan=2, columnspan=3,
                       sticky=[E, W, N, S],
                       pady=5)
        if lMetadata:
            # Subframe to allow padding inside the border
            lMetaSubframe = Frame(lMetadata)
            lMetaSubframe.grid(row=0, column=0, padx=15, pady=10)

            # Descriptor labels
            mFPS = Label(lMetaSubframe, text='FPS: ')
            mFPS.grid(row=0, column=0, sticky=E)
            mDim = Label(lMetaSubframe, text='Dimensions: ')
            mDim.grid(row=1, column=0, sticky=E)
            mFrames = Label(lMetaSubframe, text='# of Frames: ')
            mFrames.grid(row=2, column=0, sticky=E)
            mLength = Label(lMetaSubframe, text='Video Length: ')
            mLength.grid(row=3, column=0, sticky=E)

            # Dynamic labels - get updated when video selected
            dFPS = Label(lMetaSubframe, textvariable=self.FPS)
            dFPS.grid(row=0, column=1, sticky=W)
            dDim = Label(lMetaSubframe, textvariable=self.dim)
            dDim.grid(row=1, column=1, sticky=W)
            dFrames = Label(lMetaSubframe, textvariable=self.numFrames)
            dFrames.grid(row=2, column=1, sticky=W)
            dLength = Label(lMetaSubframe, textvariable=self.length)
            dLength.grid(row=3, column=1, sticky=W)

            lMetaSubframe.rowconfigure(4, minsize=10)



            # Input validation
            chkSpace = self.register(lambda P: ((P.isdigit() and int(P) < int(self.numFrames.get())) or P == ""))
            chkScale = self.register(lambda P: (P.replace('.','', 1).isdigit() or P == ""))

            mlSpacing = Label(lMetaSubframe, text='Max. Spacing: ')
            mlSpacing.grid(row=5, column=0, sticky=E)
            mlScaling = Label(lMetaSubframe, text='Physical Scaling: ')
            mlScaling.grid(row=6, column=0, sticky=E)
            mlTime = Label(lMetaSubframe, text='Delta Spacing: ')
            mlTime.grid(row=7, column=0, sticky=E)

            fSpacing = Frame(lMetaSubframe)
            fSpacing.grid(row=5, column=1, sticky=W)
            self.spacing = Entry(fSpacing, width=5, validate='all', validatecommand=(chkSpace, '%P'))
            self.spacing.grid(row=0, column=0, sticky=W)
            self.spacingHelp = StringVar()
            fsHelp = Label(fSpacing, textvariable=self.spacingHelp)
            fsHelp.grid(row=0, column=1, sticky=W)
            self.scaling = Entry(lMetaSubframe, width=5, validate='all', validatecommand=(chkScale, '%P'))
            self.scaling.grid(row=6, column=1, sticky=W)

            self.spacing['state'] = 'disabled'
            self.scaling['state'] = 'disabled'
            self.deltaExponential = BooleanVar()

            fTimeSpacing = Frame(lMetaSubframe)
            fTimeSpacing.grid(row=7, column=1)
            timeLinear = Radiobutton(fTimeSpacing, text='Linear', value=False, variable=self.deltaExponential)
            timeLinear.grid(row=0, column=0, sticky=[E, W])
            timeExp = Radiobutton(fTimeSpacing, text='Exponential', value=True, variable=self.deltaExponential)
            timeExp.grid(row=0, column=1, sticky=[E, W])



            # Backend choice
            lMetaSubframe.rowconfigure(8, minsize=10)
            lProc = Label(lMetaSubframe, text='Processing Backend: ')
            lProc.grid(row=9, column=0, sticky=E)

            self.backendChoice = StringVar()

            fProc = Frame(lMetaSubframe)
            fProc.grid(row=9, column=1, sticky=[E, W])
            procCPU = Radiobutton(fProc, text='CPU', value=BACKEND_CPU, variable=self.backendChoice)
            procCPU.grid(row=0, column=0)
            procGPU = Radiobutton(fProc, text='GPU', value=BACKEND_GPU, variable=self.backendChoice)
            procGPU.grid(row=0, column=1)

            self.backendChoice.set(BACKEND_CPU) # Preselect CPU
            procCPU['width'] = len(timeLinear['text']) # Align radio buttons
            procCPU['anchor'] = W
            timeLinear['width'] = len(timeLinear['text'])
            timeLinear['anchor'] = W
            procGPU['state'] = 'disabled' # TODO

        # Button to progress to next stage
        lLoad = Button(self, text='Analyse Video', command=self.triggerAnalysis)
        lLoad.grid(row=4, column=1, pady=(WINDOW_PADDING, 0))
        lLoad['state'] = 'disabled'
        self.load_button = lLoad # store button for reactivation later

        # Favour whitespace in favour of stretching the metadata pane
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=10)
        # Equal width for space on both sides of the analysis button
        self.columnconfigure(0, minsize=lChoose.winfo_reqwidth())

    """Add UI elements for the video preview"""
    def populateR(self):
        # use a Canvas to draw the actual image
        preview = Canvas(self, relief='solid', borderwidth=1)
        preview.grid(row=0, column=4, rowspan=4, sticky=N)
        preview.configure(width=PREVIEW_DIM, height=PREVIEW_DIM)
        self.img_preview = preview # store the canvas so we can draw onto it later

        # Horizontal scrollbar to allow scrubbing through the video
        # This is kinda superfluous since we're looking at such small changes anyway 🙃
        scrub = Scrollbar(self, orient='horizontal', command=self.handleScroll)
        scrub.grid(row=4, column=4, sticky=[E, W])
        self.scrub = scrub



class ProcessingFrame(Frame):
    def __init__(self, parent, backend, fname, fps, numFrames, maxDelta, scalingFactor, exponentialSpacing):
        super().__init__(parent, padding=WINDOW_PADDING)

        self.populate()
        center(parent) # Update window dimensions/placement

        self.correlation = None
        self.plotCurves = []
        self.plotFits = []

        self.scalingFactor = scalingFactor
        self.exponentialSpacing = exponentialSpacing # this too
        self.fps = fps
        self.numFrames = numFrames

        self.fitting = ''
        self.fitCurves = None
        self.fitCache = {}

        # TODO exponential deltas
        deltaStep = 1
        # list of frame deltas to analyse
        self.deltas = range(1, maxDelta+1, deltaStep)

        backend = BACKEND_LOAD

        # Begin processing the video
        if backend == BACKEND_CPU: # TODO implement the chunking alg
            startThread(self.beginAnalysis_CPU, fname, self.deltas)
        elif backend == BACKEND_GPU:
            startThread(self.beginAnalysis_GPU, fname, self.deltas)
        elif backend == BACKEND_LOAD:
            startThread(self.loadResults, 'BulkResultChunker.txt')



    ### Processing


    def loadResults(self, fname):
        cold = numpy.loadtxt(fname)
        self.deltas = cold[:, 0]
        self.correlation = cold[:, 1:]

        for r in range(1, self.correlation.shape[1]):
            self.results.insert('end', f'q = {r}')

        self.fps = 1 / self.deltas[0]
        self.numFrames = self.deltas.shape[0] + 1

        self.progress.set(100)
        self.stage.set('Done!')

        return 'loading complete'

    """
    Begin the video analysis - see basicMain.py
    Threaded away from the UI, uses CPU
    """
    def beginAnalysis_CPU(self, fname, deltas):
        startTime = time()

        self.progressBar['mode'] = 'indeterminate'
        self.progressBar.start()
        self.stage.set('Loading frames from disk')
        frames = readVideo(fname)
        print(f'frames: {len(frames)}')

        print(f'# of deltas: {len(deltas)}')
        curves = numpy.zeros((len(deltas), len(frames[0])//2)) # store processed data

        if rRam: print('ram:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Configure UI display
        self.stage.set('Calculating Fouriers')
        self.progressBar.stop()
        self.progressBar['mode'] = 'determinate'
        self.progressBar['maximum'] = len(frames)
        self.progress.set(0)



        # Updates progressbar as Fourier transforms calculated
        def fourierWrapper(frame):
            # pass a single-item list of frames
            r = twoDFourier(numpy.asarray([frame])) # Find the transform for this frame
            self.progress.set(self.progress.get()+1)
            return r[0]

        # comprehension for all Fourier frames
        # calculating on a per-frame basis seems not to really be any more expensive
        a=time()
        fours = [fourierWrapper(f) for f in frames]
        print(f'fTime: {time()-a:.2f}')
        self.stage.set('Optimising Fourier container')
        if threadKill: return 'analysis aborted'

        a=time()
        if rRam: print('ram:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # TODO this is a hella expensive op. Maybe just use as a list instead?
        # if we don't multiprocess this, could just write straight into a blank ndarray
        fours = numpy.stack(fours) # Numpy-fy
        # stack() seems slightly faster than asarray()/asanyarray()
        if rRam: print('ram:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(f'sTime: {time()-a}')



        print(f'pre-diff-time: {time() - startTime:.2f}')
        self.progressBar['maximum'] = len(deltas)
        dTime=0
        qTime=0
        for i, spacing in enumerate(deltas):
            # Configure UI display
            self.stage.set(f'Processing δt {i+1}/{len(deltas)}')
            self.progress.set(i)


            # Calculate unique data for this frame spacing/delta
            a=time()          ###

            diff = frameDifferencer(fours, spacing)

            dTime += time()-a ###
            a=time()          ###

            q = calculateQCurves(diff)

            qTime += time()-a ###

            curves[i] = q

            if threadKill: return 'analysis aborted'

        for i in range(len(curves)):
            # Add to the UI listbox
            self.results.insert('end', f'q = {i}')
            # self.results.insert('end', f'Δ = {spacing}')
            # Select this entry in the list if all other entries selected
            if (len(self.results.curselection()) == i):
                self.results.select_set(i)

        print(f'dTime: {dTime:.2f}')
        print(f'qTime: {qTime:.2f}')


        self.progress.set(len(deltas)-0.5)
        self.stage.set('Forming correlation function')
        self.correlation = calculateCorrelation(curves)
        print(f'curves size: {curves.shape}')
        print(f'cofunc size: {self.correlation.shape}')

        self.progress.set(len(deltas))
        self.stage.set('Done!')

        if rRam: print('ram:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(f'total time: {time() - startTime:.2f}')

        return 'analysis complete'

    """
    Begin the video analysis - see gpuMain.py
    Threaded, plus uses GPU acceleration
    """
    def beginAnalysis_GPU(self, fname, deltas):
        pass # TODO

    """Saves all current data to disk in a CSV (via a file selector)"""
    def saveAllData(self): # TODO save fits
        if self.correlation is None:
            return

        filename = asksaveasfilename(initialdir = '.',
                                     title = 'Select save location',
                                     filetypes = (('Comma Seperated Values', '*.csv'),))
        if not filename:
            return # file selector aborted

        # Split IO onto its own thread
        def save(self, fn):
            target = self.correlation
            d = numpy.array([x/self.fps for x in self.deltas])
            target = numpy.hstack((d[:, None], target))

            numpy.savetxt(fn, target, delimiter=' ', fmt='%.5f')
            return f'saved {len(target)} rows to {fn}'

        startThread(save, self, filename)

    """Event handler for the list of deltas to update the displayed plots"""
    def updateGraphs(self, *_):
        # remove old plots
        while self.plotCurves:
            self.plotCurves.pop().remove()
        while self.plotFits:
            self.plotFits.pop().remove()
        self.mpl.relim()

        # and create the new ones
        for i in self.results.curselection():
            # Extract frame q value from the human-readable string
            d = int(re.search(r'(\d+)', self.results.get(i)).group(1))

            # Plot the q-vs-dt slice
            data = self.correlation[:, d]
            # Randomly choose a colour from the palette
            random.seed(d) # but make sure it's always the same for this curve, too
            colour = cm.viridis(random.random())

            curveRef = self.mpl.plot(range(len(data)), data, '-', color=colour)[0]
            self.plotCurves.append(curveRef)

            curveRef.set_label(f'q = {d}')

            if self.fitCurves is not None: # avoid "no handles" MPL warning
                fitData = self.fitCurves[d, :]
                fitRef = self.mpl.plot(range(len(fitData)), fitData, '--', color=colour)[0]
                self.plotFits.append(fitRef)


        if len(self.mpl.get_lines()):
            self.mpl.legend(loc='upper left', fontsize ='x-small')
        self.mpl.relim() # recalculate axis limits
        self.rerender() # push the new plots to the rasterised UI element

    """Spins off a thread to run the curve fitting on"""
    def triggerCurveFit(self):
        oldModel = self.fitting
        self.fitting = self.fitChoice.get()

        if self.correlation is None: # No data loaded yet
            # TODO it'd probably be good to disable buttons until this isn't the case
            self.fitChoice.set(FITTING_NONE)
            self.fitting = FITTING_NONE
            print('no available data to fit to')
            return

        if self.fitting != oldModel: # don't recalc if you press the button again
            # clear old D curve regardless of new fitting
            for artist in self.mpl_d.get_lines():
                artist.remove()
            leg = self.mpl_d.get_legend()
            if leg: leg.remove()

            if self.fitting == FITTING_NONE: # just remove D-curve and stop plotting fits
                print('Plotting without curve fitting')
                self.fitCurves = None
                self.rerender()
                return

            # show some text in mpl_d to show something's happening
            elevatorText = self.mpl_d.text(0.5, 0.5, 'Generating curve fitting...', horizontalalignment='center',
                                           verticalalignment='center', transform=self.mpl_d.transAxes)
            self.mpl_d.loaders.append(elevatorText)
            self.rerender()

            startThread(self.makeFits, self.fitting, callback=self.updateGraphs)

    """
    Generates curve fitting for the current data
    Options for `model` are as in curveFitterBasic.py
    """
    def makeFits(self, model):
        print(f'Fitting with assumption/{model}')
        qPoints = numpy.arange(1, self.correlation.shape[1]) # All valid values for q

        if model in self.fitCache: # This fitting's already been calculated
            self.fitCurves = self.fitCache[model][0] # pull curves back out
            D = self.fitCache[model][1] # as well as the diffusivity data

            msg = 'restored fitting for {model}'
        else:
            print('Calculating curve fitting over', qPoints.shape, 'points')
            corrQ = (2*numpy.pi*self.scalingFactor/((self.correlation.shape[1])*2)) # q-correction-factor for fitting

            # Find fitted equation paramaters
            fit = fitCorrelationsToFunction(self.correlation, qPoints, model, qCorrection=corrQ, timeSpacings=self.deltas)
            # and generate plots with that data
            self.fitCurves = generateFittedCurves(fit, qPoints, frameRate=self.fps, numFrames=self.numFrames, qCorrection=corrQ)
            # First curve doesn't get a fit, but good to preserve dimensions anyway, so insert a dummy row
            self.fitCurves = numpy.concatenate((numpy.zeros((1, self.fitCurves.shape[1])), self.fitCurves), 0)

            # extract diffusivity curve from fitting data too
            D = [seg[2] for seg in fit[0] if seg is not None]

            self.fitCache[model] = (self.fitCurves, D) # cache fitting data for reuse later, if necessary
            msg = f'{model} fitting complete'

        self.mpl_d.plot(qPoints, D, color=cm.tab10(0), label='Diffusivity') # plot diffusivity curve
        self.mpl_d.legend(fontsize ='x-small')

        # remove the 'calculating' text from mpl_d
        if self.mpl_d.loaders:
            self.mpl_d.loaders.pop().remove()

        return msg



    ### UI Configuration


    """Create UI elements for viewing/analysing the processed data"""
    def populate(self):
        # Container for the progress bar/label
        fProgress = Frame(self)#, relief='solid', borderwidth=1)
        fProgress.grid(row=0, column=0, sticky=[E, W])
        if fProgress:
            # Could put the label on top of the progressbar, but it has a non-transparent background
            # https://stackoverflow.com/questions/17039481/how-to-create-transparent-widgets-using-tkinter

            self.stage = StringVar()
            self.stage.set('Initialising Flux-nalysis Engine')
            l = Label(fProgress, textvariable=self.stage)
            l.grid(row=0, column=0)

            self.progress = IntVar()
            self.progress.set(0)
            pBar = Progressbar(fProgress, mode='determinate', variable=self.progress)
            pBar.grid(row=1, column=0, sticky=[E, W])
            self.progressBar = pBar

            # Yeah look I don't know why I need this, but otherwise there's a bunch of wasted space
            fProgress.columnconfigure(0, weight=10000)

        # Left pane containing all calculated options for q
        pResults = Listbox(self, selectmode='extended')
        pResults.grid(row=2, column=0, sticky=[E, W, N, S])
        pResults.bind('<<ListboxSelect>>', self.updateGraphs)
        self.results = pResults
        # TODO make a custom Listbox subclass and overhaul the selection system


        # Container for the buttons above the plotting figure
        fFitting = Frame(self)
        fFitting.grid(row=0, column=2, sticky=E)
        if fFitting:
            lFitting = Label(fFitting, text='Curve fitting, assuming: ')
            lFitting.grid(row=0, column=0)

            self.fitChoice = StringVar()
            bBrownian = Radiobutton(fFitting, text='Brownian Motion', indicatoron=0, variable=self.fitChoice,
                                    value=FITTING_BROWNIAN, command=self.triggerCurveFit)
            bDirectional = Radiobutton(fFitting, text='Directional Motion', indicatoron=0, variable=self.fitChoice,
                                    value=FITTING_DIRECTIONAL, command=self.triggerCurveFit)
            bNone = Radiobutton(fFitting, text='No Fitting', indicatoron=0, variable=self.fitChoice,
                                    value=FITTING_NONE, command=self.triggerCurveFit)

            bSave = Button(fFitting, text='Save All to Disk...', command=self.saveAllData)

            bBrownian.grid(row=0, column=1)
            bDirectional.grid(row=0, column=2, padx=(WINDOW_PADDING, WINDOW_PADDING))
            bNone.grid(row=0, column=3, padx=(WINDOW_PADDING, WINDOW_PADDING*4))
            bSave.grid(row=0, column=4)

            bDirectional['state'] = 'disabled' # TODO directional fitting

        # matplotlib figure on the right
        pFigure = Figure(figsize=(10, 6), dpi=100)
        pFigure.set_tight_layout(True) # reduce the huge default margins
        self.mpl = pFigure.add_subplot(2, 1, 1, xmargin=0, ymargin=0.1)
        self.mpl.set_xscale('log')
        self.mpl.set_ylabel(r'$\Delta(\delta t)$')
        self.mpl.set_xlabel(r'$\delta t\ (s)$')
        self.mpl_d = pFigure.add_subplot(2, 1, 2, xmargin=0, ymargin=0.1)
        self.mpl_d.set_ylabel(r'$D\ (\mu m^2 s^{-1})$')
        self.mpl_d.set_xlabel(r'$q\ (\mu m^{-1})$')
        self.mpl_d.loaders = [] # container for loading text for curve fitting

        pCanvas = FigureCanvasTkAgg(pFigure, master=self) # tkinter portal for the MPL figure
        pCanvas.draw()
        self.rerender = pCanvas.draw # expose for access elsewhere
        pCanvas.get_tk_widget().grid(row=2, column=2)
        self.pc = pCanvas


        self.rowconfigure(1, minsize=WINDOW_PADDING) # buttons and list/canvas
        self.columnconfigure(0, weight=1, minsize=175) # allow for longer progress strings
        self.columnconfigure(1, minsize=WINDOW_PADDING) # Gap between left and right sections
        self.columnconfigure(2, weight=3)

"""
Starts a thread on the ThreadPool.
arguments:
    threadFunc: function that runs in the thread
    *args: arguments to give to the threadFunc() call
    callback [optional]: function to be run on the main thread after execution finishes
"""
def startThread(threadFunc, *args, callback=None):
    future = threadPool.submit(threadFunc, *args)

    # Allow functions to be run on the main thread after execution of a thread
    # This is necessary since Tkinter isn't thread-safe and will crash if you reference it from another thread
    future.callback = callback
    threadResults.append(future)

"""Set the window's position"""
def center(win):
    # Screen width/height
    ww = win.winfo_screenwidth()
    wh = win.winfo_screenheight()


    win.geometry(f'+{ww*2}+{wh*2}') # move offscreen before rendering
    win.update() # generate geometry before moving window


    # Current window dimensions
    w = win.winfo_width()
    h = win.winfo_height()

    # and move back onscreen at correct position
    x = ((ww//2) - (w//4))
    y = ((wh//2) - (h//2))


    win.geometry(f'+{x}+{y}') # And set them
    # win.geometry(f'{w}x{h}+{x}+{y}') # W/H auto-calculated - don't need to force them

if __name__ == '__main__':
    # Create a new Tk framework instance / window
    window = Tk()

    loader = LoadFrame(window)
    loader.grid()

    center(window) # set window location and generate geometry
    window.winfo_toplevel().title('quickDDM')
    window.resizable(False, False) # It's not really super responsive

    # TODO delete lol
    loader.address.set('../tests/data/10frames.avi')
    for c in loader.winfo_children():
        c.event_generate('<FocusOut>', when='tail')
    # TODO

    with threadPool: # Ensure worker pools get shutdown
        def checkThreads(): # periodic poll to check for threaded errors
            # get any completed threads (without blocking)
            [done, ndone] = futures.wait(threadResults, timeout=0, return_when=futures.FIRST_COMPLETED)
            for future in done:
                print(f'Thread return: {future.result()}')
                threadResults.remove(future)

                # Run any post-thread stuff on the main thread - e.g. Tkinter draw calls
                if future.callback is not None:
                    future.callback()

            window.after(100, checkThreads) # schedule next check
        window.after(100, checkThreads)

        def onQuit():
            global threadKill

            threadKill = True
            if loader.winfo_exists(): # Only query if analysis has begun # TODO don't question it if results saved to disk?
                pass
                # if askokcancel('Abort?', 'Are you sure you want to abort analysis?', default='cancel'):
            window.destroy()

        window.protocol('WM_DELETE_WINDOW', onQuit)
        window.mainloop() # hand control off to tkinter

# TODO Think about splitting this into multiple files
#      Maybe have a UI subpackage instead of a UI module
