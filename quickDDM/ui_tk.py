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

from tkinter import *
from tkinter.ttk import *
from tkinter import messagebox
from tkinter.filedialog import askopenfilename, asksaveasfilename

from PIL import Image
from PIL.ImageTk import PhotoImage

import cv2
import numpy

import concurrent.futures as futures
from queue import Queue

import resource
from timeit import default_timer as time


WINDOW_PADDING = 10 # pixels
# LoadFrame constants
ADDR_PLACEHOLDER = 'Choose a file...' # Placeholder text for filepath entry box
SCRUB_STEPS = 10 # previewable frames
PREVIEW_DIM = 256 # size of the preview image (pixels, square)
# ProcessingFrame constants
SAMPLE_DIM = 512

threadPool = futures.ThreadPoolExecutor(max_workers=10)
threadResults = []



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
            fps = videoFile.get(cv2.CAP_PROP_FPS)
            width = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
            height = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))

            # Update the labels in the UI that show the metadata
            self.FPS.set(f'{fps:.1f}')
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

            self.video_file = videoFile
            self.handleScroll('moveto', '0.0') # Activate scrollbar, show preview

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

        self.scrub.set('0.0', '1.0') # Reset/disable scrollbar
        self.img_preview.delete('all') # Delete the image preview
        if self.video_file: # if necessary
            self.video_file.release() # release the previous video stream
        self.video_file = None # and allow garbage collection

    """
    Click handler for analysis button
    Basically throws away the current window and creates a new Frame for the processing UI
    """
    def triggerAnalysis(self):
        win = self._nametowidget(self.winfo_parent())
        filename = self.address.get()

        self.video_file.release()
        self.destroy()
        # The old has passed away...

        # ...behold, the new has come!
        pframe = ProcessingFrame(win, filename)
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
        lMetadata.grid(row = 1, column=0,
                       rowspan=2, columnspan=3,
                       sticky=[E, W, N, S],
                       pady=5)
        if lMetadata:
            # Subframe to allow padding inside the border
            lMetaSubframe = Frame(lMetadata)
            lMetaSubframe.grid(row=0, column=0, padx=15, pady=10)

            # Descriptor labels
            mFPS = Label(lMetaSubframe, text='FPS: ')
            mFPS.grid(row = 0, column=0, sticky=E)
            mDim = Label(lMetaSubframe, text='Dimensions: ')
            mDim.grid(row = 1, column=0, sticky=E)
            mFrames = Label(lMetaSubframe, text='# of Frames: ')
            mFrames.grid(row = 2, column=0, sticky=E)
            mLength = Label(lMetaSubframe, text='Video Length: ')
            mLength.grid(row = 3, column=0, sticky=E)

            # Dynamic labels - get updated when video selected
            dFPS = Label(lMetaSubframe, textvariable=self.FPS)
            dFPS.grid(row = 0, column=1, sticky=W)
            dDim = Label(lMetaSubframe, textvariable=self.dim)
            dDim.grid(row = 1, column=1, sticky=W)
            dFrames = Label(lMetaSubframe, textvariable=self.numFrames)
            dFrames.grid(row = 2, column=1, sticky=W)
            dLength = Label(lMetaSubframe, textvariable=self.length)
            dLength.grid(row = 3, column=1, sticky=W)

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
    def __init__(self, parent, fname):
        super().__init__(parent, padding=WINDOW_PADDING)

        self.populate()
        center(parent) # Update window dimensions/placement

        self.correlation = None

        # TODO calc max delta
        maxDelta = 4
        deltaStep = 1

        # Begin processing the video
        startThread(self.beginAnalysis, fname, maxDelta, deltaStep)



    ### Processing


    """
    Begin the video analysis - see basicMain.py
    Threaded away from the UI
    """
    def beginAnalysis(self, fname, maxDelta, deltaStep):
        startTime = time()

        self.stage.set('Loading frames from disk')
        frames = readVideo(fname)
        print(f'frames: {len(frames)}')

        # list of frame deltas to analyse
        deltas = range(1, maxDelta+1, deltaStep)
        print(f'# of deltas: {len(deltas)}')
        curves = [] # store processed data

        # TODO test multiprocessing the deltas
        # https://stackoverflow.com/questions/659865/multiprocessing-sharing-a-large-read-only-object-between-processes
        # TODO test Fourier before delta
        # check shelving out if memory an issues
        # https://docs.python.org/3/library/shelve.html

        print(frames[0].shape)
        fours = numpy.array([twoDFourier(f) for f in frames])

        for i, spacing in enumerate(deltas):
            self.stage.set(f'Processing Δ {i+1}/{len(deltas)}')
            self.progress.set(100*i/len(deltas))
            print(f'\tdelta={spacing}')


            diff = frameDifferencer(fours, spacing)
            # four = twoDFourier(diff)
            q = calculateQCurves(diff)
            curves.append(q)


            self.results.insert('end', f'Δ = {spacing}')

            # Select this entry in the list if all other entries selected
            if (len(self.results.curselection()) == i):
                self.results.select_set(i)

        self.progress.set(100*(len(deltas)-0.5)/len(deltas))
        self.stage.set('Forming correlation function')
        self.correlation = calculateCorrelation(curves)
        print('cofunc:', self.correlation.shape)

        self.progress.set(100)
        self.stage.set('Done!')
        print('Done')

        self.deltas = deltas
        self.curves = curves

        #                   kb RAM, time taken
        # for a 10-frame vid
        #   naively:        410860, 4.77s
        #   fourier first:  353368, 2.04s
        # 100-frames
        #   naively:       4318268, 51.9s
        #   fourier first: 2683788, 13.6s
        print('ram:', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        print(f'time: {time() - startTime:.2f}')

    """Saves all current data to disk in a CSV (via a file selector)"""
    def saveAllData(self):
        if self.correlation is None:
            return

        filename = asksaveasfilename(initialdir = '.',
                                     title = 'Select save location',
                                     filetypes = (('Comma Seperated Values', '*.csv'),))
        if not filename:
            return # file selector aborted

        # Split IO onto its own thread
        def save(self, fn):
            numpy.savetxt(fn, self.correlation, delimiter='\t', fmt='%10.5f')
            print('Saved to '+ fn)

        startThread(save, self, filename)

    """Event handler for the list of deltas to update the displayed plots"""
    def updateGraphs(self, event):
        print([event.widget.get(i) for i in event.widget.curselection()])

        for i in event.widget.curselection():
            pass

    """
    Generates curve fitting for the current data
    Options for `model` are
        'motility'
        'brownian'
    """
    def curveFit(self, model):
        print(f'Fitting with assumption/{model}')
        # TODO



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

            # Yeah look I don't know why I need this, but otherwise there's a bunch of wasted space
            fProgress.columnconfigure(0, weight=10000)


        # Container
        fFitting = Frame(self)
        fFitting.grid(row=0, column=2, sticky=E)
        if fFitting:
            lFitting = Label(fFitting, text='Curve fitting, assuming: ')
            lFitting.grid(row=0, column=0)

            bBrownian = Button(fFitting, text='Brownian Motion', command=lambda: self.curveFit('brownian'))
            bDirectional = Button(fFitting, text='Directional Motion', command=lambda: self.curveFit('motility'))
            bSave = Button(fFitting, text='Save All to Disk...', command=self.saveAllData)

            bBrownian.grid(row=0, column=1)
            bDirectional.grid(row=0, column=2, padx=(WINDOW_PADDING, WINDOW_PADDING*4))
            bSave.grid(row=0, column=3)

        pResults = Listbox(self, selectmode='extended')
        pResults.grid(row=2, column=0, sticky=[E, W, N, S])
        pResults.bind('<<ListboxSelect>>', self.updateGraphs)
        self.results = pResults
        # TODO make a custom Listbox subclass and overhaul the selection system

        pCurves = Canvas(self, relief='solid', borderwidth=1)
        pCurves.configure(width=SAMPLE_DIM, height=SAMPLE_DIM)
        pCurves.grid(row=2, column=2)

        self.rowconfigure(1, minsize=WINDOW_PADDING) # buttons and list/canvas
        self.columnconfigure(0, weight=1, minsize=175) # allow for longer progress strings
        self.columnconfigure(1, minsize=WINDOW_PADDING) # Gap between left and right sections
        self.columnconfigure(2, weight=3)
        # TODO allow resizing?


def startThread(*args):
    future = threadPool.submit(*args)
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
    x = ((ww//2) - (w//2))
    y = ((wh//2) - (h//2))


    win.geometry(f'+{x}+{y}') # And set them
    # win.geometry(f'{w}x{h}+{x}+{y}') # W/H auto-calculated - don't need to force them

if __name__ == '__main__':
    # Create a new Tk framework instance / window
    window = Tk()

    loader = LoadFrame(window)
    loader.grid()


    center(window) # set window location
    window.winfo_toplevel().title("quickDDM")
    window.resizable(False, False) # It's not really super responsive

    with threadPool: # Ensure thread pool gets shutdown
        def checkThreads(): # periodic poll to check for threaded errors
            [done, ndone] = futures.wait(threadResults, timeout=0, return_when=futures.FIRST_COMPLETED)
            for future in done:
                print(f'Thread return: {future.result()}')
                threadResults.remove(future)

            window.after(100, checkThreads)
        window.after(100, checkThreads)

        window.mainloop() # hand control off to tkinter

# TODO Think about splitting this into multiple files
#      Maybe have a UI module instead of a UI *file*
