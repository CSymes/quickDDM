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
from tkinter.filedialog import askopenfilename
from PIL import ImageTk

import cv2



### Input Handlers

"""Click handler for the video selection button"""
def triggerVideoSelect(vars):
    # Open a file selection dialogue
    filename = askopenfilename(initialdir = '../tests/data', 
                               title = 'Select video file', 
                               filetypes = (('avi files', '*.avi'), 
                                            ('all files','*.*')))
    # OpenCV sees much that is not clear to mere mortals...
    videoFile = cv2.VideoCapture(filename)

    # Successfully opened the video
    if videoFile.isOpened():
        # ...including all its metadata, thankfully
        frames = int(videoFile.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = videoFile.get(cv2.CAP_PROP_FPS)
        width = int(videoFile.get(cv2.CAP_PROP_FRAME_HEIGHT))
        height = int(videoFile.get(cv2.CAP_PROP_FRAME_WIDTH))

        print(f'{frames}, {fps:.1f}, {width}, {height}')

        # Update the labels in the UI that show the metadata
        vars.FPS.set(f'{fps:.1f}')
        vars.dim.set(f'{width}x{height} pixels')
        vars.frames.set(f'{frames}')

        # Calculate the video length
        length = frames/fps
        lu = 's'
        if (length > 60*60): # on the order of hours
            length = length/60/60
            lu = 'hours'
        elif (length > 60*2): # more than 120 seconds long, might as well show as minutes
            length = length/60
            lu = 'min'
        vars.length.set(f'{length:.2f} {lu}')

        vars.address.set(filename) # update the displayed filename
        vars.filename = filename # store the filename for the actual processing
        vars.load_button['state'] = 'normal' # activate processing button

        # TODO add preview frame(s?)
    # File didn't open - not a video, corrupt, read error, or whatever else it could be
    else:
        # TODO complain
        pass

"""Click handler for analysis button"""
def triggerAnalysis(vars):
    import sys
    sys.exit(0)


### UI Configuration
"""Set the window's position"""
def center(win):
    # Current window dimensions
    w = win.winfo_height()
    h = win.winfo_width()

    # Screen width/height
    ww = win.winfo_screenwidth()
    wh = win.winfo_screenheight()

    # Calculate window coordinates
    x = ((ww//2) - (w//2))
    y = ((wh//2) - (h//2))

    # win.geometry(f'{w}x{h}+{x}+{y}') # And set them
    win.geometry(f'+{x}+{y}') # position only for now

"""Add elements to allow picking of a video file and for metadata preview"""
def populateL(parent, vars):
    # StringVars for dynamic label updating
    vars.address = StringVar(); vars.address.set('Choose a file...')
    vars.FPS = StringVar()
    vars.dim = StringVar()
    vars.frames = StringVar()
    vars.length = StringVar()


    # file address label
    lAdd = Label(parent, textvariable=vars.address)
    lAdd.grid(row=1, column=1, columnspan=2, padx=15, sticky=E)

    # file chooser button
    lChoose = Button(parent, text='Choose Source', command=lambda: triggerVideoSelect(vars))
    lChoose.grid(row=1, column=3)

    # metadata pane - contains all metadata labels added in the below block
    lMetadata = Frame(parent, borderwidth=1, relief='solid')
    lMetadata.grid(row = 2, column=1, 
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
        dFPS = Label(lMetaSubframe, textvariable=vars.FPS)
        dFPS.grid(row = 0, column=1, sticky=W)
        dDim = Label(lMetaSubframe, textvariable=vars.dim)
        dDim.grid(row = 1, column=1, sticky=W)
        dFrames = Label(lMetaSubframe, textvariable=vars.frames)
        dFrames.grid(row = 2, column=1, sticky=W)
        dLength = Label(lMetaSubframe, textvariable=vars.length)
        dLength.grid(row = 3, column=1, sticky=W)

    # Button to progress to next stage
    lload = Button(parent, text='Analyse Video', command=lambda: triggerAnalysis(vars))
    lload.grid(row=5, column=2)
    lload['state'] = 'disabled'
    vars.load_button = lload # store button for reactivation later

    # Favour whitespace in favour of stretching the metadata pane
    parent.rowconfigure(3, weight=1)
    parent.rowconfigure(4, weight=10)
    # Equal width for space on both sides of the analysis button
    parent.columnconfigure(1, minsize=lChoose.winfo_reqwidth())

"""Add UI elements for the video preview"""
def populateR(parent, vars):
    img_preview_dim = 256 # size of the preview image (pixels, square)

    # use a Canvas to draw the actual image
    preview = Canvas(parent, relief='solid', borderwidth=1)
    preview.grid(row=1, column=5, rowspan=4)
    preview.configure(width=img_preview_dim, height=img_preview_dim)

    # Horizontal scrollbar to allow scrubbing through the video - TODO may not be necessary 🙃
    scrub = Scrollbar(parent, orient='horizontal')#, command=a.yview)
    #src.configure(xscrollcommand=scrub.set)
    scrub.grid(row=5, column=5, sticky=[E, W])

"""Add padding around the root window"""
def configureFrame(parent):
    padding = 10

    parent.rowconfigure(0, minsize=padding)
    parent.rowconfigure(5, pad=padding*2)
    parent.columnconfigure(0, minsize=padding)
    parent.columnconfigure(4, minsize=padding)
    parent.columnconfigure(6, minsize=padding)

    center(window) # set window location

if __name__ == '__main__':
    # Create a new Tk framework instance / window
    window = Tk()

    # Blank container to store shared data in
    # Allows dynamic updating of text in the UI
    vars = type('', (), {})()

    populateL(window, vars) # video load/data elements
    populateR(window, vars) # video preview elements
    configureFrame(window)  # window margins, etc.

    window.mainloop() # hand control off to tkinter
