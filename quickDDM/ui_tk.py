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
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, Image
from PIL.ImageTk import PhotoImage

import cv2


ADDR_PLACEHOLDER = 'Choose a file...' # Placeholder text for filepath entry box
WINDOW_PADDING = 10 # pixels
SCRUB_STEPS = 10 # previewable frames
PREVIEW_DIM = 256 # size of the preview image (pixels, square)



### Input Handlers


"""Click handler for the video selection button"""
def triggerVideoSelect(vars):
    # Open a file selection dialogue
    filename = askopenfilename(initialdir = '../tests/data', 
                               title = 'Select video file', 
                               filetypes = (('avi files', '*.avi'), 
                                            ('all files','*.*')))
    if not filename:
        return # file selector aborted

    loadVideo(vars, filename)

"""Handle direct file path entry"""
def triggerFromEntry(vars, widget):
    loadVideo(vars, widget.get())

"""Select a video after being called by one of the above trigger methods"""
def loadVideo(vars, filename):
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

        vars.video_file = videoFile
        handleScroll(vars, 'moveto', '0.0', '') # Activate scrollbar, show preview

        # Leave videoFile open for fetching of previews

    # File didn't open - not a video, corrupt, read error, or whatever else it could be
    else:
        messagebox.showerror('File Error',
                             'There was an issue with loading or accessing the ' +
                             'specified video file. Double check it\'s a video and that ' +
                             'you have the necessary permissions')
        videoFile.release() # Release anyway

        # TODO delete any previously loaded data

"""Click handler for analysis button"""
def triggerAnalysis(vars):
    vars.video_file.release()

    import sys
    sys.exit(0)

def setPreview(vars, index):
    vars.video_file.set(cv2.CAP_PROP_POS_FRAMES, index)
    status, frame = vars.video_file.read()

    if status:
        thumSize = vars.img_preview.winfo_width()
        pimg = Image.fromarray(frame) # convert from OpenCV to PIL
        pimg.thumbnail((thumSize, thumSize)) # downsize image
        tkimg = PhotoImage(image=pimg) # and convert to TK

        # Insert image onto the canvas
        vars.img_preview.create_image((0, 0), image=tkimg, anchor='nw')

        # prevent garbage collection
        # see http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
        vars.img_preview.img = tkimg
    else:
        print('Video could not be previewed - good luck processing it')

def handleScroll(vars, msg, x, units):
    if not vars.frames.get():
        return # No scrolling until a video's been selected, mate

    x = float(x) # It's a string for some reason

    if msg == 'moveto': # Truck was dragged
        pass
    elif msg == 'scroll':
        if units == 'pages': # Scrollbar was clicked on
            pass
        elif units == 'units': # Arrow clicked on
            x = vars.scrub.get()[0] + x/SCRUB_STEPS # budge the truck one index

    x = min(max(float(x), 0), 1-1/SCRUB_STEPS) # Bound its movement
    x = float(f'{x:.1f}') # Show discrete steps # this will be an issue for steps other than 0.1
    vars.scrub.set(str(x), str(x+1/SCRUB_STEPS)) # Update scrollbar position

    # Note this will never show the final frame.
    setPreview(vars, int(int(vars.frames.get())*x))
    # TODO some sort of caching - this wastes a lot of cpu redoing frames when scrubbing



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

    win.geometry(f'{h}x{w}+{x}+{y}') # And set them

"""Add elements to allow picking of a video file and for metadata preview"""
def populateL(parent, vars):
    # StringVars for dynamic label updating
    vars.address = StringVar(); vars.address.set(ADDR_PLACEHOLDER)
    vars.FPS = StringVar()
    vars.dim = StringVar()
    vars.frames = StringVar()
    vars.length = StringVar()


    # file address label
    lAdd = Entry(parent, textvariable=vars.address)
    lAdd.grid(row=1, column=1, columnspan=2, sticky=[E, W], padx=(0, WINDOW_PADDING))
    # Juggling to clear/restore placeholder text
    lAdd.bind('<FocusIn>', lambda e: e.widget.delete(0, 'end') if 
                                (e.widget.get() == ADDR_PLACEHOLDER) else None)
    lAdd.bind('<FocusOut>', lambda e: e.widget.insert(0, ADDR_PLACEHOLDER) if 
                                (e.widget.get() == '') else triggerFromEntry(vars, e.widget))

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
    lLoad = Button(parent, text='Analyse Video', command=lambda: triggerAnalysis(vars))
    lLoad.grid(row=5, column=2)
    lLoad['state'] = 'disabled'
    vars.load_button = lLoad # store button for reactivation later

    # Favour whitespace in favour of stretching the metadata pane
    parent.rowconfigure(3, weight=1)
    parent.rowconfigure(4, weight=10)
    # Equal width for space on both sides of the analysis button
    parent.columnconfigure(1, minsize=lChoose.winfo_reqwidth())

"""Add UI elements for the video preview"""
def populateR(parent, vars):
    # use a Canvas to draw the actual image
    preview = Canvas(parent, relief='solid', borderwidth=1)
    preview.grid(row=1, column=5, rowspan=4, sticky=N)
    preview.configure(width=PREVIEW_DIM, height=PREVIEW_DIM)
    vars.img_preview = preview # store the canvas so we can draw onto it later

    # Horizontal scrollbar to allow scrubbing through the video
    # This is kinda superfluous since we're looking at such small changes anyway 🙃
    scrub = Scrollbar(parent, orient='horizontal',
                      command=lambda x, y, z='': handleScroll(vars, x, y, z))
    scrub.grid(row=5, column=5, sticky=[E, W])
    vars.scrub = scrub

"""Add padding around the root window"""
def configureFrame(parent):
    parent.rowconfigure(0, minsize=WINDOW_PADDING) # Padding off top of the window
    parent.rowconfigure(5, pad=WINDOW_PADDING*2) # Bump processing button/slider off window edge+canvas
    parent.columnconfigure(0, minsize=WINDOW_PADDING) # Padding off left side of the window
    parent.columnconfigure(4, minsize=WINDOW_PADDING) # Gap between left and right sections
    parent.columnconfigure(6, minsize=WINDOW_PADDING) # Padding off right side of the window

    window.update() # generate geometry before moving window
    center(window) # set window location
    window.winfo_toplevel().title("quickDDM")
    window.resizable(False, False) # It's not really super responsive

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
