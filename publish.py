import PyInstaller.__main__
import win32com.client
from os.path import abspath

import reikna
import cv2
import os

if __name__ == '__main__':
    if os.name != 'nt':
        print('This script only builds under Windows - aborting')
        exit()

    # Get paths to the install locations for OpenCV and Reikna
    # PyInstaller misses a few files by default, make sure they get added
    cv_p = cv2.__path__[0]
    r_p = reikna.__path__[0]

    params = [r'launcher.py',
              r'-y',
              r'--paths=quickDDM',
              r'--name=quickDDM',
              r'--onedir',
              r'--noconsole',
              r'--exclude-module=libopenblas',
             fr'--add-data={r_p};reikna',
             fr'--add-binary={cv_p}\opencv_ffmpeg400_64.dll;.'
             ]

    PyInstaller.__main__.run(params)
