import subprocess
import reikna
import cv2
import os

if __name__ == '__main__':
    if os.name != 'nt':
        print('This script only builds under Windows - aborting')
        exit()

    cv_p = cv2.__path__[0]
    r_p = reikna.__path__[0]

    command = (r'pyinstaller quickDDM/ui_tk.py -y '
               r'-p "./quickDDM" '
               r'--onefile '
               r'--noconsole '
               r'--exclude-module libopenblas '
              fr'--add-data "{r_p};reikna" '
              fr'--add-binary "{cv_p}/opencv_ffmpeg400_64.dll;."')

    subprocess.call(command)
