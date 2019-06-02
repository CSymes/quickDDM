# quickDDM [![Build Status](https://dev.azure.com/s3550167/quickDDM/_apis/build/status/CSymes.quickDDM?branchName=develop)](https://dev.azure.com/s3550167/quickDDM/_build/latest?definitionId=1&branchName=develop)
Efficient processing of Differential Dynamic Microscopy (DDM) allowing both
traditional CPU-based and GPU-accelerated processing, with a GUI for analysis.

## Contributors
Cary Symes (s3550167@student.rmit.edu.au)  
Lionel Penington (s3604656@student.rmit.edu.au)

GitHub address: https://github.com/CSymes/quickDDM

Release 1.0 - 03-JUNE-2019  
Changes since last release:  
This is the first release.

## Installation Instructions
To install the source code, please install Python 3.6 or greater. Ensure that
at least the following modules (or more recent versions thereof) are installed:

```
matplotlib==3.0.3
numpy==1.16.2
opencv-python==4.0.0.21
Pillow==6.0.0
pyopencl==2018.2.5
reikna==0.7.2
scipy==1.3.0
```

These may be installed easily by running `pip install -r requirements.txt`
from in the source directory, assuming pip is already installed.

## Running

To launch the UI from the command line, navigate to the 'quickDDM' directory
within the source directory and execute `python ui_tk.py`. Alternativly,
download and run a binary build from [Releases](https://github.com/CSymes/quickDDM/releases)
or a build artifact from [Azure](dev.azure.com/s3550167/quickDDM/_build).

## Building

If you need a binary build, run `publish.py` from the project root.
Creates a portable executable `ui_tk.exe` in a `dist` folder, with all
necessary libraries bundled.
Requires the `PyInstaller` packaged to be installed.

## Credits

This project incorporates concepts from:

G. Cerchiari, F. Croccolo, F. Cardinaux, and F. Scheffold, "Note:
Quasi-real-time analysis of dynamic near field scattering data using a graphics
processing unit," (in English), Review of Scientific Instruments, Article vol.
83, no. 10, p. 3, Oct 2012, Art no. 106101, doi: 10.1063/1.4755747  
The flow of "sequentialChunkerMain" in "processingCore.py" and
"sequentialGPUChunker" in "gpuCore.py" in paricular are
closely based on the GPU memory management technique detailed therein.

Our understanding of the core process was heaviy informed by:  
L. G. Wilson et al., "Differential Dynamic Microscopy of Bacterial Motility,"
Physical Review Letters, vol. 106, no. 1, p. 4, Jan 2011.
