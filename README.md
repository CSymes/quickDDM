# quickDDM [![Build Status](https://dev.azure.com/s3550167/quickDDM/_apis/build/status/CSymes.quickDDM?branchName=develop)](https://dev.azure.com/s3550167/quickDDM/_build/latest?definitionId=1&branchName=develop)
Efficient processing of Differential Dynamic Microscopy (DDM) allowing both
traditional CPU-based, and GPU-accelerated processing, with a GUI for analysis.

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
scipy==1.3.0

[Optionally for OpenCL utilisation]
pyopencl==2018.2.5
reikna==0.7.2
```

These may be installed easily by running `pip install -r requirements.txt`
from in the source directory, assuming pip is already installed.
It is recommended that a virtual environment is used to separate this from the
system environment.

## Running

To launch the UI from the command line, navigate to the project directory
and execute `python launcher.py`.  
Alternatively, download and run a binary build from
[Releases](https://github.com/CSymes/quickDDM/releases)
or a build artifact from [Azure](dev.azure.com/s3550167/quickDDM/_build).

## Testing

There are a series of tests in the `tests` directory.
They can be run by calling
`python -m unittest [-v]`.  
Individual tests may be run as such:
(e.g.) `python -m unittest tests.testFourierTransforms`

## Building

If you need a binary build, run `publish.py` from the project root.
Creates a portable build of the program in a `dist` folder, with all
necessary libraries bundled. It can be run by executing the `quickDDM.exe`
file inside it.  
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

Our understanding of the core process was heavily informed by:  
L. G. Wilson et al., "Differential Dynamic Microscopy of Bacterial Motility,"
Physical Review Letters, vol. 106, no. 1, p. 4, Jan 2011.
