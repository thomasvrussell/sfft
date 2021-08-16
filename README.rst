Package Description
-------------------
sfft is a Python package to perform image subtraction in Fourier space.

Documentation
-------------
Package documentation is available at: TBD

Installation
-----------
sfft can be simply installed in your Python environment by

- python setup.py install

sfft has the following three backends to perform the image subtraction.

- NumPy backend: sfft will totally run on the CPU devices. No other dependencies are required for this backend.
- PyCUDA backend: The core functions of sfft are written in `PyCUDA <https://github.com/inducer/pycuda>`_ and `Scikit-Cuda <https://github.com/lebedov/scikit-cuda>`_. Users need to install PyCUDA and Scikit-Cuda according to their CUDA version to enable this backend. Note this backend require GPU device(s) with double-precision support.
- CuPy backend: The core functions of sfft are written in `CuPy <https://github.com/cupy/cupy>`_. Users need to install CuPy according to their CUDA version to enable this backend. Note this backend require GPU device(s) with double-precision support.

For example, you may enable the GPU backends (i.e., PyCUDA backend and CuPy backend) for CUDA 10.1 via

- conda install -c conda-forge cudatoolkit=10.1
- pip install pycuda==2020.1 scikit-cuda==0.5.3 cupy-cuda101

Tips
-----------
If your Python environment already has some version of llvmlite (a package required by NumPy backend) before installing sfft. 
The setup.py in sfft cannot properly update llvmlite to the desired version, then you may get errors related to Numba or llvmlite. 
If so, please manually install llvmlite by 

- pip install llvmlite==0.36.0 —ignore-installed

Development
-----------
The latest source code can be obtained from
`<https://github.com/thomasvrussell/sfft>`_.

When submitting bug reports or questions via the `issue tracker 
<https://github.com/thomasvrussell/sfft/issues>`_, please include the following 
information:

- OS platform.
- Python version.
- CUDA, PyCUDA and CuPy version.
- Version or git revision of sfft.

TODO-list
-----------

- add decorrelation module.
- allow users to define image mask for subtraction.

Citing
------
TBD
