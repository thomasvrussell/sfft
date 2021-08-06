Package Description
-------------------
sfft is a Python package to perform image subtraction in Fourier space.

Documentation
-------------
Package documentation is available at: TBD

Installation
-----------
sfft has the following three backends to perform the image subtraction.

- NumPy backend: sfft will totally run on the CPU devices. 
- PyCUDA backend: The core functions of sfft are written in `PyCUDA <https://github.com/inducer/pycuda>`_ and `Scikit-Cuda <https://github.com/lebedov/scikit-cuda>`_. If you want to use this backend, please install `PyCUDA <https://github.com/inducer/pycuda>`_ and `Scikit-Cuda <https://github.com/lebedov/scikit-cuda>`_ according to your own CUDA version.
- CuPy backend: The core functions of sfft are written in `CuPy <https://github.com/cupy/cupy>`_. If you want to use this backend, please install `CuPy <https://github.com/cupy/cupy>`_ according to your own CUDA version.

The latter two backends require GPU device(s) with double-precision support. 
 
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

Citing
------
TBD