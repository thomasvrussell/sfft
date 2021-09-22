..  image:: https://github.com/thomasvrussell/sfft/blob/master/docs/sfft_logo_transparent.png

Package Description
-------------------
Saccadic Fast Fourier Transform (SFFT) is an algorithm for image subtraction in Fourier space.

Installation
-----------
One can install the latest version of sfft from pip: ::
    
    pip install sfft

Or alternatively, install the desired version of sfft from Github `<https://github.com/thomasvrussell/sfft>`_: ::

    python setup.py install

sfft has the following three backends to perform the image subtraction.

- ``NumPy backend``: sfft will totally run on the CPU devices. No other dependencies are required for this backend.

- ``PyCUDA backend``: The core functions of sfft are written in `PyCUDA <https://github.com/inducer/pycuda>`_ and `Scikit-Cuda <https://github.com/lebedov/scikit-cuda>`_. Users need to install PyCUDA and Scikit-Cuda according to their CUDA version to enable this backend. Note this backend require GPU device(s) with double-precision support.

- ``CuPy backend``: The core functions of sfft are written in `CuPy <https://github.com/cupy/cupy>`_. Users need to install CuPy according to their CUDA version to enable this backend. Note this backend require GPU device(s) with double-precision support.

For example, you may enable the GPU backends (i.e., PyCUDA backend and CuPy backend) for CUDA 10.1 via: ::

    conda install -c conda-forge cudatoolkit=10.1
    pip install pycuda==2020.1 scikit-cuda==0.5.3 cupy-cuda101

If your Python environment already has some version of llvmlite (a package required by NumPy backend) before installing sfft. 
The setup.py in sfft cannot properly update llvmlite to the desired version, then you may get errors related to Numba or llvmlite. 
If so, please manually install llvmlite by: ::

    pip install llvmlite==0.36.0 —ignore-installed

Finally, you need further to install additional astronomical softwares for sfft.

- `SExtractor <https://github.com/astromatic/sextractor>`_: SExtractor is required for sfft subtraction, as it enables sfft to determine a proper pixel mask over the input image-pair before the image subtraction (this is critical for a more reasonable parameter-solving). Note that we have wrapped SExtractor into a Python module ``sfft.utils.pyAstroMatic.PYSEx``, so you also get an alternative way to trigger SExtractor.

- `SWarp <https://github.com/astromatic/swarp>`_ (optional): This is not required for sfft subtraction itself. However, it is normally necessary to align the input image-pair before image subtraction. We have additionally wrapped SWarp into a Python module ``sfft.utils.pyAstroMatic.PYSWarp`` so that you can align images in a more Pythonic way.

Quick start guide
-----------
We have prepared several examples in the test directory so that you can familar with the usage of the main functions in our software.

- **sfft subtraction for crowded field**: the example in subdirectory named subtract_test_crowded_flavor. We use crowded-flavor-sfft (module ``sfft.EasyCrowdedPacket``) to perform image subtraction for ZTF M31 observations. More detailed explanations of the module: help(``sfft.EasyCrowdedPacket``).

- **sfft subtraction for sparse field**: the example in subdirectory named subtract_test_sparse_flavor. We use sparse-flavor-sfft (module ``sfft.EasySparsePacket``) to perform image subtraction for CTIO-4m DECam observations. More detailed explanations of the module: help(``sfft.EasySparsePacket``).

- **difference noise decorrelation**: the example in subdirectory named difference_noise_decorrelation. We use noise-decorrelation toolkit (module ``sfft.utils.DeCorrelationCalculator``) to whiten the background noise on difference image. In this test, the difference image is generated from image subtraction (by sfft) between a coadded reference image and a coadded science image, each stacked from 5 DECam individual observations with PSF homogenization (by sfft). The toolkit can be also applied to whiten a coadded image as long as convolution is involved in the stacking process.

Note that sfft subtraction is implemented as a two-step process. First of all, we need to mask the regions which can be be well-modeled by image subtraction (e.g., saturated stars, variables, cosmic rays, bad pixels, etc.). We designed two flavors of sfft (crowded & sparse), which actually follow the same subtraction algorithm but differ only in ways of the preliminary image-masking. The masked images are used to solve the parameters of image matching, and the solution will be ultimately applied on the raw un-masked images to get the final difference image. You may want to create customized masked images with more elaborate pixel masking to replace the built-in masking process in sfft. 

- **customized sfft subtraction**: the example in subdirectory named subtract_test_customized. The test data is the same as those for crowded-flavor-sfft (ZTF-M31 observations), however, the built-in automatic image-masking has been skipped by using given customized masked images as inputs. Such *pure* version of sfft is conducted by the module ``sfft.CustomizedPacket`` . More detailed explanations of the module: help(``sfft.CustomizedPacket``).

Common issues
-----------
- If you are using GPU backends and you have a queue of observations to be processed, the first time in the loop of image subtraction can be very slow, and runtime is going to be stable after the first time. This might be due to some unknown initialization process in GPU devices. You can find the problem if you wrap any sfft subtraction task in a loop (e.g., try this in the customized sfft subtraction test). This problem can be solved by running a trivial subtraction (e.g., simply using empty images) in advance and making the pipe waiting for the subsequent observations.

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
- Version of sfft.

Citing
------
Hu, L., Wang, L., & Chen, X. 2021, Image Subtraction in Fourier Space. https://arxiv.org/abs/2109.09334
