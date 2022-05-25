..  image:: https://github.com/thomasvrussell/sfft/blob/master/docs/sfft_logo_transparent.png

Package Description
-------------------
Saccadic Fast Fourier Transform (SFFT) is an algorithm for image subtraction in Fourier space.

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6576426.svg
   :target: https://doi.org/10.5281/zenodo.6576426
.. image:: https://img.shields.io/pypi/v/sfft.svg
    :target: https://pypi.python.org/pypi/sfft
    :alt: Latest Version
.. image:: https://static.pepy.tech/personalized-badge/sfft?period=total&units=international_system&left_color=grey&right_color=red&left_text=Downloads
    :target: https://pepy.tech/project/sfft
.. image:: https://static.pepy.tech/personalized-badge/sfft?period=month&units=international_system&left_color=grey&right_color=yellow&left_text=Downloads/month
    :target: https://pepy.tech/project/sfft
.. image:: https://img.shields.io/badge/python-3.7-green.svg
    :target: https://www.python.org/downloads/release/python-370/
.. image:: https://img.shields.io/badge/License-MIT-blue.svg
    :target: https://opensource.org/licenses/MIT

Installation
-----------
One can install the latest stable version of sfft from pip (recommended): ::
    
    pip install sfft

Or alternatively, install any desired version of sfft from Github `<https://github.com/thomasvrussell/sfft>`_: ::

    python setup.py install

sfft has the following three backends to perform the image subtraction.

.. [#] **NumPy backend** : sfft will totally run on the CPU devices. NO GPU devices and CUDA dependencies are required for this backend.
.. [#] **PyCUDA backend** : The core functions of sfft are written in `PyCUDA <https://github.com/inducer/pycuda>`_ and `Scikit-Cuda <https://github.com/lebedov/scikit-cuda>`_. Users need to install PyCUDA and Scikit-Cuda according to their CUDA version to enable this backend. Note this backend require GPU device(s) with double-precision support.
.. [#] **CuPy backend** : The core functions of sfft are written in `CuPy <https://github.com/cupy/cupy>`_. Users need to install CuPy according to their CUDA version to enable this backend. Note this backend require GPU device(s) with double-precision support.

- Case 1. You may enable the GPU backends (i.e., PyCUDA backend and CuPy backend) for CUDA 10.1 via: ::

    pip install pycuda==2020.1 scikit-cuda==0.5.3  # PyCUDA backend
    pip install cupy-cuda101                       # CuPy backend

- Case 2. You may enable the CuPy backend for CUDA 11.5 via: ::

    pip install cupy-cuda115  # CuPy backend
                   
**Additional Remarks**: CuPy backend is faster than PyCUDA backend, while it consumes more GPU memory. Generally, I recommend users to adopt CuPy backend as long as it does not incur GPU out-of-memory issue. Note that PyCUDA backend is still not compatiable with CUDA 11.

Dependencies
-----------

You need further to install additional astronomical software for sfft.

- `SExtractor <https://github.com/astromatic/sextractor>`_: SExtractor is required for sfft subtraction, as it enables sfft to determine a proper pixel mask over the input image-pair before the image subtraction (this is critical for a more reasonable parameter-solving). Note that we have wrapped SExtractor into a Python module ``sfft.utils.pyAstroMatic.PYSEx`` so that one can trigger SExtractor from Python. As an AstrOmatic software, you can install SExtractor following `<https://www.astromatic.net/software/sextractor/>`_, or alternatively, install via conda: ::

    conda install -c conda-forge astromatic-source-extractor

- `SWarp <https://github.com/astromatic/swarp>`_ (optional): This is not required for sfft subtraction itself. However, it is normally necessary to align the input image-pair before image subtraction. We have additionally wrapped SWarp into a Python module ``sfft.utils.pyAstroMatic.PYSWarp`` so that you can align images in a more Pythonic way. As an AstrOmatic software, you can install SWarp following `<https://www.astromatic.net/software/swarp/>`_, or alternatively, install via conda: ::

    conda install -c conda-forge astromatic-swarp

Quick start guide
-----------
We have prepared several examples in the test directory so that you can familar with the usage of the main functions in our software:

.. [*] **sfft subtraction for crowded field** : The example in subdirectory named subtract_test_crowded_flavor. We use crowded-flavor-sfft (module ``sfft.EasyCrowdedPacket``) to perform image subtraction for ZTF M31 observations. More detailed explanations of this module, see help(``sfft.EasyCrowdedPacket``).

.. [*] **sfft subtraction for sparse field** : The example in subdirectory named subtract_test_sparse_flavor. We use sparse-flavor-sfft (module ``sfft.EasySparsePacket``) to perform image subtraction for CTIO-4m DECam observations. More detailed explanations of this module, see help(``sfft.EasySparsePacket``). **IMPORTANT NOTICE: the input images of sparse-flavor-sfft should be SKY-SUBTRACTED!**

.. [*] **difference noise decorrelation** : The example in subdirectory named difference_noise_decorrelation. We use noise-decorrelation toolkit (module ``sfft.utils.DeCorrelationCalculator``) to whiten the background noise on difference image. In this test, the difference image is generated from image subtraction (by sfft) between a coadded reference image and a coadded science image, each stacked from 5 DECam individual observations with PSF homogenization (by sfft). The toolkit can be also applied to whiten a coadded image as long as convolution is involved in the stacking process.

Note that sfft subtraction is implemented as a two-step process. First of all, we need to mask the regions which can be be well-modeled by image subtraction (e.g., saturated stars, variables, cosmic rays, bad pixels, etc.). We designed two flavors of sfft (crowded & sparse), which actually follow the same subtraction algorithm but differ only in ways of the preliminary image-masking. The masked images are used to solve the parameters of image matching, and the solution will be ultimately applied on the raw un-masked images to get the final difference image. 

Advanced users may want to create the customized masked images with more elaborate pixel masking to replace the built-in masking process in sfft:

.. [*] **customized sfft subtraction** : The example in subdirectory named subtract_test_customized. The test data is the same as those for crowded-flavor-sfft (ZTF-M31 observations), however, the built-in automatic image-masking has been skipped by using given customized masked images as inputs. Such *pure* version of sfft is conducted by the module ``sfft.CustomizedPacket`` . More detailed explanations of the module: help(``sfft.CustomizedPacket``).

Parallel Computing
-----------

- In a particular time-domain survey, one may need to process a large set of image-pairs simultaneously. Assume that you have Nt tasks which should be processed by a computing platform with Nc CPU threads and Ng GPU devices. Generally, Nt >> Ng and Nc >> Ng. 

    E.g., Nt = 61 (A DECam exposure with CCDs), Nc = 40 (A CPU with 40 threads), and Ng = 1 (A Tesla A100 available).

- Note that we generally need to avoid multiple tasks using one GPU at the same time (GPU out-of-memory issue). That is to say, we CANNOT simply trigger a set of sfft functions (e.g., ``sfft.EasySparsePacket``) to process a large set of image-pairs simultaneously.

- Since version 1.1, sfft has allowed for multiple tasks without conflicting GPU usage, by using the modules ``sfft.MultiEasySparsePacket`` for sparse-flavor-sfft and ``sfft.MultiEasyCrowdedPacket`` for crowded-flavor-sfft, respectively. Please see the directory test/subtract_test_multiprocessing to find the examples. Note that ONLY the CuPy backend is supported in multiprocessing mode.

Common issues
-----------

- If your Python environment already has some version of llvmlite (a package required by NumPy backend) before installing sfft. The setup.py in sfft cannot properly update llvmlite to the desired version, then you may get errors related to Numba or llvmlite. If so, please manually install llvmlite by: ::

    pip install llvmlite==0.36.0 --ignore-installed

- If you are using GPU backends and you have a queue of observations to be processed, the first time in the loop of image subtraction can be very slow, and runtime is going to be stable after the first time. This might be due to some unknown initialization process in GPU devices. You can find the problem if you wrap any sfft subtraction task in a loop (e.g., try this in the customized sfft subtraction test). This problem can be solved by running a trivial subtraction (e.g., simply using empty images) in advance and making the pipe waiting for the subsequent observations (please see what we do in the test subtract_test_customized).

What's new
-----------

- The sfft is now optimized for multiple tasks since version 1.1.0. [Lei, May 24, 2022]

- A few argument-names have been changed since version 1.1.0, please see the test scripts. [Lei, May 24, 2022]

- Locking file is removed since version 1.1.0, as I found it unreliable in our tests, i.e., -GLockFile is removed. [Lei, May 24, 2022]

- The trial subtraction for refinement is removed since version 1.1.0. However, I add a post-subtraction check to search anomalies on the difference image using the same logic. One can feed the coordinates of the anomalies to sfft again as Prior-Banned sources to refine the subtraction (see -XY_PriorBan in ``sfft.MultiEasySparsePacket``). [Lei, May 24, 2022]

Todo list
-----------

- Write a detailed documentation for sfft! [Lei, May 24, 2022]

- We notice that SExtractor may have been called to perform astrometric calibration before image subtraction. It is definitely not wise to run SExtractor again in sfft, I need to develop a module which allows users to feed SExtractor products as inputs of sfft, which will significantly reduce the preprocessing time in sfft. [Lei, May 24, 2022]

- The multiprocessing mode is expected to accomondate multiple GPU devices, however, the function has not tested on such a multi-GPUs platform. [Lei, May 24, 2022]

- Add a function for optimizing sfft on a given computing platform with multiple CPU threading and one/multiple GPU card(s). This would be very useful to reduce the overall time cost when users have a large set of image-pairs to be processed simultaneously (e.g., serve for DECam, each exposure produces 61 CCD images). [Lei, May 20, 2022] **[ALREADY DONE]**

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

*Image Subtraction in Fourier Space. Hu, L., Wang, L., Chen, X. and Yang, J. 2021*

Arxiv link: `<https://arxiv.org/abs/2109.09334>`_.

Related DOI: TBD (Accepted by ApJ, waiting for DOI)
