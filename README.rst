..  image:: https://github.com/thomasvrussell/sfft/blob/master/docs/sfft_logo_gwbkg.png

Package Description
---------------------

.. image:: https://img.shields.io/pypi/v/sfft.svg
    :target: https://pypi.python.org/pypi/sfft
    :alt: Latest Version

Saccadic Fast Fourier Transform (SFFT) is an algorithm for image subtraction in Fourier space. SFFT brings about a remarkable improvement of computational performance of around an order of magnitude compared to other published image subtraction codes. 

..  image:: https://github.com/thomasvrussell/sfft/blob/master/docs/sfft_subtract_speed.png

To get a clear picture of our method, we summarize a variety of features from different perspectives for SFFT and other existing image subtraction methods.

..  image:: https://github.com/thomasvrussell/sfft/blob/master/docs/sfft_features.png

SFFT method is the transient detection engine for several ongoing time-domain programs, including the `DESIRT <https://ui.adsabs.harvard.edu/abs/2022TNSAN.107....1P/abstract>`_ survey based on DECam & DESI, the DECam GW-MMADS Survey for GW Follow-ups and the JWST Cycle 3 Archival program `AR 5965 <https://www.stsci.edu/jwst/science-execution/program-information?id=5965>`_. SFFT is also the core engine for the differential photometry pipeline of the `Roman Supernova PIT <https://github.com/Roman-Supernova-PIT>`_.

Installation
--------------
.. image:: https://static.pepy.tech/personalized-badge/sfft?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads
    :target: https://pepy.tech/project/sfft

One can install the latest stable version of sfft from pip (recommended): ::
    
    pip install sfft

Or alternatively, install any desired version of sfft from Github `<https://github.com/thomasvrussell/sfft>`_: ::

    python setup.py install

sfft now has the following two backends to perform the image subtraction.

**NumPy backend** : sfft will totally run on the CPU devices. NO GPU devices and CUDA dependencies are required for this backend.

**CuPy backend** : The core functions of sfft are written in `CuPy <https://github.com/cupy/cupy>`_. Users need to install CuPy according to their CUDA version to enable this backend. Note this backend require GPU device(s) with double-precision support.

- CUDA 12: E.g, you may enable the CuPy backend for CUDA 12.0 via: ::

    pip install cupy-cuda12x

- CUDA 11: E.g, you may enable the CuPy backend for CUDA 11.5 via: ::

    pip install cupy-cuda115

- CUDA 10: E.g, you may enable the CuPy backend for CUDA 10.1 via: ::

    pip install cupy-cuda101

**Additional Remarks**: There was a PyCUDA backend in sfft but now deprecated since v1.4.0. For sfft < v1.4.0, PyCUDA backend was preserved as it consumes less GPU memory. However, the CuPy backend is now better implemented for GPU memory allocation, making the PyCUDA backend no longer useful.

Dependencies
--------------

You need further to install additional astronomical software for sfft.

- `SExtractor <https://github.com/astromatic/sextractor>`_: SExtractor is required for sfft preprocessing, as it enables sfft to determine a proper pixel mask over the input image-pair before the image subtraction (this is critical for a more reasonable parameter-solving). Note that we have wrapped SExtractor into a Python module ``sfft.utils.pyAstroMatic.PYSEx`` so that one can trigger SExtractor within Python (please type help(``sfft.utils.pyAstroMatic.PYSEx``) to find its usage). As an AstrOmatic software, you can install SExtractor following `<https://www.astromatic.net/software/sextractor/>`_, or alternatively, install via conda: ::

    conda install -c conda-forge astromatic-source-extractor

- `SWarp <https://github.com/astromatic/swarp>`_ (optional): This is not required for sfft subtraction itself. However, it is normally necessary to align the input image-pair before image subtraction. We have additionally wrapped SWarp into a Python module ``sfft.utils.pyAstroMatic.PYSWarp`` so that you can align images in a more Pythonic way (please type help(``sfft.utils.pyAstroMatic.PYSWarp``) to find its usage). As an AstrOmatic software, you can install SWarp following `<https://www.astromatic.net/software/swarp/>`_, or alternatively, install via conda: ::

    conda install -c conda-forge astromatic-swarp

Quick start guide
------------------
We have prepared several examples in the test directory so that you can familar with the usage of the main functions in our software:

.. [*] **sfft subtraction for crowded field** : The example in subdirectory named subtract_test_crowded_flavor. We use crowded-flavor-sfft (module ``sfft.EasyCrowdedPacket``) to perform image subtraction for ZTF M31 observations. More detailed explanations of this module, see help(``sfft.EasyCrowdedPacket``).

.. [*] **sfft subtraction for sparse field** : The example in subdirectory named subtract_test_sparse_flavor. We use sparse-flavor-sfft (module ``sfft.EasySparsePacket``) to perform image subtraction for CTIO-4m DECam observations. More detailed explanations of this module, see help(``sfft.EasySparsePacket``).

- **sparse-flavor-sfft requires background subtraction.** One can make use of SExtractor to subtract the sky background, which has been also wrapped in this package, please use the module ``sfft.utils.SExSkySubtract`` and type help(``sfft.utils.SExSkySubtract``) to find its usage. There is an additional example in subtract_test_sparse_flavor (see directory prepare_data_example), that shows how to prepare the input image pair for sparse-flavor-sfft by ``sfft.utils.SExSkySubtract`` for sky subtraction and ``sfft.utils.pyAstroMatic.PYSWarp`` for image alignment, respecitvely.

Remarks on the SFFT Preprocessing
-----------------------------------------------

- Our software provides two flavors for image subtraction, crowded-flavor-sfft and sparse-flavor-sfft, to accommodate the situations for the crowded and sparse fields, respectively. The two flavors actually follow the same routine for image subtraction and differ only in ways of masking the data. 

- Proper image-masking is required by SFFT to identify the pixels that cannot be correctly modeled (hereafter, distraction pixels), e.g., saturated sources, casual cosmic rays and moving objects, bad CCD pixels, optical ghosts, and even the variable objects and transients themselves. The pre-subtraction processing for image-masking is referred to as **preprocessing** in sfft.

- Our software provides a generic and robust function to perform **preprocessing** of the data, which has been extensively tested with data from various transient surveys. When you run crowded-flavor-sfft and sparse-flavor-sfft, sfft actually performs the generic **preprocessing** for image-masking and do the sfft subtraction subsequently. 

- More specificially, the built-in **preprocessing** in sfft consists of two steps: [1] identify the distraction pixels in the input image-pair [2] create the masked version of the input image-pair via replacing the identified distraction pixels by proper flux values. In sparse-flavor-sfft, we designed a source-selection based on SExtractor catalogs and identify the unselected regions as distraction pixels. Given that the input images are required to be sky-subtracted in sparse-flavor-sfft, we simply replace the distraction pixels by zeros; In crowded-flavor-sfft, we only identify the pixels contaminated by saturated sources as distraction pixels using SExtractor, and then replace the distraction pixels by local background flux. 

Customized usage
------------------

The built-in **preprocessing** in sfft (based on SExtractor) is only designed to provide a safe and generic approach which can adapt to diverse imaging data. In contrast to the high speed of the image subtraction, the computing performance of the built-in **preprocessing** is much less remarkable (says, 10 times more computing time). Given a particular time-domain program, we do believe there is plenty of room for further optimization of the computing expense on the **preprocessing**. The two suggestions below might be helpful for users who would like to incorporate sfft in their pipeline efficiently:

- For sparse-flavor-sfft, the built-in **preprocessing** performs a source-selection based on SExtractor catalogs and then create the masked images for subsequent subtraction. To optimize the overall computing expense of the pipeline, one can make use of the SExtractor products already generated in the preceding modules (e.g., astrometric calibration) for the source-selection (which is much faster than SExtractor) of sfft. It will avoid repeated SExtractor photometry and reduce computing time significantly.

- For crowded-flavor-sfft, the built-in **preprocessing** only mask the saturation-contaminated pixels using SExtractor. When data quality masks for the observed imaging data are available in a survey program, one can instead identify the invalid pixels using the data quality masks and mask them by local background. Hence, the built-in **preprocessing** can be totally skipped.

Besides, we encourage users to design dedicated image-masking strategies for their survey programs to unleash the great power of sfft subtraction!

Our software provides a customized module which allows users to feed their own image-masking results, i.e., the module only perform the sfft subtraction. In this test, you would see the lightning fast speed of sfft subtraction on GPU devices!

.. [*]  **customized sfft subtraction** : The example in subdirectory named subtract_test_customized. The test data is the same as those for crowded-flavor-sfft (ZTF-M31 observations), however, the built-in automatic image-masking has been skipped by using given customized masked images as inputs. Such *pure* version of sfft is conducted by the module ``sfft.CustomizedPacket``. More detailed explanations of the module: help(``sfft.CustomizedPacket``).

**Additional Remarks**: If you are using GPU backends and you have a queue of observations to be processed, the first time in the loop of sfft subtraction can be very slow, and runtime is going to be stable after the first time. This might be due to some unknown initialization process in GPU devices. You can find in above test that the GPU warming-up is quite slow. Fortunately, this problem can be esaily solved by running a trivial subtraction (e.g., on empty images) in advance and making the pipe waiting for the subsequent observations (see above test).

Parallel Computing
--------------------

We have also developed modules to optimize the overall computing performance of sparse-flavor-sfft and crowded-flavor-sfft for the cases when you need to deal with multiple tasks simultaneously.

- In a particular time-domain survey, one may need to process a large set of image-pairs simultaneously. Assume that you have Nt tasks which should be processed by a computing platform with Nc CPU threads and Ng GPU devices. Generally, Nt >> Ng and Nc >> Ng. 

    E.g., Nt = 61 (A DECam exposure with CCDs), Nc = 40 (A CPU with 40 threads), and Ng = 1 (A Tesla A100 available).

- Note that we generally need to avoid multiple tasks using one GPU at the same time (GPU out-of-memory issue). That is to say, we CANNOT simply trigger a set of sfft functions (e.g., ``sfft.EasySparsePacket``) to process a large set of image-pairs simultaneously.

- Since version 1.1, sfft has allowed for multiple tasks without conflicting GPU usage, by using the modules ``sfft.MultiEasySparsePacket`` for sparse-flavor-sfft and ``sfft.MultiEasyCrowdedPacket`` for crowded-flavor-sfft, respectively. Please see the directory test/subtract_test_multiprocessing to find the examples. Note that ONLY the CuPy backend is supported in multiprocessing mode.

Remarks on the direction of image subtraction
-----------------------------------------------

There is a universal argument named -ForceConv to control the direction of image subtraction, which works on all image subtraction modules in sfft.

.. [#]  'AUTO' means sfft will determine the direction of image subtraction automatically according to the estimated FWHM of reference image and science image. The image which has smaller FWHM will be convolved in the image subtraction to avoid deconvolution. After comparing the FWHM, 'AUTO' becomes 'REF' or 'SCI' (see below). One can get to know which image is eventually convolved in image subtraction from the primary header of the difference image (see the keyword 'CONVD'). This mode does not supported in the Customized module ``sfft.CustomizedPacket``.

.. [#]  'REF' means sfft will convolve the reference image and DIFF = SCI - Convolved_REF. As a result, the psf and flux zero-point of difference image is consistent with the unconvolved image, i.e., the science image. One can perform PSF / Aperture photometry on the transients on difference image as if it is an object living in the science image: using the same psf model / aperture and magnitude zeropoint.

.. [#]  'SCI' means sfft will convolve the reference image and DIFF = Convolved_SCI - REF. Consequently, the psf and flux zero-point of difference image is consistent with the unconvolved image, i.e., the reference image. One can perform PSF / Aperture photometry on the transients on difference image as if it is an object living in the reference image: using the same psf model / aperture and magnitude zeropoint (but of course, not including the observation date!).

Note that a transient on science image is always a positive signal on difference image whatever -ForceConv is.

Additional Function
---------------------

We also present a decorrelation module to whiten the background noise of the difference image.

.. [*]  **difference noise decorrelation** : The example in subdirectory named difference_noise_decorrelation. We use noise-decorrelation toolkit (module ``sfft.utils.DeCorrelationCalculator``) to whiten the background noise on difference image. In this test, the difference image is generated from image subtraction (by sfft) between a coadded reference image and a coadded science image, each stacked from 5 DECam individual observations with PSF homogenization (by sfft). The toolkit can be also applied to whiten a coadded image as long as convolution is involved in the stacking process.


Comments on Backward Compatiablity
------------------------------------

We have tried our best to ensure the backward compatiablity, however, the rule was sometimes overrided in the development of sfft, e.g., some arguments might be deprecated in higher version of sfft. Users might get errors when they use old scripts but update sfft to a higher version. To solve the problem, I have been maintaining the test scripts on Github to make sure they can always work for the lastest version of sfft. You can also find the change log of arguments in the test scripts. 

What's new
------------

- New release sfft v1.4.2+ that can support Python 3.10! [Lei, Jan 9, 2024]

- New tutorial jupyter notebooks are available now, find them in test directories! [Lei, Dec 4, 2023]

- A warning for users: As scikit-image has changed something in its function of hough detection since version 0.19.0, I recently found that the source selection in sfft will be affected by this upgrade. I have not checked the new function yet, for the time being I would recommend users to install a scikit-image >= 0.16.2 but <= 0.18.3. Possibly I may add a constrain on scikit-image version in sfft 1.3.5. [Lei, Nov 9, 2022]

- A warning message about the usage of ``sfft.MultiEasySparsePacket`` and ``sfft.MultiEasyCrowdedPacket`` is added in the related test scripts. [Lei, Oct 25, 2022]

- The preprocessing in sparse-flavor-sfft is refined using an additional rejection of mild varaibles since version 1.3.0. [Lei, Aug 19, 2022]

- The sfft is now optimized for multiple tasks since version 1.1.0. [Lei, May 24, 2022]

- A few argument-names have been changed since version 1.1.0, please see the test scripts. [Lei, May 24, 2022]

- Locking file is removed since version 1.1.0, as I found it unreliable in our tests, i.e., -GLockFile is removed. [Lei, May 24, 2022]

- The trial subtraction for refinement is removed since version 1.1.0. However, I add a post-subtraction check to search anomalies on the difference image using the same logic. One can feed the coordinates of the anomalies to sfft again as Prior-Banned sources to refine the subtraction (see -XY_PriorBan in ``sfft.MultiEasySparsePacket``). [Lei, May 24, 2022]

Todo list
-----------

- The total GPU memory usage is only optimized for KerPolyOrder = 2 & BGPolyOrder = 2, I will extend the optimization to other cases ASAP! In fact, I believe there is ample space for reducing the total GPU usage and I will explore it. [Lei, Nov 11, 2022] **[ALREADY DONE]**

- I will allows users to disable the hough detection for preprocessing when there are too few sources in the field in the next version sfft v1.3.5. [Lei, Nov 11, 2022]

- Add a verbose argument for sfft so that users can get more clean printed messages. [Lei, Nov 9, 2022] **[ALREADY DONE]**

- Test if we can use sep to replace SExtractor in preprocessing to make sfft more Pythonic. [Lei, Nov 9, 2022] 

- Incorporate the separate functions (in the folder beta4spline) for spline form sfft into the unified sfft functions. Note that only Numpy backend is currently available and the spline form is very memory-consuming. [Lei, July 6, 2022]

- Write a detailed documentation for sfft! [Lei, May 24, 2022]

- We notice that SExtractor may have been called to perform astrometric calibration before image subtraction. It is definitely not wise to run SExtractor again in sfft, I need to develop a module which allows users to feed SExtractor products as inputs of sfft, which will significantly reduce the preprocessing time in sfft. [Lei, May 24, 2022]

- The multiprocessing mode is expected to accomondate multiple GPU devices, however, the function has not tested on such a multi-GPUs platform. [Lei, May 24, 2022] **[ALREADY DONE]**

- Add a function for optimizing sfft on a given computing platform with multiple CPU threading and one/multiple GPU card(s). This would be very useful to reduce the overall time cost when users have a large set of image-pairs to be processed simultaneously (e.g., serve for DECam, each exposure produces 61 CCD images). [Lei, May 20, 2022] **[ALREADY DONE]**

Common issues
---------------

- If your Python environment already has some version of llvmlite (a package required by NumPy backend) before installing sfft. The setup.py in sfft cannot properly update llvmlite to the desired version, then you may get errors related to Numba or llvmlite. If so, please manually install llvmlite by: ::

    pip install llvmlite==0.36.0 --ignore-installed

Development
-------------
The latest source code can be obtained from
`<https://github.com/thomasvrussell/sfft>`_.

When submitting bug reports or questions via the `issue tracker 
<https://github.com/thomasvrussell/sfft/issues>`_, please include the following 
information:

- OS platform.
- Python version.
- CUDA and CuPy (or PyCUDA) version.
- Version of sfft.

Citing
--------

*Image Subtraction in Fourier Space. Lei Hu et al. 2022, The Astrophysical Journal, 936, 157*

Arxiv link: `<https://arxiv.org/abs/2109.09334>`_.

ApJ Publication link: `<https://doi.org/10.3847/1538-4357/ac7394>`_.

Related DOI: 10.3847/1538-4357/ac7394

Publications using SFFT method
--------------------------------

See ADS Library: https://ui.adsabs.harvard.edu/public-libraries/lc4tiTR_T--92f9k0YrRQg

Acknowledgment
--------------------------------

The version SFFT v1.6.* is developed during the `NASA GPU Hackathon 2024 <https://www.nas.nasa.gov/hackathon/#home>`_ as part of the pipeline optimization efforts for the `Roman Supernova PIT team <https://github.com/Roman-Supernova-PIT>`_. The valuable support and insightful suggestions from our team members and Hackathon mentors are essential for the improvements in this version of SFFT. 

I would like to specifically acknowledge the contributions of the following team members: Lauren Aldoroty (Duke), Robert Knop (LBNL), Shu Liu (Pitt), and Michael Wood-Vasey (Pitt). Additionally, I am grateful for the guidance provided by our mentors, Marcus Manos and Lucas Erlandson from NVIDIA.
