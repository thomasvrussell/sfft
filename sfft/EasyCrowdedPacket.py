import time
import numpy as np
import os.path as pa
from astropy.io import fits
from sfft.AutoCrowdedPrep import Auto_CrowdedPrep
# version: Aug 31, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.2"

class Easy_CrowdedPacket:
    @staticmethod
    def ECP(FITS_REF, FITS_SCI, FITS_DIFF=None, FITS_Solution=None, ForceConv='AUTO', \
        GKerHW=None, KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, \
        ConstPhotRatio=True, MaskSatContam=False, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        BACK_TYPE='AUTO', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=5.0, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
        ONLY_FLAGS=None, BoundarySIZE=0.0, BACK_SIZE_SUPER=128, StarExt_iter=2, PriorBanMask=None, \
        BACKEND_4SUBTRACT='Cupy', CUDA_DEVICE_4SUBTRACT='0', NUM_CPU_THREADS_4SUBTRACT=8):

        """
        # NOTE: This function is to Perform Crowded-Flavor SFFT for single task:
        #       Pycuda & Cupy backend: do preprocessing on one CPU thread, and do subtraction on one GPU device.
        #       Numpy backend: do preprocessing on one CPU thread, and do subtraction with pyFFTW and Numba 
        #                      using multiple threads (-NUM_CPU_THREADS_4SUBTRACT).
        
        * Parameters for Crowded-Flavor SFFT [single task]
        
        # ----------------------------- Computing Enviornment --------------------------------- #

        -BACKEND_4SUBTRACT ['Cupy']         # can be 'Pycuda', 'Cupy' and 'Numpy'. 
                                            # Pycuda backend and Cupy backend require GPU device(s), 
                                            # while 'Numpy' is a pure CPU-based backend for sfft subtraction.
                                            # Cupy backend is even faster than Pycuda, however, it consume more GPU memory.
                                            # NOTE: Cupy backend can support CUDA 11*, while Pycuda does not (issues from Scikit-Cuda).

        -CUDA_DEVICE_4SUBTRACT ['0']        # it specifies certain GPU device (index) to conduct the subtraction task.
                                            # the GPU devices are usually numbered 0 to N-1 (you may use command nvidia-smi to check).
                                            # this argument becomes trivial for Numpy backend.

        -NUM_CPU_THREADS_4SUBTRACT [8]      # it specifies the number of CPU threads used for sfft subtraction in Numpy backend.
                                            # SFFT in Numpy backend has been implemented with pyFFTW and numba, 
                                            # that allow for parallel computing on CPUs. Of course, the Numpy 
                                            # backend is generally much slower than GPU backends.

        # ----------------------------- Preprocessing with Saturation Rejection for Image-Masking --------------------------------- #

        # > Configurations for SExtractor

        -GAIN_KEY ['GAIN']                  # SExtractor Parameter GAIN_KEY
                                            # i.e., keyword of GAIN in FITS header (of reference & science)

        -SATUR_KEY ['SATURATE']             # SExtractor Parameter SATUR_KEY
                                            # i.e., keyword of effective saturation in FITS header (of reference & science)
                                            # Remarks: note that Crowded-Flavor SFFT does not require sky-subtracted images as inputs,
                                            #          so the default keyword is the common name for saturation level.
        
        -BACK_TYPE ['AUTO']                 # SExtractor Parameter BACK_TYPE = [AUTO or MANUAL].
         
        -BACK_VALUE [0.0]                   # SExtractor Parameter BACK_VALUE (only work for BACK_TYPE='MANUAL')

        -BACK_SIZE [64]                     # SExtractor Parameter BACK_SIZE

        -BACK_FILTERSIZE [3]                # SExtractor Parameter BACK_FILTERSIZE

        -DETECT_THRESH [5.0]                # SExtractor Parameter DETECT_THRESH

        -DETECT_MINAREA [5]                 # SExtractor Parameter DETECT_MINAREA
        
        -DETECT_MAXAREA [0]                 # SExtractor Parameter DETECT_MAXAREA

        -DEBLEND_MINCONT [0.005]            # SExtractor Parameter DEBLEND_MINCONT (typically, 0.001 - 0.005)

        -BACKPHOTO_TYPE ['LOCAL']           # SExtractor Parameter BACKPHOTO_TYPE

        -ONLY_FLAGS [None]                  # Restrict SExtractor Output Photometry Catalog by Source FLAGS
                                            # Common FLAGS (Here None means no restrictions on FLAGS):
                                            # 1: aperture photometry is likely to be biased by neighboring sources 
                                            #    or by more than 10% of bad pixels in any aperture
                                            # 2: the object has been deblended
                                            # 4: at least one object pixel is saturated

        -BoundarySIZE [30]                  # Restrict SExtractor Output Photometry Catalog by Dropping Sources at Boundary 
                                            # NOTE: This would help to avoid selecting sources too close to image boundary. 

        # > Other Configurations

        -BACK_SIZE_SUPER [128]              # Also a SExtractor configuration BACK_SIZE, however, the background is 
                                            # used to fill the masked regions (saturation contaminated pixels).
                                            # NOTE: '_SUPER' means we would like to use a realtively large BACK_SIZE to
                                            #       make a super-smooth (structure-less) background. So the default
                                            #       -BACK_SIZE_SUPER > -BACK_SIZE

        -StarExt_iter [2]                   # make a further dilation for the masked region initially determined by SExtractor SEGMENTATION.
                                            # -StarExt_iter means the iteration times of the dilation process. 
        
        -PriorBanMask [None]                # a Numpy boolean array, with shape consistent with reference (science).
                                            # one can deliver a customized mask that covers known 
                                            # variables/transients/bad-pixels by this argument.
                                            # SFFT will then merge (Union operation) the user-defined mask and 
                                            # the SExtractor-determined saturation mask as the final mask.

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv ['AUTO']                 # it determines which image will be convolved, can be 'REF', 'SCI' and 'AUTO'.
                                            # -ForceConv = 'AUTO' means SFFT will determine the convolution direction according to 
                                            # FWHM_SCI and FWHM_REF: the image with better seeing will be convolved to avoid deconvolution.

        -GKerHW [None]                      # The given kernel half-width, None means the kernel size will be 
                                            # automatically determined by -KerHWRatio (to be seeing-related). 

        -KerHWRatio [2.0]                   # The ratio between FWHM and the kernel half-width
                                            # KerHW = int(KerHWRatio * Max(FWHM_REF, FWHM_SCI))

        -KerHWLimit [(2, 20)]               # The lower & upper bounds for kernel half-width 
                                            # KerHW is updated as np.clip(KerHW, KerHWLimit[0], KerHWLimit[1]) 
                                            # Remarks: this is useful for a survey since it can constrain the peak GPU memory usage.

        -KerPolyOrder [2]                   # Polynomial degree of kernel spatial variation, can be [0,1,2,3].

        -BGPolyOrder [2]                    # Polynomial degree of differential-background spatial variation, can be [0,1,2,3].
                                            # It is non-trivial for Crowded-Flavor SFFT, as the input images are usually not sky subtracted.

        -ConstPhotRatio [True]              # Constant photometric ratio between images ? can be True or False
                                            # ConstPhotRatio = True: the sum of convolution kernel is restricted to be a 
                                            #                constant across the field. 
                                            # ConstPhotRatio = False: the flux scaling between images is modeled by a 
                                            #                polynomial with degree -KerPolyOrder.

        -MaskSatContam [False]              # Mask saturation-contaminated regions on difference image? can be True or False
                                            # NOTE the pixels enclosed in the regions are replaced by NaN.

        # ----------------------------- Input & Output --------------------------------- #

        -FITS_REF []                        # File path of input reference image.

        -FITS_SCI []                        # File path of input science image.

        -FITS_DIFF [None]                   # File path of output difference image.

        -FITS_Solution [None]               # File path of the solution of the linear system. 
                                            # it is an array of (..., a_ijab, ... b_pq, ...).

        # Important Notice:
        #
        # a): if reference is convolved in SFFT, then DIFF = SCI - Convolved_REF.
        #     [difference image is expected to have PSF & flux zero-point consistent with science image]
        #     e.g., -ForceConv='REF' or -ForceConv='AUTO' when reference has better seeing.
        #
        # b): if science is convolved in SFFT, then DIFF = Convolved_SCI - REF
        #     [difference image is expected to have PSF & flux zero-point consistent with reference image]
        #     e.g., -ForceConv='SCI' or -ForceConv='AUTO' when science has better seeing.
        #
        # Remarks: this convention is to guarantee that transients emerge on science image always 
        #          show a positive signal on difference images.

        """

        # * Perform Auto Crowded-Prep [Mask Saturation]
        _ACP = Auto_CrowdedPrep(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            BACK_TYPE=BACK_TYPE, BACK_VALUE=BACK_VALUE, BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, \
            DETECT_THRESH=DETECT_THRESH, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, \
            DEBLEND_MINCONT=DEBLEND_MINCONT, BACKPHOTO_TYPE=BACKPHOTO_TYPE, ONLY_FLAGS=ONLY_FLAGS, BoundarySIZE=BoundarySIZE)
        SFFTPrepDict = _ACP.AutoMask(BACK_SIZE_SUPER=BACK_SIZE_SUPER, StarExt_iter=StarExt_iter, PriorBanMask=PriorBanMask)

        # * Determine ConvdSide & KerHW
        FWHM_REF = SFFTPrepDict['FWHM_REF']
        FWHM_SCI = SFFTPrepDict['FWHM_SCI']

        assert ForceConv in ['AUTO', 'REF', 'SCI']
        if ForceConv == 'AUTO':
            if FWHM_SCI >= FWHM_REF: ConvdSide = 'REF'
            else: ConvdSide = 'SCI'
        else: ConvdSide = ForceConv

        if GKerHW is None:
            FWHM_La = np.max([FWHM_REF, FWHM_SCI])
            KerHW = int(np.clip(KerHWRatio * FWHM_La, KerHWLimit[0], KerHWLimit[1]))
        else: KerHW = GKerHW

        # * Compile Functions in SFFT Subtraction
        from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure

        PixA_REF = SFFTPrepDict['PixA_REF']
        PixA_SCI = SFFTPrepDict['PixA_SCI']
        
        Tcomp_start = time.time()
        SFFTConfig = SingleSFFTConfigure.SSC(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
        print('\nMeLOn Report: Compiling Functions in SFFT Subtraction Takes [%.3f s]' %(time.time() - Tcomp_start))

        # * Perform SFFT Subtraction
        from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract

        SatMask_REF = SFFTPrepDict['REF-SAT-Mask']
        SatMask_SCI = SFFTPrepDict['SCI-SAT-Mask']
        NaNmask_U = SFFTPrepDict['Union-NaN-Mask']
        PixA_mREF = SFFTPrepDict['PixA_mREF']
        PixA_mSCI = SFFTPrepDict['PixA_mSCI']

        if ConvdSide == 'REF':
            PixA_mI, PixA_mJ = PixA_mREF, PixA_mSCI
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_REF.copy(), PixA_SCI.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_REF, PixA_SCI
            if MaskSatContam: 
                ContamMask_I = SatMask_REF
                ContamMask_J = SatMask_SCI
            else: ContamMask_I = None

        if ConvdSide == 'SCI':
            PixA_mI, PixA_mJ = PixA_mSCI, PixA_mREF
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_SCI.copy(), PixA_REF.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_SCI, PixA_REF
            if MaskSatContam: 
                ContamMask_I = SatMask_SCI
                ContamMask_J = SatMask_REF
            else: ContamMask_I = None
        
        Tsub_start = time.time()
        _tmp = GeneralSFFTSubtract.GSS(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
            SFFTConfig=SFFTConfig, ContamMask_I=ContamMask_I, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
        Solution, PixA_DIFF, ContamMask_CI = _tmp
        if MaskSatContam:
            ContamMask_DIFF = np.logical_or(ContamMask_CI, ContamMask_J)
        print('\nMeLOn Report: SFFT Subtraction Takes [%.3f s]' %(time.time() - Tsub_start))
        
        # * Modifications on the difference image
        #   a) when REF is convolved, DIFF = SCI - Conv(REF)
        #      PSF(DIFF) is coincident with PSF(SCI), transients on SCI are positive signal in DIFF.
        #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
        #      PSF(DIFF) is coincident with PSF(REF), transients on SCI are still positive signal in DIFF.

        if NaNmask_U is not None:
            # ** Mask Union-NaN region
            PixA_DIFF[NaNmask_U] = np.nan
        if MaskSatContam:
            # ** Mask Saturate-Contaminate region 
            PixA_DIFF[ContamMask_DIFF] = np.nan
        if ConvdSide == 'SCI': 
            # ** Flip difference when science is convolved
            PixA_DIFF = -PixA_DIFF
        
        # * Save difference image
        if FITS_DIFF is not None:
            _hdl = fits.open(FITS_SCI)
            _hdl[0].data[:, :] = PixA_DIFF.T
            _hdl[0].header['NAME_REF'] = (pa.basename(FITS_REF), 'MeLOn: SFFT')
            _hdl[0].header['NAME_SCI'] = (pa.basename(FITS_SCI), 'MeLOn: SFFT')
            _hdl[0].header['FWHM_REF'] = (FWHM_REF, 'MeLOn: SFFT')
            _hdl[0].header['FWHM_SCI'] = (FWHM_SCI, 'MeLOn: SFFT')
            _hdl[0].header['KERORDER'] = (KerPolyOrder, 'MeLOn: SFFT')
            _hdl[0].header['BGORDER'] = (BGPolyOrder, 'MeLOn: SFFT')
            _hdl[0].header['CPHOTR'] = (str(ConstPhotRatio), 'MeLOn: SFFT')
            _hdl[0].header['KERHW'] = (KerHW, 'MeLOn: SFFT')
            _hdl[0].header['CONVD'] = (ConvdSide  , 'MeLOn: SFFT')
            _hdl.writeto(FITS_DIFF, overwrite=True)
            _hdl.close()
        
        # * Save solution array
        if FITS_Solution is not None:
            phdu = fits.PrimaryHDU()
            phdu.header['DK'] = (SFFTConfig[0]['DK'], 'MeLOn: SFFT')
            phdu.header['DB'] = (SFFTConfig[0]['DB'], 'MeLOn: SFFT')
            phdu.header['L0'] = (SFFTConfig[0]['L0'], 'MeLOn: SFFT')
            phdu.header['L1'] = (SFFTConfig[0]['L1'], 'MeLOn: SFFT')
            
            phdu.header['FIJ'] = (SFFTConfig[0]['Fij'], 'MeLOn: SFFT')
            phdu.header['FAB'] = (SFFTConfig[0]['Fab'], 'MeLOn: SFFT')
            phdu.header['FPQ'] = (SFFTConfig[0]['Fpq'], 'MeLOn: SFFT')
            phdu.header['FIJAB'] = (SFFTConfig[0]['Fijab'], 'MeLOn: SFFT')

            PixA_Solution = Solution.reshape((-1, 1))
            phdu.data = PixA_Solution.T
            fits.HDUList([phdu]).writeto(FITS_Solution, overwrite=True)
        
        return SFFTPrepDict, Solution, PixA_DIFF
