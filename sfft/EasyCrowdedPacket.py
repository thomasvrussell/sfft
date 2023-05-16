import time
import cupy as cp
import numpy as np
import os.path as pa
from astropy.io import fits
from sfft.AutoCrowdedPrep import Auto_CrowdedPrep
from sfft.utils.SFFTSolutionReader import Realize_FluxScaling
from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract
from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure
# version: Feb 25, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class Easy_CrowdedPacket:
    @staticmethod
    def ECP(FITS_REF, FITS_SCI, FITS_DIFF=None, FITS_Solution=None, ForceConv='AUTO', GKerHW=None, \
        KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
        MaskSatContam=False, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', BACK_TYPE='AUTO', \
        BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=5.0, ANALYSIS_THRESH=5.0, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
        ONLY_FLAGS=None, BoundarySIZE=0.0, BACK_SIZE_SUPER=128, StarExt_iter=2, PriorBanMask=None, \
        BACKEND_4SUBTRACT='Cupy', CUDA_DEVICE_4SUBTRACT='0', NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):
        
        """
        # NOTE: This function is to Perform Crowded-Flavor SFFT for single task:
        #       Cupy backend: do preprocessing on one CPU thread, and perform subtraction on one GPU device.
        #       Numpy backend: do preprocessing on one CPU thread, and perform subtraction with pyFFTW and Numba 
        #                      using multiple threads (-NUM_CPU_THREADS_4SUBTRACT).
        
        * Parameters for Crowded-Flavor SFFT [single task]
        
        # ----------------------------- Computing Enviornment --------------------------------- #

        -BACKEND_4SUBTRACT ['Cupy']         # 'Cupy' or 'Numpy'. 
                                            # Cupy backend require GPU(s) that is capable of performing double-precision calculations,
                                            # while Numpy backend is a pure CPU-based backend for sfft subtraction.
                                            # NOTE: 'Pycuda' backend is no longer supported since sfft v1.4.0.

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
                                            # NOTE: One may notice that the default DETECT_THRESH in SExtractor is 1.5.
                                            #       Using a 'very cold' detection threshold here is to speed up SExtractor.
                                            #       Although DETECT_THRESH = 5.0 means we will miss the faint-end sources with 
                                            #       SNR < 20 (approximately), the cost is generally acceptable for saturation mask.

        -ANALYSIS_THRESH [5.0]              # SExtractor Parameter ANALYSIS_THRESH
                                            # NOTE: By default, let -ANALYSIS_THRESH = -DETECT_THRESH

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
        
        # ----------------------------- Miscellaneous --------------------------------- #
        
        -VERBOSE_LEVEL [2]                  # The level of verbosity, can be [0, 1, 2]
                                            # 0/1/2: QUIET/NORMAL/FULL mode

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
        # Remarks: this convention is to guarantee that a transient emerge on science image 
        #          always shows itself as a positive signal on the difference images.

        """

        # * Perform Auto Crowded-Prep [Mask Saturation]
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER Crowded-Flavor Auto Preprocessing!')

        _ACP = Auto_CrowdedPrep(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            BACK_TYPE=BACK_TYPE, BACK_VALUE=BACK_VALUE, BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, \
            DETECT_THRESH=DETECT_THRESH, ANALYSIS_THRESH=ANALYSIS_THRESH, DETECT_MINAREA=DETECT_MINAREA, \
            DETECT_MAXAREA=DETECT_MAXAREA, DEBLEND_MINCONT=DEBLEND_MINCONT, BACKPHOTO_TYPE=BACKPHOTO_TYPE, \
            ONLY_FLAGS=ONLY_FLAGS, BoundarySIZE=BoundarySIZE, VERBOSE_LEVEL=VERBOSE_LEVEL)

        SFFTPrepDict = _ACP.AutoMask(BACK_SIZE_SUPER=BACK_SIZE_SUPER, \
            StarExt_iter=StarExt_iter, PriorBanMask=PriorBanMask)

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

        # * Choose GPU device for Cupy backend
        if BACKEND_4SUBTRACT == 'Cupy':
            device = cp.cuda.Device(int(CUDA_DEVICE_4SUBTRACT))
            device.use()

        # * Compile Functions in SFFT Subtraction
        PixA_REF = SFFTPrepDict['PixA_REF']
        PixA_SCI = SFFTPrepDict['PixA_SCI']
        
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER Function Compilations of SFFT-SUBTRACTION!')

        Tcomp_start = time.time()
        SFFTConfig = SingleSFFTConfigure.SSC(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, \
            VERBOSE_LEVEL=VERBOSE_LEVEL)

        if VERBOSE_LEVEL in [1, 2]:
            _message = 'Function Compilations of SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tcomp_start)
            print('\nMeLOn Report: %s' %_message)

        # * Perform SFFT Subtraction
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
        
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER SFFT-SUBTRACTION!')

        Tsub_start = time.time()
        _tmp = GeneralSFFTSubtract.GSS(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
            SFFTConfig=SFFTConfig, ContamMask_I=ContamMask_I, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)

        Solution, PixA_DIFF, ContamMask_CI = _tmp
        if VERBOSE_LEVEL in [1, 2]:
            _message = 'SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tsub_start)
            print('\nMeLOn Report: %s' %_message)
        
        # ** adjust the sign of difference image 
        #    NOTE: Our convention is to make the transients on SCI always show themselves as positive signal on DIFF
        #    (a) when REF is convolved, DIFF = SCI - Conv(REF)
        #    (b) when SCI is convolved, DIFF = Conv(SCI) - REF

        if ConvdSide == 'SCI':
            PixA_DIFF = -PixA_DIFF

        if VERBOSE_LEVEL in [1, 2]:
            if ConvdSide == 'REF':
                _message = 'Reference Image is Convolved in SFFT-SUBTRACTION [DIFF = SCI - Conv(REF)]!'
                print('MeLOn CheckPoint: %s' %_message)

            if ConvdSide == 'SCI':
                _message = 'Science Image is Convolved in SFFT-SUBTRACTION [DIFF = Conv(SCI) - REF]!'
                print('MeLOn CheckPoint: %s' %_message)

        # ** estimate the flux scaling through the convolution of image subtraction
        #    NOTE: if photometric ratio is not constant (ConstPhotRatio=False), we measure of a grid
        #          of coordinates to estimate the flux scaling and its fluctuation (polynomial form).
        #    NOTE: here we set the tile-size of the grid about 64 x 64 pix, but meanwhile, 
        #          we also require the tiles should no less than 6 along each axis.

        N0, N1 = SFFTConfig[0]['N0'], SFFTConfig[0]['N1']
        L0, L1 = SFFTConfig[0]['L0'], SFFTConfig[0]['L1']
        DK, Fpq = SFFTConfig[0]['DK'], SFFTConfig[0]['Fpq']

        if ConstPhotRatio:
            SFFT_FSCAL_NSAMP = 1
            XY_q = np.array([[N0/2.0, N1/2.0]]) + 0.5
            RFS = Realize_FluxScaling(XY_q=XY_q)
            SFFT_FSCAL_q = RFS.FromArray(Solution=Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)
            SFFT_FSCAL_MEAN, SFFT_FSCAL_SIG = SFFT_FSCAL_q[0], 0.0
        
        if not ConstPhotRatio:
            TILESIZE_X, TILESIZE_Y = 64, 64    # FIXME emperical/arbitrary values
            NTILE_X = np.max([round(N0/TILESIZE_X), 6])
            NTILE_Y = np.max([round(N1/TILESIZE_Y), 6])
            GX = np.linspace(0.5, N0+0.5, NTILE_X+1)
            GY = np.linspace(0.5, N1+0.5, NTILE_Y+1)
            YY, XX = np.meshgrid(GY, GX)
            XY_q = np.array([XX.flatten(), YY.flatten()]).T
            SFFT_FSCAL_NSAMP = XY_q.shape[0]
            
            RFS = Realize_FluxScaling(XY_q=XY_q)
            SFFT_FSCAL_q = RFS.FromArray(Solution=Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)
            SFFT_FSCAL_MEAN, SFFT_FSCAL_SIG = np.mean(SFFT_FSCAL_q), np.std(SFFT_FSCAL_q)
        
        if VERBOSE_LEVEL in [1, 2]:
            _message = 'The Flux Scaling through the Convolution of SFFT-SUBTRACTION '
            _message += '[%.6f +/- %.6f] from [%d] positions!\n' %(SFFT_FSCAL_MEAN, SFFT_FSCAL_SIG, SFFT_FSCAL_NSAMP)
            print('MeLOn CheckPoint: %s' %_message)
        
        # ** final tweaks on the difference image
        if NaNmask_U is not None:
            PixA_DIFF[NaNmask_U] = np.nan   # Mask Union-NaN regions
        
        if MaskSatContam:
            ContamMask_DIFF = np.logical_or(ContamMask_CI, ContamMask_J)
            PixA_DIFF[ContamMask_DIFF] = np.nan   # Mask Saturation-Contaminated regions

        # * Save difference image
        if FITS_DIFF is not None:

            """
            # Remarks on header of difference image
            # [1] In general, the FITS header of difference image would mostly be inherited from science image.
            #
            # [2] when science image is convolved, we turn to set GAIN_DIFF = GAIN_SCI / SFFT_FSCAL_MEAN (unit: e-/ADU)
            #     so that one can correctly convert ADU to e- for a transient appears on science image.
            #     WARNING: for a transient appears on reference image, GAIN_DIFF = GAIN_SCI is correct.
            #              we should keep in mind that GAIN_DIFF is not an absolute instrumental value.
            #     WARNING: one can only estimate the Poission noise from the science transient via GIAN_DIFF.
            #              the Poission noise from the background source (host galaxy) can enhance the 
            #              background noise at the position of the transient. photometry softwares which 
            #              only take background noise and transient Possion noise into account would tend 
            #              to overestimate the SNR of the transient.
            # 
            # [3] when science image is convolved, we turn to set SATURA_DIFF = SATURA_SCI * SFFT_FSCAL_MEAN         
            #     WARNING: it is still not a good idea to use SATURA_DIFF to identify the SATURATION regions
            #              on difference image. more appropriate way is masking the saturation contaminated 
            #              regions on difference image (MaskSatContam=True), or alternatively, leave them  
            #              alone and using AI stamp classifier to reject saturation-related bogus.
            #
            """

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
            _hdl[0].header['CONVD'] = (ConvdSide, 'MeLOn: SFFT')

            if ConvdSide == 'SCI':
                GAIN_SCI = _hdl[0].header[GAIN_KEY]
                SATUR_SCI = _hdl[0].header[SATUR_KEY]
                GAIN_DIFF = GAIN_SCI / SFFT_FSCAL_MEAN
                SATUR_DIFF = SATUR_SCI * SFFT_FSCAL_MEAN

                _hdl[0].header[GAIN_KEY] = (GAIN_DIFF, 'MeLOn: SFFT')
                _hdl[0].header[SATUR_KEY] = (SATUR_DIFF, 'MeLOn: SFFT')

            _hdl.writeto(FITS_DIFF, overwrite=True)
            _hdl.close()
        
        # * Save solution array
        if FITS_Solution is not None:
            phdu = fits.PrimaryHDU()
            phdu.header['N0'] = (SFFTConfig[0]['N0'], 'MeLOn: SFFT')
            phdu.header['N1'] = (SFFTConfig[0]['N1'], 'MeLOn: SFFT')
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

        return PixA_DIFF, SFFTPrepDict, Solution, SFFT_FSCAL_MEAN, SFFT_FSCAL_SIG
