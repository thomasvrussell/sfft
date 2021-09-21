import os
import time
import numpy as np
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
from astropy.time import Time
from sfft.utils.meta.FileLockKit import FileLock
from sfft.AutoCrowdedPrep import Auto_CrowdedPrep

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class Easy_CrowdedPacket:
    @staticmethod
    def ECP(FITS_REF, FITS_SCI, FITS_DIFF=None, FITS_Solution=None, ForceConv=None, \
        GKerHW=None, KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, \
        ConstPhotRatio=True, backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8, \
        MaskSatContam=False, BACKSIZE_SUPER=128, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        DETECT_THRESH=5.0, StarExt_iter=2, PriorBanMask=None, GLockFile=None):

        """
        * Parameters for Crowded-Flavor SFFT
        # ----------------------------- Computing Enviornment --------------------------------- #

        -backend ['Pycuda']   # can be 'Pycuda', 'Cupy' and 'Numpy'. 
                              # Pycuda backend and Cupy backend require GPU device(s), 
                              # while 'Numpy' is a pure CPU-based backend.
                              # Cupy backend is even faster than Pycuda, however, it consume more GPU memory.

        -CUDA_DEVICE ['0']    # it specifies certain GPU device to conduct the subtraction task.
                              # the GPU devices are usually numbered 0 to N-1 (you may use command nvidia-smi to check).
                              # this argument becomes trivial for Numpy backend

        -NUM_CPU_THREADS [8]  # it specifies the number of CPU threads used for Numpy backend.
                              # SFFT of Numpy backend has been implemented with pyfftw and numba, 
                              # that allow parallel computing on CPUs. However, Numpy backend is 
                              # generally much slower than GPU backends.

        -GLockFile [None]     # File path for file locking to avoid GPU to deal with multiple tasks at the same time.
                              # -GLockFile = None means SFFT will automatically create a temporary file.

        # ----------------------------- Preliminary Image-Masking with Saturation Rejection --------------------------------- #

        -GAIN_KEY ['GAIN']      # keyword of GAIN in FITS header (of reference & science), for SExtractor configuration
                                # NOTE: we need to use SExtractor check image SEGMENTATION to mask Saturated sources.

        -GAIN_KEY ['SATURATE']  # keyword of saturation in FITS header (of reference & science), for SExtractor configuration
                                # Remarks: note that Crowded-Flavor SFFT does not require sky-subtracted images as inputs,
                                #          so the default keyword is the common name for saturation level.

        -DETECT_THRESH [5.0]    # Detect threshold for SExtractor configuration.

        -StarExt_iter [2]       # make a further dilation for the masked region initially determined by SExtractor SEGMENTATION.
                                # -StarExt_iter means the iteration times of the dilation process. 
        
        -PriorBanMask [None]    # a Numpy boolean array, with shape consistent with reference (science).
                                # one can deliver a customized mask that covers known 
                                # variables/transients/bad-pixels by this argument.
                                # SFFT will then merge (Union operation) the user-defined mask and 
                                # the SExtractor-determined saturation mask as the final mask.

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv [None]       # it determines which image will be convolved, can be 'REF', 'SCI' and None.
                                # -ForceConv = None means SFFT will determine the convolution direction according to 
                                # FWHM_SCI and FWHM_REF: the image with better seeing will be convolved to avoid deconvolution.

        -GKerHW [None]          # The given kernel half-width, None means the kernel size will be 
                                # automatically determined by -KerHWRatio (to be seeing-related). 

        -KerHWRatio [2.0]       # The ratio between FWHM and the kernel half-width
                                # KerHW = int(KerHWRatio * Max(FWHM_REF, FWHM_SCI))

        -KerHWLimit [(2, 20)]   # The lower & upper bounds for kernel half-width 
                                # KerHW is updated as np.clip(KerHW, KerHWLimit[0], KerHWLimit[1]) 
                                # Remarks: this is useful for a survey since it can constrain the peak GPU memory usage.

        -KerPolyOrder [2]       # Polynomial degree of kernel spatial variation.

        -BGPolyOrder [2]        # Polynomial degree of background spatial variation.
                                # This argument is trivial for Sparse-Flavor SFFT as input images have been sky subtracted.

        -ConstPhotRatio [True]  # Constant photometric ratio between images ? can be True or False
                                # ConstPhotRatio = True: the sum of convolution kernel is restricted to be a 
                                #                constant across the field. 
                                # ConstPhotRatio = False: the flux scaling between images is modeled by a 
                                #                polynomial with degree -KerPolyOrder.

        # ----------------------------- Input & Output --------------------------------- #

        -FITS_REF []            # File path of input reference image

        -FITS_SCI []            # File path of input science image

        -FITS_DIFF [None]       # File path of output difference image

        -FITS_Solution [None]   # File path of the solution of the linear system 
                                # it is an array of (..., a_ijab, ... b_pq, ...)

        # Important Notice:
        # a): if reference is convolved in SFFT, then DIFF = SCI - Convolved_REF.
        #     [difference image is expected to have PSF & flux zero-point consistent with science image]
        #     e.g., -ForceConv='REF' or -ForceConv=None and reference has better seeing.
        # b): if science is convolved in SFFT, then DIFF = Convolved_SCI - REF
        #     [difference image is expected to have PSF & flux zero-point consistent with reference image]
        #     e.g., -ForceConv='SCI' or -ForceConv=None and science has better seeing.
        #
        # Remarks: this convention is to guarantee that transients emerge on science image always 
        #          show a positive signal on difference images.

        """

        # * Perform Crowded-Prep [MaskSat]
        SFFTPrepDict = Auto_CrowdedPrep(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI).\
            AutoMask(BACKSIZE_SUPER=BACKSIZE_SUPER, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            DETECT_THRESH=DETECT_THRESH, StarExt_iter=StarExt_iter, PriorBanMask=PriorBanMask)
        
        # * Determine ConvdSide & KerHW
        FWHM_REF = SFFTPrepDict['FWHM_REF']
        FWHM_SCI = SFFTPrepDict['FWHM_SCI']

        if ForceConv is None:
            if FWHM_SCI >= FWHM_REF: ConvdSide = 'REF'
            else: ConvdSide = 'SCI'
        else: ConvdSide = ForceConv

        if GKerHW is None:
            FWHM_La = np.max([FWHM_REF, FWHM_SCI])
            KerHW = int(np.clip(KerHWRatio * FWHM_La, KerHWLimit[0], KerHWLimit[1]))
        else: KerHW = GKerHW

        if GLockFile is None:
            TDIR = mkdtemp(suffix=None, prefix='4lock', dir=None)
            LockFile = pa.join(TDIR, 'tmplock.txt')
        else: LockFile = GLockFile

        # * Configure SFFT 
        from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure

        PixA_REF = SFFTPrepDict['PixA_REF']
        PixA_SCI = SFFTPrepDict['PixA_SCI']
        SFFTConfig = SingleSFFTConfigure.SSC(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)
        
        with FileLock(LockFile):
            with open(LockFile, "a") as f:
                LTIME0 = Time.now()

                # * Perform Ultimate Subtraction
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
                    SFFTConfig=SFFTConfig, ContamMask_I=ContamMask_I, backend=backend, \
                    CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)
                Solution, PixA_DIFF, ContamMask_CI = _tmp
                if MaskSatContam:
                    ContamMask_DIFF = np.logical_or(ContamMask_CI, ContamMask_J)
                print('MeLOn Report: Ultimate Subtraction Takes [%.3f s]' %(time.time() - Tsub_start))
                
                # * Modifications on difference image
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
                
                LTIME1 = Time.now()
                Lmessage = 'FILELOCK | REF = %s & SCI = %s | %s + %.2f s' \
                    %(pa.basename(FITS_REF), pa.basename(FITS_SCI), \
                    LTIME0.isot, (LTIME1.mjd-LTIME0.mjd)*24*3600)
                
                print('\n---@--- %s ---@---\n' %Lmessage)
                f.write('EasyCrowdedPacket: %s \n' %Lmessage)
                f.flush()

        if GLockFile is None:
            os.system('rm -rf %s' %TDIR)

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
