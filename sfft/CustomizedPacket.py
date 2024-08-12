import tracemalloc
import logging
import sys
import time
import numpy as np
import os.path as pa
from astropy.io import fits
from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract
from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure
# version: Feb 25, 2023

__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class Customized_Packet:
    @staticmethod
    def CP(FITS_REF, FITS_SCI, FITS_mREF, FITS_mSCI, ForceConv, GKerHW, \
        FITS_DIFF=None, FITS_Solution=None, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
        BACKEND_4SUBTRACT='Cupy', CUDA_DEVICE_4SUBTRACT='0', NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2, logger=None):
        
        """
        * Parameters for Customized SFFT
        # @ Customized: Skip built-in preprocessing (automatic image-masking) by a customized masked image-pair.
        #
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

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv []                       # it determines which image will be convolved, can be 'REF' or 'SCI'.
                                            # here ForceConv CANNOT be 'AUTO'!

        -GKerHW []                          # the given kernel half-width. 

        -KerPolyOrder [2]                   # Polynomial degree of kernel spatial variation.

        -BGPolyOrder [2]                    # Polynomial degree of background spatial variation.

        -ConstPhotRatio [True]              # Constant photometric ratio between images ? can be True or False
                                            # ConstPhotRatio = True: the sum of convolution kernel is restricted to be a 
                                            #                constant across the field. 
                                            # ConstPhotRatio = False: the flux scaling between images is modeled by a 
                                            #                polynomial with degree -KerPolyOrder.

        # ----------------------------- Input & Output --------------------------------- #

        -FITS_REF []                        # File path of input reference image.

        -FITS_SCI []                        # File path of input science image.

        -FITS_mREF []                       # File path of input masked reference image (NaN-free).

        -FITS_mSCI []                       # File path of input masked science image (NaN-free).

        -FITS_DIFF [None]                   # File path of output difference image.

        -FITS_Solution [None]               # File path of the solution of the linear system.
                                            # it is an array of (..., a_ijab, ... b_pq, ...).

        # ----------------------------- Miscellaneous --------------------------------- #
        
        -VERBOSE_LEVEL [2]                  # The level of verbosity, can be [0, 1, 2]
                                            # 0/1/2: QUIET/NORMAL/FULL mode

        # Important Notice:
        #
        # a): if reference is convolved in SFFT (-ForceConv='REF'), then DIFF = SCI - Convolved_REF.
        #     [the difference image is expected to have PSF & flux zero-point consistent with science image]
        #
        # b): if science is convolved in SFFT (-ForceConv='SCI'), then DIFF = Convolved_SCI - REF
        #     [the difference image is expected to have PSF & flux zero-point consistent with reference image]
        #
        # Remarks: this convention is to guarantee that transients emerge on science image always 
        #          show a positive signal on difference images.

        """

        # Configure logger (Rob)
        _logger = logging.getLogger(f'sfft.CustomizedPacket.CP')
        if not _logger.hasHandlers():
            log_out = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter(f'[%(asctime)s - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            log_out.setFormatter(formatter)
            _logger.addHandler(log_out)
            _logger.setLevel(logging.DEBUG) # ERROR, WARNING, INFO, or DEBUG (in that order by increasing detail)

        # * Read input images
        PixA_REF = fits.getdata(FITS_REF, memmap=False, ext=0).T
        PixA_SCI = fits.getdata(FITS_SCI, memmap=False, ext=0).T
        PixA_mREF = fits.getdata(FITS_mREF, memmap=False, ext=0).T
        PixA_mSCI = fits.getdata(FITS_mSCI, memmap=False, ext=0).T

        if not PixA_REF.flags['C_CONTIGUOUS']:
            PixA_REF = np.ascontiguousarray(PixA_REF, np.float64)
        else: PixA_REF = PixA_REF.astype(np.float64)

        if not PixA_SCI.flags['C_CONTIGUOUS']:
            PixA_SCI = np.ascontiguousarray(PixA_SCI, np.float64)
        else: PixA_SCI = PixA_SCI.astype(np.float64)

        if not PixA_mREF.flags['C_CONTIGUOUS']:
            PixA_mREF = np.ascontiguousarray(PixA_mREF, np.float64)
        else: PixA_mREF = PixA_mREF.astype(np.float64)

        if not PixA_mSCI.flags['C_CONTIGUOUS']:
            PixA_mSCI = np.ascontiguousarray(PixA_mSCI, np.float64)
        else: PixA_mSCI = PixA_mSCI.astype(np.float64)

        NaNmask_U = None
        NaNmask_REF = np.isnan(PixA_REF)
        NaNmask_SCI = np.isnan(PixA_SCI)
        if NaNmask_REF.any() or NaNmask_SCI.any():
            NaNmask_U = np.logical_or(NaNmask_REF, NaNmask_SCI)

        assert np.sum(np.isnan(PixA_mREF)) == 0
        assert np.sum(np.isnan(PixA_mSCI)) == 0
        
        assert ForceConv in ['REF', 'SCI']
        ConvdSide = ForceConv
        KerHW = GKerHW

        # * Choose GPU device for Cupy backend
        if BACKEND_4SUBTRACT == 'Cupy':
            import cupy as cp
            device = cp.cuda.Device(int(CUDA_DEVICE_4SUBTRACT))
            device.use()

        # * Compile Functions in SFFT Subtraction
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER Function Compilations of SFFT-SUBTRACTION!')

        Tcomp_start = time.time()

        ### I ADDED DEBUG MEM TRACING STATEMENTS
        tracemalloc.start()
        SFFTConfig = SingleSFFTConfigure.SSC(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, \
            VERBOSE_LEVEL=VERBOSE_LEVEL)

        size, peak = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY IN CUSTOMIZEDPACKET FROM SFFTCONFIG size={size}, peak={peak}')
        tracemalloc.clear_traces()
        tracemalloc.stop()

        if VERBOSE_LEVEL in [1, 2]:
            _message = 'Function Compilations of SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tcomp_start)
            print('\nMeLOn Report: %s' %_message)

        # * Perform SFFT Subtraction
        if ConvdSide == 'REF':
            PixA_mI, PixA_mJ = PixA_mREF, PixA_mSCI
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_REF.copy(), PixA_SCI.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_REF, PixA_SCI

        if ConvdSide == 'SCI':
            PixA_mI, PixA_mJ = PixA_mSCI, PixA_mREF
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_SCI.copy(), PixA_REF.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_SCI, PixA_REF
        
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER SFFT-SUBTRACTION!')

        Tsub_start = time.time()

        tracemalloc.start()

        _tmp = GeneralSFFTSubtract.GSS(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
            SFFTConfig=SFFTConfig, ContamMask_I=None, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)

        size, peak = tracemalloc.get_traced_memory()
        logger.debug(f'MEMORY IN CUSTOMIZEDPACKET FROM GENERALSFFTSUBTRACT size={size}, peak={peak}')
        tracemalloc.clear_traces()
        tracemalloc.stop()

        Solution, PixA_DIFF = _tmp[:2]
        if VERBOSE_LEVEL in [1, 2]:
            _message = 'SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tsub_start)
            print('\nMeLOn Report: %s' %_message)
        
        # * Modifications on the difference image
        #   a) when REF is convolved, DIFF = SCI - Conv(REF)
        #      PSF(DIFF) is coincident with PSF(SCI), transients on SCI are positive signal in DIFF.
        #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
        #      PSF(DIFF) is coincident with PSF(REF), transients on SCI are still positive signal in DIFF.

        if NaNmask_U is not None:
            # ** Mask Union-NaN region
            PixA_DIFF[NaNmask_U] = np.nan
        if ConvdSide == 'SCI': 
            # ** Flip difference when science is convolved
            PixA_DIFF = -PixA_DIFF
        
        # * Save difference image
        if FITS_DIFF is not None:
            _hdl = fits.open(FITS_SCI)
            _hdl[0].data[:, :] = PixA_DIFF.T
            _hdl[0].header['NAME_REF'] = (pa.basename(FITS_REF), 'MeLOn: SFFT')
            _hdl[0].header['NAME_SCI'] = (pa.basename(FITS_SCI), 'MeLOn: SFFT')
            _hdl[0].header['KERORDER'] = (KerPolyOrder, 'MeLOn: SFFT')
            _hdl[0].header['BGORDER'] = (BGPolyOrder, 'MeLOn: SFFT')
            _hdl[0].header['CPHOTR'] = (str(ConstPhotRatio), 'MeLOn: SFFT')
            _hdl[0].header['KERHW'] = (KerHW, 'MeLOn: SFFT')
            _hdl[0].header['CONVD'] = (ConvdSide, 'MeLOn: SFFT')
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
        
        return Solution, PixA_DIFF
