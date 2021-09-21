import os
import time
import numpy as np
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
from astropy.time import Time
from sfft.utils.meta.FileLockKit import FileLock

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class Customized_Packet:
    @staticmethod
    def CP(FITS_REF, FITS_SCI, FITS_mREF, FITS_mSCI, ForceConv, GKerHW, FITS_DIFF=None, \
        FITS_Solution=None, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
        backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8, GLockFile=None):
        
        """
        * Parameters for Customized SFFT
        # Customized: Skip built-in image-masking process by given customized masked image-pair
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

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv []           # it determines which image will be convolved, can be 'REF' or 'SCI'.

        -GKerHW []              # The given kernel half-width. 

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

        -FITS_mREF []           # File path of input masked reference image (NaN-free)

        -FITS_mSCI []           # File path of input masked science image (NaN-free)

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

        # * Read input images
        PixA_REF = fits.getdata(FITS_REF, ext=0).T
        PixA_SCI = fits.getdata(FITS_SCI, ext=0).T
        PixA_mREF = fits.getdata(FITS_mREF, ext=0).T
        PixA_mSCI = fits.getdata(FITS_mSCI, ext=0).T

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
        
        ConvdSide = ForceConv
        KerHW = GKerHW
        if GLockFile is None:
            TDIR = mkdtemp(suffix=None, prefix='4lock', dir=None)
            LockFile = pa.join(TDIR, 'tmplock.txt')
        else: LockFile = GLockFile
        
        # * Configure SFFT
        from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure

        Tcomp_start = time.time()
        SFFTConfig = SingleSFFTConfigure.SSC(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)
        print('MeLOn Report: Compilation Takes [%.3f s]' %(time.time() - Tcomp_start))
        
        with FileLock(LockFile):
            with open(LockFile, "a") as f:
                LTIME0 = Time.now()

                # * Perform SFFT Subtraction
                from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract

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
                
                Tsub_start = time.time()
                _tmp = GeneralSFFTSubtract.GSS(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
                    SFFTConfig=SFFTConfig, ContamMask_I=None, backend=backend, \
                    CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)
                Solution, PixA_DIFF = _tmp[:2]
                print('MeLOn Report: Subtraction Takes [%.3f s]' %(time.time() - Tsub_start))
                
                # * Modifications on difference image
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
        
        return Solution, PixA_DIFF
