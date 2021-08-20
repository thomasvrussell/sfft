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
    def ECP(FITS_REF, FITS_SCI, FITS_DIFF=None, FITS_Solution=None, ForceConv=None, GKerHW=None, KerHWRatio=2.0, \
        KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8, \
        MaskSatContam=False, BACKSIZE_SUPER=128, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        DETECT_THRESH=5.0, StarExt_iter=2, GLockFile=None):

        # * Perform Crowded-Prep [MaskSat]
        SFFTPrepDict = Auto_CrowdedPrep(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI).\
            MaskSat(BACKSIZE_SUPER=BACKSIZE_SUPER, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            DETECT_THRESH=DETECT_THRESH, StarExt_iter=StarExt_iter)
        
        # * Determine ConvdSide & KerHW
        FWHM_REF = SFFTPrepDict['FWHM_REF']
        FWHM_SCI = SFFTPrepDict['FWHM_SCI']

        if ForceConv is None:
            if FWHM_SCI >= FWHM_REF: ConvdSide = 'REF'
            else: ConvdSide = 'SCI'
        else: ConvdSide = ForceConv

        if GKerHW is None:
            FWHM_La = np.max([FWHM_REF, FWHM_SCI])
            KerHW = int(np.clip(KerHWRatio * FWHM_La, 2.0, 15.0))
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
                print('MeLOn Report: Ultimate Subtraction Takes [%.2f s]' %(time.time() - Tsub_start))
                
                # * Modifications on difference image
                #   a) when REF is convolved, DIFF = SCI - Conv(REF)
                #       PSF_DIFF is coincident with SCI, transients on SCI are positive signal in DIFF.
                #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
                #       PSF_DIFF is coincident with REF, transients on SCI are still positive signal in DIFF.

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

