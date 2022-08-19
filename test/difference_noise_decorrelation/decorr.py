import os
import numpy as np
import os.path as pa
from astropy.io import fits
from astropy.convolution import convolve
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator
CDIR = pa.dirname(pa.abspath(__file__))

# ** Science Group
FITS_04a = CDIR + '/input_data/DEC-OBS04a.mini.fits'
FITS_04b = CDIR + '/input_data/DEC-OBS04b.mini.fits'
FITS_04c = CDIR + '/input_data/DEC-OBS04c.mini.fits'
FITS_04d = CDIR + '/input_data/DEC-OBS04d.mini.fits'
FITS_04e = CDIR + '/input_data/DEC-OBS04e.mini.fits'

# ** Reference Group
FITS_18a = CDIR + '/input_data/DEC-OBS18a.mini.fits'
FITS_18b = CDIR + '/input_data/DEC-OBS18b.mini.fits'
FITS_18c = CDIR + '/input_data/DEC-OBS18c.mini.fits'
FITS_18d = CDIR + '/input_data/DEC-OBS18d.mini.fits'
FITS_18e = CDIR + '/input_data/DEC-OBS18e.mini.fits'

# ** Match Kernels
FITS_MK_04a = None
FITS_MK_04b = CDIR + '/input_data/DEC-OBS04b.MatchKernel.fits'
FITS_MK_04c = CDIR + '/input_data/DEC-OBS04c.MatchKernel.fits'
FITS_MK_04d = CDIR + '/input_data/DEC-OBS04d.MatchKernel.fits'
FITS_MK_04e = CDIR + '/input_data/DEC-OBS04e.MatchKernel.fits'
FITS_MK_18a = None
FITS_MK_18b = CDIR + '/input_data/DEC-OBS18b.MatchKernel.fits'
FITS_MK_18c = CDIR + '/input_data/DEC-OBS18c.MatchKernel.fits'
FITS_MK_18d = CDIR + '/input_data/DEC-OBS18d.MatchKernel.fits'
FITS_MK_18e = CDIR + '/input_data/DEC-OBS18e.MatchKernel.fits'
FITS_FinMK = CDIR + '/input_data/FinalMatchKernel.fits'

# *** File Path for Outputs
FITS_Stack04 = CDIR + '/output_data/Stack-DEC-OBS04.fits'
FITS_Stack18 = CDIR + '/output_data/Stack-DEC-OBS18.fits'
FITS_FinDIFF = CDIR + '/output_data/FinalDifference.fits'

FITS_DeCorrKer = CDIR + '/output_data/DeCorrKernel.fits'
FITS_dcFinDIFF = CDIR + '/output_data/FinalDifference.DeCorr.fits'

def func_preproc():

    # ** Stack Science Group
    PixA_Stack04 = []
    _SEQ1 = [FITS_04a, FITS_04b, FITS_04c, FITS_04d, FITS_04e]
    _SEQ2 = [FITS_MK_04a, FITS_MK_04b, FITS_MK_04c, FITS_MK_04d, FITS_MK_04e]
    for FITS_IMG, FITS_MK in zip(_SEQ1, _SEQ2):
        if FITS_MK is not None:
            _PixA = fits.getdata(FITS_IMG, ext=0).T
            _MK = fits.getdata(FITS_MK, ext=0).T
            _CPixA = convolve(_PixA, _MK, boundary='extend', normalize_kernel=False)
        else: _CPixA = fits.getdata(FITS_IMG, ext=0).T
        PixA_Stack04.append(_CPixA)
    PixA_Stack04 = np.array(PixA_Stack04)
    PixA_Stack04 = np.median(PixA_Stack04, axis=0)

    hdl = fits.open(FITS_04a)
    hdl[0].data[:, :] = PixA_Stack04.T
    hdl.writeto(FITS_Stack04, overwrite=True)
    hdl.close()

    # ** Stack Reference Group
    PixA_Stack18 = []
    _SEQ1 = [FITS_18a, FITS_18b, FITS_18c, FITS_18d, FITS_18e]
    _SEQ2 = [FITS_MK_18a, FITS_MK_18b, FITS_MK_18c, FITS_MK_18d, FITS_MK_18e]
    for FITS_IMG, FITS_MK in zip(_SEQ1, _SEQ2):
        if FITS_MK is not None:
            _PixA = fits.getdata(FITS_IMG, ext=0).T
            _MK = fits.getdata(FITS_MK, ext=0).T
            _CPixA = convolve(_PixA, _MK, boundary='extend', normalize_kernel=False)
        else: _CPixA = fits.getdata(FITS_IMG, ext=0).T
        PixA_Stack18.append(_CPixA)
    PixA_Stack18 = np.array(PixA_Stack18)
    PixA_Stack18 = np.median(PixA_Stack18, axis=0)

    hdl = fits.open(FITS_18a)
    hdl[0].data[:, :] = PixA_Stack18.T
    hdl.writeto(FITS_Stack18, overwrite=True)
    hdl.close()

    # ** Final Subtraction
    MK_Fin = fits.getdata(FITS_FinMK, ext=0).T
    PixA_FinDIFF = PixA_Stack04 - convolve(PixA_Stack18, MK_Fin, boundary='extend', normalize_kernel=False)
    print('MeLOn CheckPoint: Final Subtraction is DONE !')

    hdl = fits.open(FITS_04a)
    hdl[0].data[:, :] = PixA_FinDIFF.T
    hdl.writeto(FITS_FinDIFF, overwrite=True)
    hdl.close()

    return None

def func_decorr():

    # ** Collect SkySig & Match Kernels for Science Group
    SkySig_SLst, MK_SLst = [], []
    _SEQ1 = [FITS_04a, FITS_04b, FITS_04c, FITS_04d, FITS_04e]
    _SEQ2 = [FITS_MK_04a, FITS_MK_04b, FITS_MK_04c, FITS_MK_04d, FITS_MK_04e]
    for FITS_IMG, FITS_MK in zip(_SEQ1, _SEQ2):
        _PixA = fits.getdata(FITS_IMG, ext=0).T
        _skysig = SkyLevel_Estimator.SLE(PixA_obj=_PixA)[1]
        if FITS_MK is not None:
            _MK = fits.getdata(FITS_MK, ext=0).T
        else: _MK = None
        SkySig_SLst.append(_skysig)
        MK_SLst.append(_MK)
    
    # ** Collect SkySig & Match Kernels for Referenece Group
    SkySig_RLst, MK_RLst = [], []
    _SEQ1 = [FITS_18a, FITS_18b, FITS_18c, FITS_18d, FITS_18e]
    _SEQ2 = [FITS_MK_18a, FITS_MK_18b, FITS_MK_18c, FITS_MK_18d, FITS_MK_18e]
    for FITS_IMG, FITS_MK in zip(_SEQ1, _SEQ2):
        _PixA = fits.getdata(FITS_IMG, ext=0).T
        _skysig = SkyLevel_Estimator.SLE(PixA_obj=_PixA)[1]
        if FITS_MK is not None:
            _MK = fits.getdata(FITS_MK, ext=0).T
        else: _MK = None
        SkySig_RLst.append(_skysig)
        MK_RLst.append(_MK)

    # ** Read the Match Kernel for Final Subtraction
    MK_Fin = fits.getdata(FITS_FinMK, ext=0).T

    # ** Calculate DeCorrelation Kernel
    DeCorrKer = DeCorrelation_Calculator.DCC(MK_JLst=MK_SLst, SkySig_JLst=SkySig_SLst, \
        MK_ILst=MK_RLst, SkySig_ILst=SkySig_RLst, MK_Fin=MK_Fin, KERatio=2.0)
    fits.HDUList([fits.PrimaryHDU(DeCorrKer.T)]).writeto(FITS_DeCorrKer, overwrite=True)

    # ** Perform Noise DeCorrelation
    PixA_FinDIFF = fits.getdata(FITS_FinDIFF, ext=0).T
    PixA_dcFinDIFF = convolve(PixA_FinDIFF, DeCorrKer, boundary='extend', normalize_kernel=False)
    print('MeLOn CheckPoint: Noise Decorrelation is DONE !')

    hdl = fits.open(FITS_FinDIFF)
    hdl[0].data[:, :] = PixA_dcFinDIFF.T
    hdl.writeto(FITS_dcFinDIFF, overwrite=True)
    hdl.close()

    return None

func_preproc()
func_decorr()
print('MeLOn CheckPoint: TEST FOR SFFT DECORRELATION DONE!\n')
