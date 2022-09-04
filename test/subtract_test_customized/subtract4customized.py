import os
import sys
import numpy as np
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
from sfft.CustomizedPacket import Customized_Packet
CDIR = pa.dirname(pa.abspath(__file__))
# sfft version: 1.3.2+

# configuration: computing backend and resourse
BACKEND_4SUBTRACT = 'Cupy'      # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use Numpy if you only have CPUs
CUDA_DEVICE_4SUBTRACT = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS_4SUBTRACT = 8   # FIXME ONLY work for backend Numpy  

# configuration: how to subtract
ForceConv = 'REF'               # FIXME {'REF', 'SCI'}
GKerHW = 4                      # FIXME given kernel half-width
KerPolyOrder = 2                # FIXME {0, 1, 2, 3}, Polynomial degree of kernel spatial variation
BGPolyOrder = 2                 # FIXME {0, 1, 2, 3}, Polynomial degree of differential background spatial variation
ConstPhotRatio = True           # FIXME Constant photometric ratio between images?
 
FITS_REF = CDIR + '/input_data/ztf_001735_zg_c01_q2_refimg.resampled.mini.fits'
FITS_SCI = CDIR + '/input_data/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.mini.fits'
FITS_mREF = FITS_REF[:-5] + '.masked.fits'
FITS_mSCI = FITS_SCI[:-5] + '.masked.fits'

if BACKEND_4SUBTRACT in ['Cupy', 'Pycuda']:
    print('\n---------------------------------- GPU warming-up --------------------------------------')
    print('MeLOn CheckPoint: GPU warming-up [START]!')
    print('MeLOn Note: GPU warming-up is not an ingredient of SFFT.')
    print('MeLOn Note: Instead, it is used to finish some GPU initializations before SFFT is called.')
    print('MeLOn Note: When process a queue of tasks, GPU "warming-up" only exists for the first task.')

    class HidePrint:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    with HidePrint():
        _tmp = fits.getheader(FITS_REF, ext=0)
        NX, NY = int(_tmp['NAXIS1']), int(_tmp['NAXIS2'])
        _PixA_REF = np.random.uniform(0.9, 1.1, (NX, NY)).astype(float)
        _PixA_SCI = np.random.uniform(9.9, 10.1, (NX, NY)).astype(float)
        _PixA_mREF = _PixA_REF.copy()
        _PixA_mSCI = _PixA_SCI.copy()
        
        TDIR = mkdtemp(suffix=None, prefix=None, dir=None)
        _FITS_REF = TDIR + '/ref.fits'
        fits.HDUList([fits.PrimaryHDU(_PixA_REF.T)]).writeto(_FITS_REF, overwrite=True)
        _FITS_SCI = TDIR + '/sci.fits'
        fits.HDUList([fits.PrimaryHDU(_PixA_SCI.T)]).writeto(_FITS_SCI, overwrite=True)
        _FITS_mREF = TDIR + '/mref.fits'
        fits.HDUList([fits.PrimaryHDU(_PixA_mREF.T)]).writeto(_FITS_mREF, overwrite=True)
        _FITS_mSCI = TDIR + '/msci.fits'
        fits.HDUList([fits.PrimaryHDU(_PixA_mSCI.T)]).writeto(_FITS_mSCI, overwrite=True)

        Customized_Packet.CP(FITS_REF=_FITS_REF, FITS_SCI=_FITS_SCI, FITS_mREF=_FITS_mREF, \
            FITS_mSCI=_FITS_mSCI, ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=None, FITS_Solution=None, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
        os.system('rm -rf %s' %TDIR)

    print('MeLOn CheckPoint: GPU warming-up [END]!')
    print('---------------------------------- GPU warming-up --------------------------------------\n')

# run sfft subtraction
print('\n********************************** SFFT subtraction **************************************')
print('MeLOn CheckPoint: SFFT subtraction [START]!')
FITS_DIFF = CDIR + '/output_data/sfft_diff.fits'

# NOTE: see complete descriptions of the parameters via help(sfft.Customized_Packet)
Customized_Packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, \
    FITS_mSCI=FITS_mSCI, ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)

print('MeLOn CheckPoint: SFFT subtraction [END]!')
print('********************************** SFFT subtraction **************************************\n')
print('MeLOn CheckPoint: TEST FOR CUSTOMIZED-SFFT SUBTRACTION DONE!\n')

"""
# **** **** **** **** PARAMETER CHANGE LOG **** **** **** **** #

PARAMETERS NAME CHANGED:
  -backend (v1.0.*) >>>> -BACKEND_4SUBTRACT (v1.1+)
  -CUDA_DEVICE (v1.0.*) >>>> -CUDA_DEVICE_4SUBTRACT (v1.1+)
  -NUM_CPU_THREADS (v1.0.*) >>>> -NUM_CPU_THREADS_4SUBTRACT (v1.1+)

REMOVED PARAMETERS:
  -GLockFile (v1.0.*) >>>> REMOVED (v1.1+)

"""

"""
# **** **** **** **** ABOUT TEST DATA **** **** **** **** #
#
# [1] You can find the raw ZTF observations (full 3K*3K) by
#     Download Full REF: 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/001/
#                         field001735/zg/ccd01/q2/ztf_001735_zg_c01_q2_refimg.fits'
#     Download Full SCI: 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/
#                         0705/481609/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.fits'
#     Download Full ZOGY-DIFF: https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/
#                              0705/481609/ztf_20180705481609_001735_zg_c01_o_q2_scimrefdiffimg.fits.fz
#  
# [2] Here the test only focus on a 'mini' region selected near the M31 center:
#     At (RA=10.801538, DEC=41.385839) with cutout size 1024*1024 pix 
#     REF image has been aligned to SCI by AstroMatic SWarp.
#
"""
