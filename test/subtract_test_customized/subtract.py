import os
import sys
import numpy as np
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
from sfft.CustomizedPacket import Customized_Packet
CDIR = pa.dirname(pa.abspath(__file__))

# * configurations 
backend = 'Cupy'      # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use Numpy if you only have CPUs
CUDA_DEVICE = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS = 8   # FIXME ONLY work for backend Numpy   
ForceConv = 'REF'     # FIXME {REF', 'SCI'}
GKerHW = 4            # FIXME given kernel half-width

# * input data
# NOTE 'mini' region is selected near M31 center:
#      Centred at (RA=10.801538, DEC=41.385839) with cutout size 1024*1024 pix
#
# NOTE You can find the raw ZTF observations (full 3K*3K) by
#      Download Full REF: 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/001/
#                          field001735/zg/ccd01/q2/ztf_001735_zg_c01_q2_refimg.fits'
#      Download Full SCI: 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/
#                          0705/481609/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.fits'
#      Download Full ZOGY-DIFF: https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/
#                               0705/481609/ztf_20180705481609_001735_zg_c01_o_q2_scimrefdiffimg.fits.fz
#      Use SWarp to align REF to SCI, then you are able to apply this script on the full version.
#      These files are not provided here to minimize the size of the package.

FITS_REF = CDIR + '/input_data/ztf_001735_zg_c01_q2_refimg.resampled.mini.fits'
FITS_SCI = CDIR + '/input_data/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.mini.fits'
FITS_mREF = FITS_REF[:-5] + '.masked.fits'
FITS_mSCI = FITS_SCI[:-5] + '.masked.fits'

# * trigger a GPU warming-up 
#   NOTE: the warming-up itself is not an ingredient of the image subtraction.
#         However, I put it here to demonstrate the true speed of SFFT when GPU warming-up is done.
#         In a survey program, the number of image subtraction tasks is typically >> 1,
#         the extra time-cost on the GPU "warming-up" only exists for the first task of a processing queue.

_tmp = fits.getheader(FITS_REF, ext=0)
NX, NY = int(_tmp['NAXIS1']), int(_tmp['NAXIS2'])

class HidePrint:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def GPU_wraming_up():
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

    Customized_Packet.CP(FITS_REF=_FITS_REF, FITS_SCI=_FITS_SCI, FITS_mREF=_FITS_mREF, FITS_mSCI=_FITS_mSCI, \
        ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=None, FITS_Solution=None, \
        KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, backend=backend, \
        CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS, GLockFile=None)
    os.system('rm -rf %s' %TDIR)
    return None

if backend in ['Cupy', 'Pycuda']:
    print('\n---------------------------------- GPU warming-up --------------------------------------')
    print('MeLOn CheckPoint: GPU warming-up [START]!')
    print('MeLOn Note: GPU warming-up is not an ingredient of SFFT.')
    print('MeLOn Note: Instead, it is used to finish some GPU initializations before SFFT is called.')
    with HidePrint():
        GPU_wraming_up()
    print('MeLOn CheckPoint: GPU warming-up [END]!')
    print('---------------------------------- GPU warming-up --------------------------------------\n')

# * trigger a SFFT subtraction 
print('\n********************************** SFFT subtraction **************************************')
print('MeLOn CheckPoint: SFFT subtraction [START]!')
FITS_DIFF = CDIR + '/output_data/sfft_diff.fits'
Customized_Packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, FITS_mSCI=FITS_mSCI, \
    ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, backend=backend, \
    CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS, GLockFile=None)

if pa.exists(FITS_DIFF):
    print('MeLOn CheckPoint: SFFT subtraction [END]!')
else: print('MeLOn CheckPoint: SFFT subtraction [FAIL]!')
print('********************************** SFFT subtraction **************************************\n')
