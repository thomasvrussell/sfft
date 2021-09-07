import os
import os.path as pa
from sfft.EasyCrowdedPacket import Easy_CrowdedPacket

backend = 'Pycuda'    # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use Numpy if you only have CPUs
CUDA_DEVICE = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS = 8   # FIXME ONLY work for backend Numpy   
ForceConv = 'REF'     # FIXME {None, 'REF', 'SCI'}, None mean AUTO mode that avoids deconvolution

GAIN_KEY = 'GAIN'         # NOTE Keyword of Gain in FITS header
SATUR_KEY = 'SATURATE'    # NOTE Keyword of Saturation in FITS header
CDIR = pa.dirname(pa.abspath(__file__))

# NOTE 'mini' region is selected near M31 center:
#      Centred at (RA=10.801538, DEC=41.385839) with cutout size 1024*1024 pix
#
# NOTE You can find the raw ZTF observations (full 3K*3K) by
#      Download Full REF: 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/001/field001735/zg/ccd01/q2/ztf_001735_zg_c01_q2_refimg.fits'
#      Download Full SCI: 'https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/0705/481609/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.fits'
#      Download Full ZOGY-DIFF: https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/2018/0705/481609/ztf_20180705481609_001735_zg_c01_o_q2_scimrefdiffimg.fits.fz
#      Use SWarp to align REF to SCI, then you are able to apply this script on the full version.
#      These files are not provided here to minimize the size of the package.

FITS_REF = CDIR + '/input_data/ztf_001735_zg_c01_q2_refimg.resampled.mini.fits'          
FITS_SCI = CDIR + '/input_data/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.mini.fits'   
FITS_DIFF = CDIR + '/output_data/your_sfft_difference.fits'

Easy_CrowdedPacket.ECP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    ForceConv='REF', GKerHW=None, KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
    backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS, MaskSatContam=True, \
    BACKSIZE_SUPER=128, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, DETECT_THRESH=5.0, \
    StarExt_iter=2, PriorBanMask=None, GLockFile=None)
