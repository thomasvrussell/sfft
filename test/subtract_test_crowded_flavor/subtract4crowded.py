import os.path as pa
from sfft.EasyCrowdedPacket import Easy_CrowdedPacket

# configuration: computing backend and resourse 
BACKEND_4SUBTRACT = 'Cupy'      # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use 'Numpy' if you only have CPUs
CUDA_DEVICE_4SUBTRACT = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS_4SUBTRACT = 8   # FIXME ONLY work for backend Numpy   

# configuration: how to subtract
ForceConv = 'REF'               # FIXME {'AUTO', 'REF', 'SCI'}, where AUTO mode will avoid deconvolution
KerHWRatio = 2.0                # FIXME Ratio of kernel half width to FWHM (typically, 1.5-2.5).
KerPolyOrder = 2                # FIXME {0, 1, 2, 3}, Polynomial degree of kernel spatial variation
BGPolyOrder = 2                 # FIXME {0, 1, 2, 3}, Polynomial degree of differential background spatial variation.
                                #       it is non-trivial here, as crowded-flavor-sfft does not require sky subtraction in advance.
ConstPhotRatio = True           # FIXME Constant photometric ratio between images?

# configuration: required info in FITS header
GAIN_KEY = 'GAIN'               # NOTE Keyword of Gain in FITS header
SATUR_KEY = 'SATURATE'          # NOTE Keyword of Saturation in FITS header

# run sfft subtraction (crowded-flavor)
CDIR = pa.dirname(pa.abspath(__file__))
FITS_REF = CDIR + '/input_data/ztf_001735_zg_c01_q2_refimg.resampled.mini.fits'          
FITS_SCI = CDIR + '/input_data/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.mini.fits'   
FITS_DIFF = CDIR + '/output_data/sfft_diff.fits'

# NOTE: see complete descriptions of the parameters via help(sfft.Easy_CrowdedPacket).
Easy_CrowdedPacket.ECP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    ForceConv=ForceConv, GKerHW=None, KerHWRatio=KerHWRatio, KerHWLimit=(2, 20), KerPolyOrder=KerPolyOrder, \
    BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, MaskSatContam=False, GAIN_KEY=GAIN_KEY, \
    SATUR_KEY=SATUR_KEY, BACK_TYPE='AUTO', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, \
    DETECT_THRESH=5.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
    ONLY_FLAGS=None, BoundarySIZE=0.0, BACK_SIZE_SUPER=128, StarExt_iter=2, PriorBanMask=None, \
    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
print('MeLOn CheckPoint: TEST FOR CROWDED-FLAVOR-SFFT SUBTRACTION DONE!\n')

"""
# **** **** **** **** PARAMETER CHANGE LOG **** **** **** **** #

PARAMETERS NAME CHANGED:
  -backend (v1.0.*) >>>> -BACKEND_4SUBTRACT (v1.1+)
  -CUDA_DEVICE (v1.0.*) >>>> -CUDA_DEVICE_4SUBTRACT (v1.1+)
  -NUM_CPU_THREADS (v1.0.*) >>>> -NUM_CPU_THREADS_4SUBTRACT (v1.1+)
  -ForceConv = None (v1.0.*) >>>> -ForceConv = 'AUTO' (v1.1+)

REMOVED PARAMETERS:
  -GLockFile (v1.0.*) >>>> REMOVED (v1.1+)

NEW PARAMETERS:
  (1) more complete SExtractor configurations (see sfft.utils.pyAstroMatic.PYSEx)
  >>>> -BACK_TYPE (v1.2+)
  >>>> -BACK_VALUE (v1.2+)
  >>>> -BACK_SIZE (v1.2+)
  >>>> -BACK_FILTERSIZE (v1.2+)
  >>>> -DETECT_MINAREA (v1.2+)
  >>>> -DETECT_MAXAREA (v1.2+)
  >>>> -DEBLEND_MINCONT (v1.2+)
  >>>> -BACKPHOTO_TYPE (v1.2+)
  >>>> -ONLY_FLAGS (v1.2+)
  >>>> -BoundarySIZE (v1.2+)

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

"""
# **** **** **** **** ABOUT 4check **** **** **** **** #
# [1] The sfft difference in the directory 4check was generated with MaskSatContam=True, 
#     then sfft will mask the saturation-contaminated regions by NaN. It is 
#     only for better visually comparison to ZOGY difference. 
# [2] In sfft, MaskSatContam is not computing economical, especially for a survey program.
#     Here, I have disabled MaskSatContam (use default MaskSatContam=False) in the test. 
#     In practice, One may have better alternatives to identify the contaminated regions 
#     on the difference image, e.g., using data quality masks.
#
"""
