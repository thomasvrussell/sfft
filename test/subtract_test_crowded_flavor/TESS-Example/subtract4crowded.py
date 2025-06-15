import os
import os.path as pa
from sfft.EasyCrowdedPacket import Easy_CrowdedPacket
# sfft version: 1.4.0+

# configuration: computing backend and resourse 
BACKEND_4SUBTRACT = 'Numpy' #'Cupy'      # FIXME {'Cupy', 'Numpy'}, Use 'Numpy' if you only have CPUs
CUDA_DEVICE_4SUBTRACT = '0'     # FIXME ONLY work for backend Cupy
NUM_CPU_THREADS_4SUBTRACT = 8   # FIXME ONLY work for backend Numpy   

# configuration: how to subtract
ForceConv = 'REF'               # FIXME {'AUTO', 'REF', 'SCI'}
                                # 'AUTO': convolve the image with smaller FWHM to avoid deconvolution. [default & recommended]
                                # 'REF': convolve the reference image, DIFF = SCI - Convolved_REF.
                                # 'SCI': convolve the science image, DIFF = Convolved_SCI - REF.

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
FITS_REF = CDIR + '/input_data/tess2018258205941-s0002-1-1-0121-s_ffic.001.resamp.stp.fits'          
FITS_SCI = CDIR + '/input_data/tess2018258225941-s0002-1-1-0121-s_ffic.001.stp.fits'   
FITS_DIFF = CDIR + '/output_data/sfft_diff.fits'

# NOTE: see complete descriptions of the parameters via help(sfft.Easy_CrowdedPacket).
Easy_CrowdedPacket.ECP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    ForceConv=ForceConv, GKerHW=None, KerHWRatio=KerHWRatio, KerHWLimit=(2, 15), KerPolyOrder=KerPolyOrder, \
    BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, MaskSatContam=True, GAIN_KEY=GAIN_KEY, \
    SATUR_KEY=SATUR_KEY, BACK_TYPE='AUTO', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, \
    DETECT_THRESH=5.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
    ONLY_FLAGS=None, BoundarySIZE=0.0, BACK_SIZE_SUPER=128, StarExt_iter=2, PriorBanMask=None, \
    BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
    NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=2)
print('MeLOn CheckPoint: TEST FOR CROWDED-FLAVOR-SFFT SUBTRACTION DONE!\n')
