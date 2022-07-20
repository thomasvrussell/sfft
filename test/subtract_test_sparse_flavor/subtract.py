import os
import os.path as pa
from sfft.EasySparsePacket import Easy_SparsePacket

"""
* Updates in Version 1.2.3+

-MatchTol    # New parameter!

* Updates in Version 1.2+

-XY_PriorSelect    # New parameter!

* Updates in Version 1.1+

-backend (1.0.*) > -BACKEND_4SUBTRACT (1.1+)
-CUDA_DEVICE (1.0.*) > -CUDA_DEVICE_4SUBTRACT (1.1+)
-NUM_CPU_THREADS (1.0.*) > -NUM_CPU_THREADS_4SUBTRACT (1.1+)

-ForceConv = None (1.0.*) > -ForceConv = 'AUTO' (1.1+)

-trSubtract (1.0.*) > removed (1.1+)
-GLockFile > removed (1.1+)

"""

BACKEND_4SUBTRACT = 'Cupy'      # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use Numpy if you only have CPUs
CUDA_DEVICE_4SUBTRACT = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS_4SUBTRACT = 8   # FIXME ONLY work for backend Numpy   
ForceConv = 'REF'               # FIXME {'AUTO', 'REF', 'SCI'}, where AUTO mode will avoid deconvolution

MaskSatContam = False           # FIXME {True, False}, mask the saturation-contaminated regions by NaN on the difference?
                                # P.S. Although the difference in 4check is generated with MaskSatContam=True (just for better comparison),
                                #      Generally, I would recommend users to adopt the default False as it will be faster.
                                #      In practice, you may have good alternatives to find the contaminated regions, 
                                #      e.g., using data quality masks.

GAIN_KEY = 'GAIN'               # NOTE Keyword of Gain in FITS header
SATUR_KEY = 'ESATUR'            # NOTE Keyword of Saturation in FITS header

CDIR = pa.dirname(pa.abspath(__file__))
FITS_REF = CDIR + '/input_data/c4d_180806_052738_ooi_i_v1.S20.skysub.resampled.fits'
FITS_SCI = CDIR + '/input_data/c4d_180802_062754_ooi_i_v1.S20.skysub.fits'
FITS_DIFF = CDIR + '/output_data/my_sfft_difference.fits'

# *************************** IMPORTANT NOTICE *************************** #
#  I strongly recommend users to read the descriptions of the parameters 
#  via help(sfft.Easy_SparsePacket).
# *************************** IMPORTANT NOTICE *************************** #

Easy_SparsePacket.ESP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    ForceConv=ForceConv, GKerHW=None, KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, \
    ConstPhotRatio=True, MaskSatContam=MaskSatContam, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
    BACK_TYPE='MANUAL', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=2.0, \
    DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
    ONLY_FLAGS=[0], BoundarySIZE=30, XY_PriorSelect=None, Hough_FRLowerLimit=0.1, BeltHW=0.2, \
    PS_ELLIPThresh=0.3, MatchTol=None, MatchTolFactor=3.0, MAGD_THRESH=0.12, StarExt_iter=4, XY_PriorBan=None, \
    CheckPostAnomaly=False, PARATIO_THRESH=3.0, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
