import os.path as pa
from sfft.EasySparsePacket import Easy_SparsePacket

# configuration: computing backend and resourse
BACKEND_4SUBTRACT = 'Cupy'      # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use Numpy if you only have CPUs
CUDA_DEVICE_4SUBTRACT = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS_4SUBTRACT = 8   # FIXME ONLY work for backend Numpy   

# configuration: how to subtract
ForceConv = 'REF'               # FIXME {'AUTO', 'REF', 'SCI'}, where AUTO mode will avoid deconvolution
KerHWRatio = 2.0                # FIXME Ratio of kernel half-width to FWHM (typically, 1.5-2.5).
KerPolyOrder = 2                # FIXME {0, 1, 2, 3}, Polynomial degree of kernel spatial variation
BGPolyOrder = 0                 # FIXME {0, 1, 2, 3}, Polynomial degree of differential background spatial variation.
                                #       it is trivial here, as sparse-flavor-sfft requires sky subtraction in advance.
ConstPhotRatio = True           # FIXME Constant photometric ratio between images?

# configuration: required info in FITS header
GAIN_KEY = 'GAIN'               # NOTE Keyword of Gain in FITS header
SATUR_KEY = 'ESATUR'            # NOTE Keyword of Effective Saturation in FITS header

# run sfft subtraction
CDIR = pa.dirname(pa.abspath(__file__))
FITS_REF = CDIR + '/input_data/c4d_180806_052738_ooi_i_v1.S20.skysub.resampled.fits'
FITS_SCI = CDIR + '/input_data/c4d_180802_062754_ooi_i_v1.S20.skysub.fits'
FITS_DIFF = CDIR + '/output_data/sfft_diff.fits'

# NOTE: see complete descriptions of the parameters via help(sfft.Easy_SparsePacket)
Easy_SparsePacket.ESP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    ForceConv=ForceConv, GKerHW=None, KerHWRatio=KerHWRatio, KerHWLimit=(2, 20), KerPolyOrder=KerPolyOrder, \
    BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, MaskSatContam=False, GAIN_KEY=GAIN_KEY, \
    SATUR_KEY=SATUR_KEY, BACK_TYPE='MANUAL', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, \
    DETECT_THRESH=2.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
    ONLY_FLAGS=[0], BoundarySIZE=30, XY_PriorSelect=None, Hough_FRLowerLimit=0.1, BeltHW=0.2, \
    PS_ELLIPThresh=0.3, MatchTol=None, MatchTolFactor=3.0, COARSE_VAR_REJECTION=True, CVREJ_MAGD_THRESH=0.12, \
    ELABO_VAR_REJECTION=True, EVREJ_RATIO_THREH=5.0, EVREJ_SAFE_MAGDEV=0.04, StarExt_iter=4, \
    XY_PriorBan=None, PostAnomalyCheck=False, PAC_RATIO_THRESH=5.0, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
print('MeLOn CheckPoint: TEST FOR SPARSE-FLAVOR-SFFT SUBTRACTION DONE!')

"""
# **** **** **** **** PARAMETER CHANGE LOG **** **** **** **** #

PARAMETERS NAME CHANGED:
  -backend (v1.0.*) >>>> -BACKEND_4SUBTRACT (v1.1+)
  -CUDA_DEVICE (v1.0.*) >>>> -CUDA_DEVICE_4SUBTRACT (v1.1+)
  -NUM_CPU_THREADS (v1.0.*) >>>> -NUM_CPU_THREADS_4SUBTRACT (v1.1+)
  -ForceConv = None (v1.0.*) >>>> -ForceConv = 'AUTO' (v1.1+)
  -MAGD_THRESH (v1.0, v1.1, v1.2) >>>> -CVREJ_MAGD_THRESH (v1.3+)
  -CheckPostAnomaly (v1.0, v1.1, v1.2) >>>> -PostAnomalyCheck (v1.3+)
  -PARATIO_THRESH (v1.0, v1.1, v1.2) >>>> -PAC_RATIO_THRESH (v1.3+)

PARAMETERS DEFUALT-VALUE CHANGED:
  -BGPolyOrder=2 (v1.0, v1.1, v1.2) >>>> -BGPolyOrder=0 (v1.3+)

REMOVED PARAMETERS:
  -GLockFile (v1.0.*) >>>> REMOVED (v1.1+)
  -trSubtract (1.0.*) >>>> REMOVED (v1.1+)

NEW PARAMETERS:
  (1) more complete SExtractor (see sfft.utils.pyAstroMatic.PYSEx) configurations
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

  (2) allows for source cross-match with given tolerance
  >>>> -MatchTol (v1.2.3+)

  (3) allows for user-defined source selection
  >>>> -XY_PriorSelect (v1.2+)

  (4) allows coarse variable rejection can be T/F
  >>>> -COARSE_VAR_REJECTION (v1.3+)

  (5) allows for elaborate varaible rejection
  >>>> ELABO_VAR_REJECTION (v1.3+)
  >>>> EVREJ_RATIO_THREH (v1.3+)
  >>>> EVREJ_SAFE_MAGDEV (v1.3+)

"""

"""
# **** **** **** **** ABOUT 4check **** **** **** **** #
# [1] The sfft difference in the directory 4check was generated with MaskSatContam=True, 
#     then sfft will mask the saturation-contaminated regions by NaN. It is 
#     only for better visually comparison to HOTPANTS difference. 
# [2] In sfft, MaskSatContam is not computing economical, especially for a survey program.
#     Here, I have disabled MaskSatContam (use default MaskSatContam=False) in the test. 
#     In practice, One may have better alternatives to identify the contaminated regions 
#     on the difference image, e.g., using data quality masks.
#
"""
