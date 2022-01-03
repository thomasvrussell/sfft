import os
import os.path as pa
from sfft.EasySparsePacket import Easy_SparsePacket

backend = 'Pycuda'    # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use Numpy if you only have CPUs
CUDA_DEVICE = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS = 8   # FIXME ONLY work for backend Numpy   
ForceConv = 'REF'     # FIXME {None, 'REF', 'SCI'}, None mean AUTO mode that avoids deconvolution
trSubtract = False    # FIXME {True, False} do trail subtraction or not 
                      #       It is an iterative process to refine mask by further rejecting variables detected 
                      #       from difference image of a trial subtraction. This makes sfft (slightly) more accurate but slower.

GAIN_KEY = 'GAIN'     # NOTE Keyword of Gain in FITS header
SATUR_KEY = 'ESATUR'  # NOTE Keyword of Saturation in FITS header
CDIR = pa.dirname(pa.abspath(__file__))

FITS_REF = CDIR + '/input_data/c4d_180806_052738_ooi_i_v1.S20.skysub.resampled.fits'
FITS_SCI = CDIR + '/input_data/c4d_180802_062754_ooi_i_v1.S20.skysub.fits'
FITS_DIFF = CDIR + '/output_data/my_sfft_difference.fits'

Easy_SparsePacket.ESP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
        ForceConv=ForceConv, GKerHW=None, KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, \
        ConstPhotRatio=True, backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS, MaskSatContam=True, \
        GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, DETECT_THRESH=2.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, BoundarySIZE=30, \
        BeltHW=0.2, MAGD_THRESH=0.12, StarExt_iter=4, trSubtract=trSubtract, RATIO_THRESH=3.0, XY_PriorBan=None, GLockFile=None)
