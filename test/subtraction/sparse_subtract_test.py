import os
import os.path as pa
from sfft.EasySparsePacket import Easy_SparsePacket

backend = 'Pycuda'    # FIXME {'Pycuda', 'Cupy', 'Numpy'}
CUDA_DEVICE = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS = 8   # FIXME ONLY work for backend Numpy   

ForceConv = 'REF'     # FIXME which image is convolved {None, 'REF', 'SCI'} | None mean AUTO and avoids deconvolution
trSubtract = True     # FIXME trail subtraction (an iteration process) makes sfft slightly more accurate but slower

GAIN_KEY = 'GAIN'
SATUR_KEY = 'ESATUR'
CDIR = pa.dirname(pa.abspath(__file__))

FITS_REF = CDIR + '/sparse_example_data/decam_alpha.skysub.resampled_reference.fits'
FITS_SCI = CDIR + '/sparse_example_data/decam_beta.skysub.science.fits'
FITS_DIFF = CDIR + '/sparse_example_data/sfft_difference.fits'  # NOTE Overwrite

Easy_SparsePacket.ESP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
        ForceConv=ForceConv, GKerHW=None, KerHWRatio=2.0, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
        backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS, MaskSatContam=True, \
        GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, DETECT_THRESH=2.0, BoundarySIZE=30, MAGD_THRESH=0.12, \
        StarExt_iter=4, trSubtract=trSubtract, RATIO_THRESH=3.0, XY_PriorBan=None, GLockFile=None)
