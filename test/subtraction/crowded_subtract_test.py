import os
import os.path as pa
from sfft.EasyCrowdedPacket import Easy_CrowdedPacket

backend = 'Pycuda'    # FIXME {'Pycuda', 'Cupy', 'Numpy'}
CUDA_DEVICE = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS = 8   # FIXME ONLY work for backend Numpy   

ForceConv = 'REF'     # FIXME which image is convolved {None, 'REF', 'SCI'} | None mean AUTO and avoids deconvolution

GAIN_KEY = 'GAIN'
SATUR_KEY = 'SATURATE'
CDIR = pa.dirname(pa.abspath(__file__))

FITS_REF = CDIR + '/crowded_example_data/ztf_alpha.resampled_reference.fits'
FITS_SCI = CDIR + '/crowded_example_data/ztf_beta.science.fits'
FITS_DIFF = CDIR + '/crowded_example_data/sfft_difference.fits'  # NOTE Overwrite

Easy_CrowdedPacket.ECP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_DIFF=FITS_DIFF, FITS_Solution=None, \
    ForceConv='REF', GKerHW=None, KerHWRatio=2.0, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
    backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS, MaskSatContam=True, \
    BACKSIZE_SUPER=128, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, DETECT_THRESH=5.0, StarExt_iter=2, GLockFile=None)
