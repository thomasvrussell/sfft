import os
import sys
import os.path as pa
BSDIR = pa.dirname(pa.dirname(pa.abspath(__file__)))
sys.path.insert(1, BSDIR)
from CustomizedPacket4Spline import Customized_Packet as spline_packet

"""
* Updates in Version 1.1+

-backend (1.0.*) > -BACKEND_4SUBTRACT (1.1+)
-CUDA_DEVICE (1.0.*) > -CUDA_DEVICE_4SUBTRACT (1.1+)
-NUM_CPU_THREADS (1.0.*) > -NUM_CPU_THREADS_4SUBTRACT (1.1+)

"""

BACKEND_4SUBTRACT = 'Numpy'      # ONLY Numpy availbale now
CUDA_DEVICE_4SUBTRACT = '0'      # invalid parameter for Numpy backend
NUM_CPU_THREADS_4SUBTRACT = 8   

ForceConv = 'REF'
GKerHW = 4

# just use the same example as ./test/subtract_test_customized
datadir = pa.join(BSDIR, 'spline_test', 'input_data')
FITS_REF = datadir + '/ztf_001735_zg_c01_q2_refimg.resampled.mini.fits'              # FIXME, reference image
FITS_SCI = datadir + '/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.mini.fits'       # FIXME, science image
FITS_mREF = FITS_REF[:-5] + '.masked.fits'                                           # FIXME, masked reference image
FITS_mSCI = FITS_SCI[:-5] + '.masked.fits'                                           # FIXME, masked science image
FITS_DIFF = 'sfft_difference_spline.fits'                                            # FIXME, output difference image
FITS_Solution = 'sfft_solution_spline.fits'                                          # FIXME, output kernel solution

# Kernel: we use BiQuadratic B-Spline with number_division = 4 * 4, spline degree = 2.
# Background: we still use a polynomial form.
# NOTE: Here -KerPolyOrder is an invalid parameter, just a placeholder.
# NOTE: I yet to consider how to deal with the PhotRatio in spline form.
#       Here I set ConstPhotRatio = Fasle.
# NOTE: it will take ~1.5 min and ~80 GB memory.

spline_packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, FITS_mSCI=FITS_mSCI, \
    ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=FITS_Solution, \
    KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=False, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
