import os
import sys
import os.path as pa
from CustomizedPacket4Spline import Customized_Packet

"""
* Updates in Version 1.1+

-backend (1.0.*) > -BACKEND_4SUBTRACT (1.1+)
-CUDA_DEVICE (1.0.*) > -CUDA_DEVICE_4SUBTRACT (1.1+)
-NUM_CPU_THREADS (1.0.*) > -NUM_CPU_THREADS_4SUBTRACT (1.1+)

"""

BACKEND_4SUBTRACT = 'Numpy'    
CUDA_DEVICE_4SUBTRACT = '0'     
NUM_CPU_THREADS_4SUBTRACT = 8   

ForceConv = 'REF'
GKerHW = 4

GDIR = '/data1/LeiHu/MESS/SFFTpaper/SplineForm'
FITS_REF = GDIR + '/MDATA/I.fits'           # FIXME, reference image
FITS_SCI = GDIR + '/MDATA/J.fits'           # FIXME, science image
FITS_mREF = GDIR + '/MDATA/mI.fits'         # FIXME, masked reference image
FITS_mSCI = GDIR + '/MDATA/mJ.fits'         # FIXME, masked science image
FITS_DIFF = GDIR + '/MDATA/D-N42.fits'      # FIXME, output difference image
FITS_Solution = GDIR + '/MDATA/So-N42.fits' # FIXME, output kernel solution

# Kernel: we use BiQuadratic B-Spline with number_division = 4 * 4, spline degree = 2.
# Background: we still use a polynomial form.
# NOTE: Here -KerPolyOrder is an invalid parameter, just a placeholder.

Customized_Packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, FITS_mSCI=FITS_mSCI, \
    ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=FITS_Solution, \
    KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=False, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
