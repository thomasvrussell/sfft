import os
import sys
import os.path as pa
from CustomizedPacket4Spline import Customized_Packet

backend = 'Numpy'    
CUDA_DEVICE = '0'     
NUM_CPU_THREADS = 8   
ForceConv = 'REF'     
GKerHW = 4

GDIR = '/data1/LeiHu/MESS/SFFTpaper/SplineForm'
FITS_REF = GDIR + '/MDATA/I.fits'           # FIXME, reference image
FITS_SCI = GDIR + '/MDATA/J.fits'           # FIXME, science image
FITS_mREF = GDIR + '/MDATA/mI.fits'         # FIXME, masked reference image
FITS_mSCI = GDIR + '/MDATA/mJ.fits'         # FIXME, masked science image
FITS_DIFF = GDIR + '/MDATA/D-N42.fits'      # FIXME, output difference image
FITS_Solution = GDIR + '/MDATA/So-N42.fits' # FIXME, output kernel solution

Customized_Packet.CP(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, FITS_mREF=FITS_mREF, FITS_mSCI=FITS_mSCI, \
    ForceConv=ForceConv, GKerHW=GKerHW, FITS_DIFF=FITS_DIFF, FITS_Solution=FITS_Solution, \
    KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=False, backend=backend, \
    CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS, GLockFile=None)

