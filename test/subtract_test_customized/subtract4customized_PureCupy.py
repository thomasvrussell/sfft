import os
import sys
import cupy as cp
import numpy as np
import os.path as pa
from astropy.io import fits
from sfft.PureCupyCustomizedPacket import PureCupy_Customized_Packet
CDIR = pa.dirname(pa.abspath(__file__))
# sfft version: 1.6.0+

# configuration: computing backend and resourse
BACKEND_4SUBTRACT = 'Cupy'      # FIXME {'Pycuda', 'Cupy', 'Numpy'}, Use Numpy if you only have CPUs
CUDA_DEVICE_4SUBTRACT = '0'     # FIXME ONLY work for backend Pycuda / Cupy
NUM_CPU_THREADS_4SUBTRACT = 8   # FIXME ONLY work for backend Numpy  

# configuration: how to subtract
ForceConv = 'REF'               # FIXME {'REF', 'SCI'}
                                # 'REF': convolve the reference image, DIFF = SCI - Convolved_REF.
                                # 'SCI': convolve the science image, DIFF = Convolved_SCI - REF.

GKerHW = 4                      # FIXME given kernel half-width
KerPolyOrder = 2                # FIXME {0, 1, 2, 3}, Polynomial degree of kernel spatial variation
BGPolyOrder = 2                 # FIXME {0, 1, 2, 3}, Polynomial degree of differential background spatial variation
ConstPhotRatio = True           # FIXME Constant photometric ratio between images?
 
FITS_REF = CDIR + '/input_data/ztf_001735_zg_c01_q2_refimg.resampled.mini.fits'
FITS_SCI = CDIR + '/input_data/ztf_20180705481609_001735_zg_c01_o_q2_sciimg.mini.fits'
FITS_mREF = FITS_REF[:-5] + '.masked.fits'
FITS_mSCI = FITS_SCI[:-5] + '.masked.fits'
FITS_DIFF = CDIR + '/output_data/sfft_diff.fits'

if BACKEND_4SUBTRACT in ['Cupy', 'Pycuda']:
    print('\n---------------------------------- GPU warming-up --------------------------------------')
    print('MeLOn CheckPoint: GPU warming-up [START]!')
    print('MeLOn Note: GPU warming-up is not an ingredient of SFFT.')
    print('MeLOn Note: Instead, it is used to finish some GPU initializations before SFFT is called.')
    print('MeLOn Note: When process a queue of tasks, GPU "warming-up" only exists for the first task.')

    class HidePrint:
        def __enter__(self):
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout.close()
            sys.stdout = self._original_stdout

    with HidePrint():
        _tmp = fits.getheader(FITS_REF, ext=0)
        NX, NY = int(_tmp['NAXIS1']), int(_tmp['NAXIS2'])
        _PixA_REF_GPU = cp.random.uniform(0.9, 1.1, (NX, NY), dtype=cp.float64)
        _PixA_SCI_GPU = cp.random.uniform(9.9, 10.1, (NX, NY), dtype=cp.float64)
        _PixA_mREF_GPU = _PixA_REF_GPU.copy()
        _PixA_mSCI_GPU = _PixA_SCI_GPU.copy()

        PureCupy_Customized_Packet.PCCP(
            PixA_REF_GPU=_PixA_REF_GPU, PixA_SCI_GPU=_PixA_SCI_GPU, PixA_mREF_GPU=_PixA_mREF_GPU, PixA_mSCI_GPU=_PixA_mSCI_GPU, 
            ForceConv=ForceConv, GKerHW=GKerHW, KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, 
            CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT
        )
    
    print('MeLOn CheckPoint: GPU warming-up [END]!')
    print('---------------------------------- GPU warming-up --------------------------------------\n')

# * Read input images
PixA_REF = fits.getdata(FITS_REF, ext=0).T
PixA_SCI = fits.getdata(FITS_SCI, ext=0).T
PixA_mREF = fits.getdata(FITS_mREF, ext=0).T
PixA_mSCI = fits.getdata(FITS_mSCI, ext=0).T

if not PixA_REF.flags['C_CONTIGUOUS']:
    PixA_REF = np.ascontiguousarray(PixA_REF, np.float64)
else: PixA_REF = PixA_REF.astype(np.float64)
PixA_REF_GPU = cp.array(PixA_REF)

if not PixA_SCI.flags['C_CONTIGUOUS']:
    PixA_SCI = np.ascontiguousarray(PixA_SCI, np.float64)
else: PixA_SCI = PixA_SCI.astype(np.float64)
PixA_SCI_GPU = cp.array(PixA_SCI)

if not PixA_mREF.flags['C_CONTIGUOUS']:
    PixA_mREF = np.ascontiguousarray(PixA_mREF, np.float64)
else: PixA_mREF = PixA_mREF.astype(np.float64)
PixA_mREF_GPU = cp.array(PixA_mREF)

if not PixA_mSCI.flags['C_CONTIGUOUS']:
    PixA_mSCI = np.ascontiguousarray(PixA_mSCI, np.float64)
else: PixA_mSCI = PixA_mSCI.astype(np.float64)
PixA_mSCI_GPU = cp.array(PixA_mSCI)

# run sfft subtraction
print('\n********************************** SFFT subtraction **************************************')
print('MeLOn CheckPoint: SFFT subtraction [START]!')

# NOTE: see complete descriptions of the parameters via help(sfft.PureCupy_Customized_Packet)
Solution_GPU, PixA_DIFF_GPU = PureCupy_Customized_Packet.PCCP(
    PixA_REF_GPU=PixA_REF_GPU, PixA_SCI_GPU=PixA_SCI_GPU, PixA_mREF_GPU=PixA_mREF_GPU, PixA_mSCI_GPU=PixA_mSCI_GPU, 
    ForceConv=ForceConv, GKerHW=GKerHW, KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, 
    CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT
)

print('MeLOn CheckPoint: SFFT subtraction [END]!')
print('********************************** SFFT subtraction **************************************\n')
print('MeLOn CheckPoint: TEST FOR CUSTOMIZED-SFFT SUBTRACTION DONE!\n')

PixA_DIFF = cp.asnumpy(PixA_DIFF_GPU)
# * Save difference image
if FITS_DIFF is not None:
    _hdl = fits.open(FITS_SCI)
    _hdl[0].data[:, :] = PixA_DIFF.T
    _hdl[0].header['NAME_REF'] = (pa.basename(FITS_REF), 'MeLOn: SFFT')
    _hdl[0].header['NAME_SCI'] = (pa.basename(FITS_SCI), 'MeLOn: SFFT')
    _hdl[0].header['KERORDER'] = (KerPolyOrder, 'MeLOn: SFFT')
    _hdl[0].header['BGORDER'] = (BGPolyOrder, 'MeLOn: SFFT')
    _hdl[0].header['CPHOTR'] = (str(ConstPhotRatio), 'MeLOn: SFFT')
    _hdl[0].header['KERHW'] = (GKerHW, 'MeLOn: SFFT')
    _hdl[0].header['CONVD'] = (ForceConv, 'MeLOn: SFFT')
    _hdl.writeto(FITS_DIFF, overwrite=True)
    _hdl.close()
