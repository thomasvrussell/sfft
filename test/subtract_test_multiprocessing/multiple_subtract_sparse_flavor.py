import os
import time
import numpy as np
import os.path as pa
from sfft.MultiEasySparsePacket import MultiEasy_SparsePacket
CDIR = pa.dirname(pa.abspath(__file__))
# sfft version: 1.3.2+

# * AN IMPORTANT WARNING (Oct 25, 2022)
#   Please do not put the function MultiEasy_SparsePacket in a Python loop,
#   since the uncleaned GPU memory in each run can accumulate and exceed your GPU limit.
#   Instead, if one want to process a sequence of exposures using this function
#   (I assume that each exposure has multiple ccd images, e.g., DECam, HSC, ...),
#   I suggest users to run a separate .py for each exposure (like the example here) 
#   and trigger them one by one in a shell script, that will avoid the undesired GPU memory issue.

# configuration: multiprocessing and computing resourse
NUM_THREADS_4PREPROC = 16          # FIXME Python threads for preprocesing
NUM_THREADS_4SUBTRACT = 1          # FIXME Python threads for sfft subtraction (= Number of GPU devices)
CUDA_DEVICES_4SUBTRACT = ['0']     # FIXME List of GPU devices (see nvidia-smi to find the GPU indices)

TIMEOUT_4PREPROC_EACHTASK = 300    # FIXME timeout (sec) of preprocessing for each task
TIMEOUT_4SUBTRACT_EACHTASK = 300   # FIXME timeout (sec) of sfft subtraction for each task

# configuration: how to subtract
ForceConv = 'REF'                  # FIXME {'AUTO', 'REF', 'SCI'}
                                   # 'AUTO': convolve the image with smaller FWHM to avoid deconvolution.
                                   # 'REF': convolve the reference image, DIFF = SCI - Convolved_REF.
                                   # 'SCI': convolve the science image, DIFF = Convolved_SCI - REF.

KerHWRatio = 2.0                   # FIXME Ratio of kernel half-width to FWHM (typically, 1.5-2.5).
KerPolyOrder = 2                   # FIXME {0, 1, 2, 3}, Polynomial degree of kernel spatial variation
BGPolyOrder = 0                    # FIXME {0, 1, 2, 3}, Polynomial degree of differential background spatial variation.
                                   #       it is trivial here, as sparse-flavor-sfft requires sky subtraction in advance.
ConstPhotRatio = True              # FIXME Constant photometric ratio between images?

# configuration: required info in FITS header
GAIN_KEY = 'GAIN'                  # NOTE Keyword of Gain in FITS header
SATUR_KEY = 'ESATUR'               # NOTE Keyword of Saturation in FITS header
                                   # Remarks: one may think 'SATURATE' is a more common keyword name for saturation level.
                                   #          However, note that Sparse-Flavor SFFT requires sky-subtracted images as inputs, 
                                   #          we need to use the 'effective' saturation level after the sky-subtraction. 
                                   #          e.g., set ESATURA = SATURATE - (SKY + 10*SKYSIG)

print('\n---------------------------------- CREATE MULTI-TASKS --------------------------------------')
# * create multiple tasks (just copy the data in subtract_test_sparse_flavor multiple times)
origin_dir = pa.join(pa.dirname(CDIR), 'subtract_test_sparse_flavor', 'input_data')
taskmdir = pa.join(CDIR, 'taskdirs_sparse_flavor')

NUM_TASK = 16   
FNAME_REF = 'c4d_180806_052738_ooi_i_v1.S20.skysub.resampled.fits'
FNAME_SCI = 'c4d_180802_062754_ooi_i_v1.S20.skysub.fits'
for taskidx in range(NUM_TASK):
    taskdir = taskmdir + '/task-%d' %taskidx
    try: os.makedirs(taskdir)
    except OSError:
        if pa.exists(taskdir): pass
    os.system('cp %s/%s %s/%s %s/' \
        %(origin_dir, FNAME_REF, origin_dir, FNAME_SCI, taskdir))
print('---------------------------------- CREATE MULTI-TASKS --------------------------------------\n')

# * run sfft subtraction in multiprocessing mode (currently only Cupy backend is supported)
FITS_REF_Queue = [taskmdir + '/task-%d/%s' %(taskidx, FNAME_REF) for taskidx in range(NUM_TASK)]
FITS_SCI_Queue = [taskmdir + '/task-%d/%s' %(taskidx, FNAME_SCI) for taskidx in range(NUM_TASK)]
FITS_DIFF_Queue = [taskmdir + '/task-%d/sfft_diff.fits' %taskidx for taskidx in range(NUM_TASK)]
ForceConv_Queue = [ForceConv] * NUM_TASK

T0 = time.time()
_MESP = MultiEasy_SparsePacket(FITS_REF_Queue=FITS_REF_Queue, FITS_SCI_Queue=FITS_SCI_Queue, \
    FITS_DIFF_Queue=FITS_DIFF_Queue, FITS_Solution_Queue=[], ForceConv_Queue=ForceConv_Queue, \
    GKerHW_Queue=[], KerHWRatio=KerHWRatio, KerHWLimit=(2, 20), KerPolyOrder=KerPolyOrder, \
    BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, MaskSatContam=False, GAIN_KEY=GAIN_KEY, \
    SATUR_KEY=SATUR_KEY, BACK_TYPE='MANUAL', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, \
    DETECT_THRESH=2.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
    ONLY_FLAGS=[0], BoundarySIZE=30, XY_PriorSelect_Queue=[], Hough_FRLowerLimit=0.1, Hough_peak_clip=0.7, \
    BeltHW=0.2, PS_ELLIPThresh=0.3, MatchTol=None, MatchTolFactor=3.0, COARSE_VAR_REJECTION=True, \
    CVREJ_MAGD_THRESH=0.12, ELABO_VAR_REJECTION=True, EVREJ_RATIO_THREH=5.0, EVREJ_SAFE_MAGDEV=0.04, \
    XY_PriorBan_Queue=[], PostAnomalyCheck=False, PAC_RATIO_THRESH=5.0)

DICT_STATUS_BAR = _MESP.MESP_Cupy(NUM_THREADS_4PREPROC=NUM_THREADS_4PREPROC, NUM_THREADS_4SUBTRACT=NUM_THREADS_4SUBTRACT, \
    CUDA_DEVICES_4SUBTRACT=CUDA_DEVICES_4SUBTRACT, TIMEOUT_4PREPROC_EACHTASK=TIMEOUT_4PREPROC_EACHTASK, \
    TIMEOUT_4SUBTRACT_EACHTASK=TIMEOUT_4SUBTRACT_EACHTASK)[0]

NSUC = np.sum(np.array([DICT_STATUS_BAR[tn] for tn in DICT_STATUS_BAR]) == 2)
_message = 'SUCCESSFUL TASKS [%d / %d]' %(NSUC, NUM_TASK)
print('\nMeLOn CheckPoint: %s' %_message)

_message = 'SFFT PROCESS [%d] TASKS (2K * 4K DECam IMAGES) WITH ' %NUM_TASK
_message += 'NUM_THREADS_4PREPROC [%d] & NUM_THREADS_4SUBTRACT [%d] ' %(NUM_THREADS_4PREPROC, NUM_THREADS_4SUBTRACT)
_message += '---- TOTAL COMPUTING TIME [%.2fs]' %(time.time() - T0)
print('\nMeLOn CheckPoint: %s\n' %_message)

"""
# **** **** **** **** PARAMETER CHANGE LOG **** **** **** **** #

THE FUNCTION IS CREATED AT v1.1

PARAMETERS NAME CHANGED:
  -MAGD_THRESH (v1.1, v1.2) >>>> -CVREJ_MAGD_THRESH (v1.3+)
  -CheckPostAnomaly (v1.1, v1.2) >>>> -PostAnomalyCheck (v1.3+)
  -PARATIO_THRESH (v1.1, v1.2) >>>> -PAC_RATIO_THRESH (v1.3+)

PARAMETERS DEFUALT-VALUE CHANGED:
  -BGPolyOrder=2 (v1.1, v1.2) >>>> -BGPolyOrder=0 (v1.3+)

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
  >>>> -XY_PriorSelect_Queue (v1.2+)

  (4) allows coarse variable rejection can be T/F
  >>>> -COARSE_VAR_REJECTION (v1.3+)

  (5) allows for elaborate varaible rejection
  >>>> ELABO_VAR_REJECTION (v1.3+)
  >>>> EVREJ_RATIO_THREH (v1.3+)
  >>>> EVREJ_SAFE_MAGDEV (v1.3+)

  (6) allows for more detailed tunning on hough transformation
  >>>> Hough_peak_clip (v1.3.1+)
  
"""
