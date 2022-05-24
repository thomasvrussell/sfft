import os
import os.path as pa
from sfft.MultiEasyCrowdedPacket import MultiEasy_CrowdedPacket
CDIR = pa.dirname(pa.abspath(__file__))

"""
* Added test in Version 1.1+

"""

NUM_TASK = 16                      # FIXME Number of tasks
NUM_THREADS_4PREPROC = 8           # FIXME Python threads for preprocesing
NUM_THREADS_4SUBTRACT = 1          # FIXME Python threads for sfft subtraction (= Number of GPU devices)
CUDA_DEVICES_4SUBTRACT = ['0']     # FIXME List of GPU devices (see nvidia-smi to find the GPU indices)

TIMEOUT_4PREPRO_EACHTASK = 120     # FIXME timeout (sec) of preprocessing for each task
TIMEOUT_4SUBTRACT_EACHTASK = 120   # FIXME timeout (sec) of sfft subtraction for each task

ForceConv = 'REF'                  # FIXME {None, 'REF', 'SCI'}, None mean AUTO mode that avoids deconvolution
GAIN_KEY = 'GAIN'                  # NOTE Keyword of Gain in FITS header
SATUR_KEY = 'ESATUR'               # NOTE Keyword of Saturation in FITS header

# * Make multiple tasks (just copy the data in the crowded flavor test multiple times)
origin_dir = pa.join(pa.dirname(CDIR), 'subtract_test_crowded_flavor', 'input_data')
taskmdir = pa.join(CDIR, 'taskdirs_crowded_flavor')

FNAME_REF = 'ztf_001735_zg_c01_q2_refimg.resampled.mini.fits'
FNAME_SCI = 'ztf_20180705481609_001735_zg_c01_o_q2_sciimg.mini.fits'
for taskidx in range(NUM_TASK):
    taskdir = taskmdir + '/task-%d' %taskidx
    try: os.makedirs(taskdir)
    except OSError:
        if pa.exists(taskdir): pass
    os.system('cp %s/%s %s/%s %s/' \
        %(origin_dir, FNAME_REF, origin_dir, FNAME_SCI, taskdir))

# * Trigger sfft subtraction in multiprocessing mode (currently only Cupy backend is supported)
FITS_REF_Queue = [taskmdir + '/task-%d/%s' %(taskidx, FNAME_REF) for taskidx in range(NUM_TASK)]
FITS_SCI_Queue = [taskmdir + '/task-%d/%s' %(taskidx, FNAME_SCI) for taskidx in range(NUM_TASK)]
FITS_DIFF_Queue = [taskmdir + '/task-%d/my_sfft_diff.fits' %taskidx for taskidx in range(NUM_TASK)]
FITS_Solution_Queue = [taskmdir + '/task-%d/my_sfft_solution' %taskidx for taskidx in range(NUM_TASK)]
ForceConv_Queue = [ForceConv] * NUM_TASK

# *************************** IMPORTANT NOTICE *************************** #
#  I strongly recommend users to read the descriptions of the parameters 
#  via help(sfft.MultiEasy_CrowdedPacket).
# *************************** IMPORTANT NOTICE *************************** #

res = MultiEasy_CrowdedPacket(FITS_REF_Queue, FITS_SCI_Queue, FITS_DIFF_Queue=FITS_DIFF_Queue, \
    FITS_Solution_Queue=FITS_Solution_Queue, ForceConv_Queue=ForceConv_Queue, GKerHW_Queue=[], KerHWRatio=2.0, \
    KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, MaskSatContam=True, BACKSIZE_SUPER=128, \
    GAIN_KEY='GAIN', SATUR_KEY='SATURATE', DETECT_THRESH=5.0, StarExt_iter=2, PriorBanMask_Queue=[]).\
    MESP_Cupy(NUM_THREADS_4PREPROC=NUM_THREADS_4PREPROC, NUM_THREADS_4SUBTRACT=NUM_THREADS_4SUBTRACT, \
    CUDA_DEVICES_4SUBTRACT=CUDA_DEVICES_4SUBTRACT, TIMEOUT_4PREPRO_EACHTASK=TIMEOUT_4PREPRO_EACHTASK, \
    TIMEOUT_4SUBTRACT_EACHTASK=TIMEOUT_4SUBTRACT_EACHTASK)

"""
# * it will produce a status-bar dictionary (indexed by task-index)
#   status 0: initial state (NO Processing Finished Yet)
#   status 1: Preprocessing Successed || status -1: Preprocessing Failed
#   status 2: Subtraction Successed || status -2: Subtraction Failed
#   NOTE: we would not trigger subtraction when the Preprocessing already fails.
"""

DICT_STATUS_BAR = res[0]
print(DICT_STATUS_BAR)
