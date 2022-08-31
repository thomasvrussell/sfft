import time
import math
import warnings
import threading
import numpy as np
import os.path as pa
from astropy.io import fits
from sfft.AutoCrowdedPrep import Auto_CrowdedPrep
from sfft.utils.meta.TimeoutKit import TimeoutAfter
# version: Aug 31, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.3"

class MultiEasy_CrowdedPacket:
    def __init__(self, FITS_REF_Queue, FITS_SCI_Queue, FITS_DIFF_Queue=[], FITS_Solution_Queue=[], \
        ForceConv_Queue=[], GKerHW_Queue=[], KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, \
        ConstPhotRatio=True, MaskSatContam=False, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        BACK_TYPE='AUTO', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=5.0, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
        ONLY_FLAGS=None, BoundarySIZE=0.0, BACK_SIZE_SUPER=128, StarExt_iter=2, PriorBanMask_Queue=[]):

        """
        # NOTE: This function is to Perform Crowded-Flavor SFFT for multiple tasks:
        #       @ it makes sense when users have multiple CPU threads and GPU(s) available.
        #       @ WARNING: the module currently only support Cupy backend!

        * Parameters for Sparse-Flavor SFFT [multiple tasks]
        
        # ----------------------------- Computing Enviornment [Cupy backend ONLY] --------------------------------- #

        -NUM_THREADS_4PREPROC [8]           # the number of Python threads for preprocessing.
                                            # WARNING: Empirically, the current code is only optimized for 
                                            #          NUM_TASK / NUM_THREADS_4PREPROC in [1.0 - 4.0].
                                            #          The computing effeciency goes down sharply as the ratio increases.
                                            #          Too large number of tasks may even cause memory overflow.

        -NUM_THREADS_4SUBTRACT [1]          # the number of Python threads for sfft subtraction.
                                            # says, if you have four GPU cards, then set it to 4.
                                            # NOTE: please do not confused the Python threads 
                                            #       with the CUDA threads (block, threads ...).

        -CUDA_DEVICES_4SUBTRACT [['0']]     # it specifies certain GPU devices (indices) to conduct the subtractions.
                                            # the GPU devices are usually numbered 0 to N-1 (you may use command nvidia-smi to check).

        -TIMEOUT_4PREPROC_EACHTASK [300]    # timeout of each task during preprocessing.
                                            
        -TIMEOUT_4SUBTRACT_EACHTASK [300]   # timeout of each task during sfft subtraction.

        # ----------------------------- Preprocessing with Saturation Rejection for Image-Masking --------------------------------- #

        # > Configurations for SExtractor

        -GAIN_KEY ['GAIN']                  # SExtractor Parameter GAIN_KEY
                                            # i.e., keyword of GAIN in FITS header (of reference & science)

        -SATUR_KEY ['SATURATE']             # SExtractor Parameter SATUR_KEY
                                            # i.e., keyword of effective saturation in FITS header (of reference & science)
                                            # Remarks: note that Crowded-Flavor SFFT does not require sky-subtracted images as inputs,
                                            #          so the default keyword is the common name for saturation level.
        
        -BACK_TYPE ['AUTO']                 # SExtractor Parameter BACK_TYPE = [AUTO or MANUAL].
         
        -BACK_VALUE [0.0]                   # SExtractor Parameter BACK_VALUE (only work for BACK_TYPE='MANUAL')

        -BACK_SIZE [64]                     # SExtractor Parameter BACK_SIZE

        -BACK_FILTERSIZE [3]                # SExtractor Parameter BACK_FILTERSIZE

        -DETECT_THRESH [5.0]                # SExtractor Parameter DETECT_THRESH

        -DETECT_MINAREA [5]                 # SExtractor Parameter DETECT_MINAREA
        
        -DETECT_MAXAREA [0]                 # SExtractor Parameter DETECT_MAXAREA

        -DEBLEND_MINCONT [0.005]            # SExtractor Parameter DEBLEND_MINCONT (typically, 0.001 - 0.005)

        -BACKPHOTO_TYPE ['LOCAL']           # SExtractor Parameter BACKPHOTO_TYPE

        -ONLY_FLAGS [None]                  # Restrict SExtractor Output Photometry Catalog by Source FLAGS
                                            # Common FLAGS (Here None means no restrictions on FLAGS):
                                            # 1: aperture photometry is likely to be biased by neighboring sources 
                                            #    or by more than 10% of bad pixels in any aperture
                                            # 2: the object has been deblended
                                            # 4: at least one object pixel is saturated

        -BoundarySIZE [30]                  # Restrict SExtractor Output Photometry Catalog by Dropping Sources at Boundary 
                                            # NOTE: This would help to avoid selecting sources too close to image boundary. 

        # > Other Configurations

        -BACK_SIZE_SUPER [128]              # Also a SExtractor configuration BACK_SIZE, however, the background is 
                                            # used to fill the masked regions (saturation contaminated pixels).
                                            # NOTE: '_SUPER' means we would like to use a realtively large BACK_SIZE to
                                            #       make a super-smooth (structure-less) background. So the default
                                            #       -BACK_SIZE_SUPER > -BACK_SIZE

        -StarExt_iter [2]                   # make a further dilation for the masked region initially determined by SExtractor SEGMENTATION.
                                            # -StarExt_iter means the iteration times of the dilation process. 
        
        -PriorBanMask_Queue [[]]]           # A queue of -PriorBanMask in sfft.Easy_CrowdedPacket.
                                            # Recall -PriorBanMask:
                                            # a Numpy boolean array, with shape consistent with reference (science).
                                            # one can deliver a customized mask that covers known 
                                            # variables/transients/bad-pixels by this argument.
                                            # SFFT will then merge (Union operation) the user-defined mask and 
                                            # the SExtractor-determined saturation mask as the final mask.

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv_Queue ['AUTO']           # A queue of -ForceConv in sfft.Easy_CrowdedPacket.
                                            # it determines which image will be convolved, can be 'REF', 'SCI' and 'AUTO'.
                                            # -ForceConv = 'AUTO' means SFFT will determine the convolution direction according to 
                                            # FWHM_SCI and FWHM_REF: the image with better seeing will be convolved to avoid deconvolution.

        -GKerHW_Queue [None]                # A queue of -GKerHW in sfft.Easy_CrowdedPacket.
                                            # The given kernel half-width, None means the kernel size will be 
                                            # automatically determined by -KerHWRatio (to be seeing-related). 

        -KerHWRatio [2.0]                   # The ratio between FWHM and the kernel half-width
                                            # KerHW = int(KerHWRatio * Max(FWHM_REF, FWHM_SCI))

        -KerHWLimit [(2, 20)]               # The lower & upper bounds for kernel half-width 
                                            # KerHW is updated as np.clip(KerHW, KerHWLimit[0], KerHWLimit[1]) 
                                            # Remarks: this is useful for a survey since it can constrain the peak GPU memory usage.

        -KerPolyOrder [2]                   # Polynomial degree of kernel spatial variation, can be [0,1,2,3].

        -BGPolyOrder [2]                    # Polynomial degree of differential-background spatial variation, can be [0,1,2,3].
                                            # It is non-trivial for Crowded-Flavor SFFT, as the input images are usually not sky subtracted.
        
        -ConstPhotRatio [True]              # Constant photometric ratio between images? can be True or False
                                            # ConstPhotRatio = True: the sum of convolution kernel is restricted 
                                            #   to be a constant across the field. 
                                            # ConstPhotRatio = False: the flux scaling between images is modeled
                                            #   by a polynomial with degree -KerPolyOrder.

        -MaskSatContam [False]              # Mask saturation-contaminated regions on difference image ? can be True or False
                                            # NOTE the pixels enclosed in the regions are replaced by NaN.

        # ----------------------------- Input & Output --------------------------------- #

        -FITS_REF_Queue []                 # A queue of -FITS_REF in sfft.Easy_CrowdedPacket.
                                           # File path of input reference image.

        -FITS_SCI_Queue []                 # A queue of -FITS_SCI in sfft.Easy_CrowdedPacket.
                                           # File path of input science image.

        -FITS_DIFF_Queue [None]            # A queue of -FITS_DIFF in sfft.Easy_CrowdedPacket.
                                           # File path of output difference image.

        -FITS_Solution_Queue [None]        # A queue of -FITS_Solution in sfft.Easy_CrowdedPacket.
                                           # File path of the solution of the linear system.
                                           # it is an array of (..., a_ijab, ... b_pq, ...).

        # Important Notice:
        #
        # a): if reference is convolved in SFFT, then DIFF = SCI - Convolved_REF.
        #     [difference image is expected to have PSF & flux zero-point consistent with science image]
        #     e.g., -ForceConv='REF' or -ForceConv='AUTO' when reference has better seeing.
        #
        # b): if science is convolved in SFFT, then DIFF = Convolved_SCI - REF
        #     [difference image is expected to have PSF & flux zero-point consistent with reference image]
        #     e.g., -ForceConv='SCI' or -ForceConv='AUTO' when science has better seeing.
        #
        # Remarks: this convention is to guarantee that transients emerge on science image always 
        #          show a positive signal on difference images.

        """

        self.FITS_REF_Queue = FITS_REF_Queue
        self.FITS_SCI_Queue = FITS_SCI_Queue

        self.KerHWRatio = KerHWRatio
        self.KerHWLimit = KerHWLimit
        self.KerPolyOrder = KerPolyOrder
        self.BGPolyOrder = BGPolyOrder
        self.ConstPhotRatio = ConstPhotRatio
        self.MaskSatContam = MaskSatContam
        
        self.GAIN_KEY = GAIN_KEY
        self.SATUR_KEY = SATUR_KEY
        self.BACK_TYPE = BACK_TYPE
        self.BACK_VALUE = BACK_VALUE
        self.BACK_SIZE = BACK_SIZE
        self.BACK_FILTERSIZE = BACK_FILTERSIZE
        
        self.DETECT_THRESH = DETECT_THRESH
        self.DETECT_MINAREA = DETECT_MINAREA
        self.DETECT_MAXAREA = DETECT_MAXAREA
        self.DEBLEND_MINCONT = DEBLEND_MINCONT
        self.BACKPHOTO_TYPE = BACKPHOTO_TYPE
        self.ONLY_FLAGS = ONLY_FLAGS
        self.BoundarySIZE = BoundarySIZE
        
        self.BACK_SIZE_SUPER = BACK_SIZE_SUPER
        self.StarExt_iter = StarExt_iter
        self.NUM_TASK = len(FITS_SCI_Queue)

        if FITS_DIFF_Queue == []:
            _warn_message = 'Argument FITS_DIFF_Queue is EMPTY then Use default [...None...] instead!'
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
            self.FITS_DIFF_Queue = [None] * self.NUM_TASK
        else: self.FITS_DIFF_Queue = FITS_DIFF_Queue
        
        if FITS_Solution_Queue == []:
            _warn_message = 'Argument FITS_Solution_Queue is EMPTY then Use default [...None...] instead!'
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
            self.FITS_Solution_Queue = [None] * self.NUM_TASK
        else: self.FITS_Solution_Queue = FITS_Solution_Queue
        
        if ForceConv_Queue == []:
            _warn_message = 'Argument ForceConv_Queue is EMPTY then Use default [...AUTO...] instead!'
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
            self.ForceConv_Queue = ['AUTO'] * self.NUM_TASK
        else: self.ForceConv_Queue = ForceConv_Queue

        if GKerHW_Queue == []:
            _warn_message = 'Argument GKerHW_Queue is EMPTY then Use default [...None...] instead!'
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
            self.GKerHW_Queue = [None] * self.NUM_TASK
        else: self.GKerHW_Queue = GKerHW_Queue

        if PriorBanMask_Queue == []:
            _warn_message = 'Argument PriorBanMask_Queue is EMPTY then Use default [...None...] instead!'
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
            self.PriorBanMask_Queue = [None] * self.NUM_TASK
        else: self.PriorBanMask_Queue = PriorBanMask_Queue
    
    def MESP_Cupy(self, NUM_THREADS_4PREPROC=8, NUM_THREADS_4SUBTRACT=1, CUDA_DEVICES_4SUBTRACT=['0'], \
        TIMEOUT_4PREPROC_EACHTASK=300, TIMEOUT_4SUBTRACT_EACHTASK=300):

        # * define task-oriented dictionaries of status-bar and products
        #   status 0: Initial State
        #   status 1: Preprocessing SUCCESS || status -1: Preprocessing FAIL
        #   status 2: Subtraction SUCCESS || status -2: Subtraction FAIL
        #   status 32: Preprocessing is IN-PROGRESS ...
        #   status 64: Subtraction is IN-PROGRESS ...
        #   NOTE: Subtraction would not be triggered when Preprocessing fails
        
        _RATIO = self.NUM_TASK / NUM_THREADS_4PREPROC
        if _RATIO > 4.0:
            _warn_message = 'THIS FUNCTION IS NOT OPTIMIZED FOR RATIO OF '
            _warn_message += 'NUM_TASK / NUM_THREADS_4PREPROC BEING TO HIGH [%.1f > 4.0]!' %_RATIO
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)

        BACKEND_4SUBTRACT = 'Cupy'
        taskidx_lst = np.arange(self.NUM_TASK)
        assert NUM_THREADS_4SUBTRACT == len(CUDA_DEVICES_4SUBTRACT)
        
        DICT_STATUS_BAR = {'task-[%d]' %taskidx: 0 for taskidx in taskidx_lst}
        DICT_PRODUCTS = {'task-[%d]' %taskidx: {} for taskidx in taskidx_lst}
        lock = threading.RLock()

        # * define the function for preprocessing (see sfft.EasyCrowdedPacket)
        def FUNC_4PREPROC(INDEX_THREAD_4PREPROC):

            THREAD_ALIVE_4PREPROC = True
            while THREAD_ALIVE_4PREPROC:
                with lock:
                    STATUS_SNAPSHOT = np.array([DICT_STATUS_BAR['task-[%d]' %taskidx] for taskidx in taskidx_lst])
                    OPEN_MASK_4PREPROC = STATUS_SNAPSHOT == 0
                    if np.sum(OPEN_MASK_4PREPROC) > 0:
                        taskidx_acquired = taskidx_lst[OPEN_MASK_4PREPROC][0]
                        DICT_STATUS_BAR['task-[%d]' %taskidx_acquired] = 32
                        _message = 'Preprocessing thread-[%d] ACQUIRED task-[%d]!' %(INDEX_THREAD_4PREPROC, taskidx_acquired)
                        print('\nMeLOn CheckPoint: %s' %_message)
                    else: 
                        THREAD_ALIVE_4PREPROC = False
                        taskidx_acquired = None
                        _message = 'TERMINATE Preprocessing thread-[%d] AS NO TASK WILL BE ALLOCATED!' %INDEX_THREAD_4PREPROC
                        print('\nMeLOn CheckPoint: %s' %_message)

                if taskidx_acquired is not None:
                    FITS_REF = self.FITS_REF_Queue[taskidx_acquired]
                    FITS_SCI = self.FITS_SCI_Queue[taskidx_acquired]
                    PriorBanMask = self.PriorBanMask_Queue[taskidx_acquired]
                    try:
                        with TimeoutAfter(timeout=TIMEOUT_4PREPROC_EACHTASK, exception=TimeoutError):
                            _ACP = Auto_CrowdedPrep(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, GAIN_KEY=self.GAIN_KEY, \
                                SATUR_KEY=self.SATUR_KEY, BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, \
                                BACK_SIZE=self.BACK_SIZE, BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, \
                                DETECT_MINAREA=self.DETECT_MINAREA, DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, \
                                BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, ONLY_FLAGS=self.ONLY_FLAGS, BoundarySIZE=self.BoundarySIZE)
                            SFFTPrepDict = _ACP.AutoMask(BACK_SIZE_SUPER=self.BACK_SIZE_SUPER, StarExt_iter=self.StarExt_iter, \
                                PriorBanMask=PriorBanMask)
                        
                        NEW_STATUS = 1
                        _message = 'Successful Preprocessing for task-[%d] ' %taskidx_acquired
                        _message += 'in Preprocessing thread-[%d]!' %INDEX_THREAD_4PREPROC
                        print('\nMeLOn CheckPoint: %s' %_message)
                    except:
                        NEW_STATUS = -1
                        _error_message = 'UnSuccessful Preprocessing for task-[%d] ' %taskidx_acquired
                        _error_message += 'in Preprocessing thread-[%d]!' %INDEX_THREAD_4PREPROC
                        print('\nMeLOn ERROR: %s' %_error_message)
                        SFFTPrepDict = None
                    
                    with lock:
                        # NOTE: IN FACT, THERE IS NO RISK HERE EVEN WITHOUT LOCK.
                        DICT_STATUS_BAR['task-[%d]' %taskidx_acquired] = NEW_STATUS
                        DICT_PRODUCTS['task-[%d]' %taskidx_acquired]['SFFTPrepDict'] = SFFTPrepDict

            return None
        
        # * define the function for sfft subtraction (see sfft.EasyCrowdedPacket)
        def FUNC_4SUBTRCT(INDEX_THREAD_4SUBTRACT):
            
            CUDA_DEVICE_4SUBTRACT = CUDA_DEVICES_4SUBTRACT[INDEX_THREAD_4SUBTRACT]
            THREAD_ALIVE_4SUBTRCT = True
            while THREAD_ALIVE_4SUBTRCT:
                SHORT_PAUSE = False
                with lock:
                    STATUS_SNAPSHOT = np.array([DICT_STATUS_BAR['task-[%d]' %taskidx] for taskidx in taskidx_lst])
                    OPEN_MASK_4SUBTRACT = STATUS_SNAPSHOT == 1
                    WAIT_MASK_4SUBTRACT = np.in1d(STATUS_SNAPSHOT, [0, 32])
                    if np.sum(OPEN_MASK_4SUBTRACT) > 0:
                        taskidx_acquired = taskidx_lst[OPEN_MASK_4SUBTRACT][0]
                        DICT_STATUS_BAR['task-[%d]' %taskidx_acquired] = 64
                        _message = 'Subtraction thread-[%d] ACQUIRED task-[%d]!' %(INDEX_THREAD_4SUBTRACT, taskidx_acquired)
                        print('\nMeLOn CheckPoint: %s' %_message)
                    elif np.sum(WAIT_MASK_4SUBTRACT) > 0:
                        SHORT_PAUSE = True
                        taskidx_acquired = None
                    else:
                        THREAD_ALIVE_4SUBTRCT = False
                        taskidx_acquired = None
                        _message = 'TERMINATE Subtraction thread-[%d] AS NO TASK WILL BE ALLOCATED!' %INDEX_THREAD_4SUBTRACT
                        print('\nMeLOn CheckPoint: %s' %_message)

                if SHORT_PAUSE:
                    # run a short sleep, otherwise this thread may always hold the lock by the while loop
                    time.sleep(0.01)
                
                if taskidx_acquired is not None:
                    FITS_REF = self.FITS_REF_Queue[taskidx_acquired]
                    FITS_SCI = self.FITS_SCI_Queue[taskidx_acquired]
                    FITS_DIFF = self.FITS_DIFF_Queue[taskidx_acquired]
                    FITS_Solution = self.FITS_Solution_Queue[taskidx_acquired]
                    ForceConv = self.ForceConv_Queue[taskidx_acquired]
                    GKerHW = self.GKerHW_Queue[taskidx_acquired]

                    with lock:
                        # NOTE: IN FACT, THERE IS NO RISK HERE EVEN WITHOUT LOCK.
                        SFFTPrepDict = DICT_PRODUCTS['task-[%d]' %taskidx_acquired]['SFFTPrepDict']

                    try:
                        with TimeoutAfter(timeout=TIMEOUT_4SUBTRACT_EACHTASK, exception=TimeoutError):
                            # * Determine ConvdSide & KerHW
                            FWHM_REF = SFFTPrepDict['FWHM_REF']
                            FWHM_SCI = SFFTPrepDict['FWHM_SCI']

                            assert ForceConv in ['AUTO', 'REF', 'SCI']
                            if ForceConv == 'AUTO':
                                if FWHM_SCI >= FWHM_REF: ConvdSide = 'REF'
                                else: ConvdSide = 'SCI'
                            else: ConvdSide = ForceConv

                            if GKerHW is None:
                                FWHM_La = np.max([FWHM_REF, FWHM_SCI])
                                KerHW = int(np.clip(self.KerHWRatio * FWHM_La, \
                                                    self.KerHWLimit[0], self.KerHWLimit[1]))
                            else: KerHW = GKerHW

                            # * Compile Functions in SFFT Subtraction
                            from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure

                            PixA_REF = SFFTPrepDict['PixA_REF']
                            PixA_SCI = SFFTPrepDict['PixA_SCI']
                            
                            Tcomp_start = time.time()
                            SFFTConfig = SingleSFFTConfigure.SSC(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
                                KerPolyOrder=self.KerPolyOrder, BGPolyOrder=self.BGPolyOrder, ConstPhotRatio=self.ConstPhotRatio, \
                                BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT)
                            print('\nMeLOn Report: Compiling Functions in SFFT Subtraction Takes [%.3f s]' %(time.time() - Tcomp_start))

                            # * Perform SFFT Subtraction
                            from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract

                            SatMask_REF = SFFTPrepDict['REF-SAT-Mask']
                            SatMask_SCI = SFFTPrepDict['SCI-SAT-Mask']
                            NaNmask_U = SFFTPrepDict['Union-NaN-Mask']
                            PixA_mREF = SFFTPrepDict['PixA_mREF']
                            PixA_mSCI = SFFTPrepDict['PixA_mSCI']

                            if ConvdSide == 'REF':
                                PixA_mI, PixA_mJ = PixA_mREF, PixA_mSCI
                                if NaNmask_U is not None:
                                    PixA_I, PixA_J = PixA_REF.copy(), PixA_SCI.copy()
                                    PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                                    PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
                                else: PixA_I, PixA_J = PixA_REF, PixA_SCI
                                if self.MaskSatContam: 
                                    ContamMask_I = SatMask_REF
                                    ContamMask_J = SatMask_SCI
                                else: ContamMask_I = None

                            if ConvdSide == 'SCI':
                                PixA_mI, PixA_mJ = PixA_mSCI, PixA_mREF
                                if NaNmask_U is not None:
                                    PixA_I, PixA_J = PixA_SCI.copy(), PixA_REF.copy()
                                    PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                                    PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
                                else: PixA_I, PixA_J = PixA_SCI, PixA_REF
                                if self.MaskSatContam: 
                                    ContamMask_I = SatMask_SCI
                                    ContamMask_J = SatMask_REF
                                else: ContamMask_I = None
                            
                            Tsub_start = time.time()
                            _tmp = GeneralSFFTSubtract.GSS(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
                                SFFTConfig=SFFTConfig, ContamMask_I=ContamMask_I, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
                                CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT)
                            Solution, PixA_DIFF, ContamMask_CI = _tmp
                            if self.MaskSatContam:
                                ContamMask_DIFF = np.logical_or(ContamMask_CI, ContamMask_J)
                            print('\nMeLOn Report: SFFT Subtraction Takes [%.3f s]' %(time.time() - Tsub_start))
                            
                            # * Modifications on the difference image
                            #   a) when REF is convolved, DIFF = SCI - Conv(REF)
                            #      PSF(DIFF) is coincident with PSF(SCI), transients on SCI are positive signal in DIFF.
                            #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
                            #      PSF(DIFF) is coincident with PSF(REF), transients on SCI are still positive signal in DIFF.

                            if NaNmask_U is not None:
                                # ** Mask Union-NaN region
                                PixA_DIFF[NaNmask_U] = np.nan
                            if self.MaskSatContam:
                                # ** Mask Saturate-Contaminate region 
                                PixA_DIFF[ContamMask_DIFF] = np.nan
                            if ConvdSide == 'SCI': 
                                # ** Flip difference when science is convolved
                                PixA_DIFF = -PixA_DIFF
                            
                            # * Save difference image
                            if FITS_DIFF is not None:
                                _hdl = fits.open(FITS_SCI)
                                _hdl[0].data[:, :] = PixA_DIFF.T
                                _hdl[0].header['NAME_REF'] = (pa.basename(FITS_REF), 'MeLOn: SFFT')
                                _hdl[0].header['NAME_SCI'] = (pa.basename(FITS_SCI), 'MeLOn: SFFT')
                                _hdl[0].header['FWHM_REF'] = (FWHM_REF, 'MeLOn: SFFT')
                                _hdl[0].header['FWHM_SCI'] = (FWHM_SCI, 'MeLOn: SFFT')
                                _hdl[0].header['KERORDER'] = (self.KerPolyOrder, 'MeLOn: SFFT')
                                _hdl[0].header['BGORDER'] = (self.BGPolyOrder, 'MeLOn: SFFT')
                                _hdl[0].header['CPHOTR'] = (str(self.ConstPhotRatio), 'MeLOn: SFFT')
                                _hdl[0].header['KERHW'] = (KerHW, 'MeLOn: SFFT')
                                _hdl[0].header['CONVD'] = (ConvdSide, 'MeLOn: SFFT')
                                _hdl.writeto(FITS_DIFF, overwrite=True)
                                _hdl.close()
                            
                            # * Save solution array
                            if FITS_Solution is not None:
                                phdu = fits.PrimaryHDU()
                                phdu.header['N0'] = (SFFTConfig[0]['N0'], 'MeLOn: SFFT')
                                phdu.header['N1'] = (SFFTConfig[0]['N1'], 'MeLOn: SFFT')
                                phdu.header['DK'] = (SFFTConfig[0]['DK'], 'MeLOn: SFFT')
                                phdu.header['DB'] = (SFFTConfig[0]['DB'], 'MeLOn: SFFT')
                                phdu.header['L0'] = (SFFTConfig[0]['L0'], 'MeLOn: SFFT')
                                phdu.header['L1'] = (SFFTConfig[0]['L1'], 'MeLOn: SFFT')
                                
                                phdu.header['FIJ'] = (SFFTConfig[0]['Fij'], 'MeLOn: SFFT')
                                phdu.header['FAB'] = (SFFTConfig[0]['Fab'], 'MeLOn: SFFT')
                                phdu.header['FPQ'] = (SFFTConfig[0]['Fpq'], 'MeLOn: SFFT')
                                phdu.header['FIJAB'] = (SFFTConfig[0]['Fijab'], 'MeLOn: SFFT')

                                PixA_Solution = Solution.reshape((-1, 1))
                                phdu.data = PixA_Solution.T
                                fits.HDUList([phdu]).writeto(FITS_Solution, overwrite=True)

                        NEW_STATUS = 2
                        _message = 'Successful Subtraction for task-[%d] ' %taskidx_acquired
                        _message += 'in Subtraction thread-[%d]!' %INDEX_THREAD_4SUBTRACT
                        print('\nMeLOn CheckPoint: %s' %_message)

                    except:
                        # ** free GPU memory when the subtraction fails
                        import cupy as cp
                        with cp.cuda.Device(int(CUDA_DEVICE_4SUBTRACT)):
                            mempool = cp.get_default_memory_pool()
                            pinned_mempool = cp.get_default_pinned_memory_pool()
                            mempool.free_all_blocks()
                            pinned_mempool.free_all_blocks()

                        NEW_STATUS = -2
                        _error_message = 'UnSuccessful Subtraction for task-[%d] ' %taskidx_acquired
                        _error_message += 'in Subtraction thread-[%d]!' %INDEX_THREAD_4SUBTRACT
                        print('\nMeLOn ERROR: %s' %_error_message)
                        Solution = None
                        PixA_DIFF = None

                    with lock:
                        DICT_STATUS_BAR['task-[%d]' %taskidx_acquired] = NEW_STATUS
                        DICT_PRODUCTS['task-[%d]' %taskidx_acquired]['Solution'] = Solution
                        DICT_PRODUCTS['task-[%d]' %taskidx_acquired]['PixA_DIFF'] = PixA_DIFF

            return None

        # * trigger the threads for the preprocessing and subtraction
        MyThreadQueue = []
        for INDEX_THREAD_4PREPROC in range(NUM_THREADS_4PREPROC):
            MyThread = threading.Thread(target=FUNC_4PREPROC, args=(INDEX_THREAD_4PREPROC,))
            MyThreadQueue.append(MyThread)
            MyThread.start()

        for INDEX_THREAD_4SUBTRACT in range(NUM_THREADS_4SUBTRACT):
            MyThread = threading.Thread(target=FUNC_4SUBTRCT, args=(INDEX_THREAD_4SUBTRACT,))
            MyThreadQueue.append(MyThread)
            MyThread.start()
  
        for MyThread in MyThreadQueue:
            MyThread.join()
        
        return DICT_STATUS_BAR, DICT_PRODUCTS
