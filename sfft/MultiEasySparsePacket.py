import time
import math
import warnings
import threading
import numpy as np
import os.path as pa
from astropy.io import fits
import scipy.ndimage as ndimage
from astropy.table import Column
from sfft.AutoSparsePrep import Auto_SparsePrep
from sfft.utils.meta.TimeoutKit import TimeoutAfter
# version: Aug 31, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.3"

class MultiEasy_SparsePacket:
    def __init__(self, FITS_REF_Queue, FITS_SCI_Queue, FITS_DIFF_Queue=[], FITS_Solution_Queue=[], \
        ForceConv_Queue=[], GKerHW_Queue=[], KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=0, \
        ConstPhotRatio=True, MaskSatContam=False, GAIN_KEY='GAIN', SATUR_KEY='ESATUR', BACK_TYPE='MANUAL', \
        BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=2.0, DETECT_MINAREA=5, \
        DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', ONLY_FLAGS=[0], BoundarySIZE=30, \
        XY_PriorSelect_Queue=[], Hough_FRLowerLimit=0.1, Hough_peak_clip=0.7, BeltHW=0.2, PS_ELLIPThresh=0.3, \
        MatchTol=None, MatchTolFactor=3.0, COARSE_VAR_REJECTION=True, CVREJ_MAGD_THRESH=0.12, \
        ELABO_VAR_REJECTION=True, EVREJ_RATIO_THREH=5.0, EVREJ_SAFE_MAGDEV=0.04, StarExt_iter=4, \
        XY_PriorBan_Queue=[], PostAnomalyCheck=False, PAC_RATIO_THRESH=5.0):

        """
        # NOTE: This function is to Perform Sparse-Flavor SFFT for multiple tasks:
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

        # ----------------------------- Preprocessing with Source Selection for Image-Masking --------------------------------- #

        # > Configurations for SExtractor

        -GAIN_KEY ['GAIN']                  # SExtractor Parameter GAIN_KEY
                                            # i.e., keyword of GAIN in FITS header (of reference & science)

        -SATUR_KEY ['ESATUR']               # SExtractor Parameter SATUR_KEY
                                            # i.e., keyword of effective saturation in FITS header (of reference & science)
                                            # Remarks: one may think 'SATURATE' is a more common keyword name for saturation level.
                                            #          However, note that Sparse-Flavor SFFT requires sky-subtracted images as inputs, 
                                            #          we need deliever the 'effective' saturation level after the sky-subtraction.
                                            #          e.g., set ESATURA = SATURATE - (SKY + 10*SKYSIG)

        -BACK_TYPE ['MANUAL']               # SExtractor Parameter BACK_TYPE = [AUTO or MANUAL].
                                            # As Sparse-Flavor-SFFT requires the input images being sky-subtracted,
                                            # the default setting uses zero-background (i.e., BACK_TYPE='MANUAL' & BACK_VALUE=0.0).
                                            # However, one may also use BACK_TYPE='AUTO' for some specific cases
                                            # E.g., point sources located at the outskirts of a bright galaxy will have a relatively
                                            #       high local background. Even if the image has been sky-subtracted properly, 
                                            #       SExtractor can only detect it when you set BACK_TYPE='AUTO'.
         
        -BACK_VALUE [0.0]                   # SExtractor Parameter BACK_VALUE (only work for BACK_TYPE='MANUAL')

        -BACK_SIZE [64]                     # SExtractor Parameter BACK_SIZE

        -BACK_FILTERSIZE [3]                # SExtractor Parameter BACK_FILTERSIZE

        -DETECT_THRESH [2.0]                # SExtractor Parameter DETECT_THRESH

        -DETECT_MINAREA [5]                 # SExtractor Parameter DETECT_MINAREA
        
        -DETECT_MAXAREA [0]                 # SExtractor Parameter DETECT_MAXAREA

        -DEBLEND_MINCONT [0.005]            # SExtractor Parameter DEBLEND_MINCONT (typically, 0.001 - 0.005)

        -BACKPHOTO_TYPE ['LOCAL']           # SExtractor Parameter BACKPHOTO_TYPE

        -ONLY_FLAGS [0]                     # Restrict SExtractor Output Photometry Catalog by Source FLAGS
                                            # Common FLAGS (Here None means no restrictions on FLAGS):
                                            # 1: aperture photometry is likely to be biased by neighboring sources 
                                            #    or by more than 10% of bad pixels in any aperture
                                            # 2: the object has been deblended
                                            # 4: at least one object pixel is saturated

        -BoundarySIZE [30]                  # Restrict SExtractor Output Photometry Catalog by Dropping Sources at Boundary 
                                            # NOTE: This would help to avoid selecting sources too close to image boundary. 

        # > Configurations for Source-Selction

        -XY_PriorSelect_Queue [[]]      # A queue of -XY_PriorSelect in sfft.Easy_SparsePacket.
                                        # Recall -XY_PriorSelect: 
                                        # a Numpy array of pixels coordinates, 
                                        # with shape (N, 2) (e.g., [[x0, y0], [x1, y1], ...]).
                                        # this allows sfft to us a prior source selection to solve the subtraction.
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE

        -Hough_FRLowerLimit [0.1]       # The lower bound of FLUX_RATIO for line feature detection using Hough transformation.
                                        # Setting a proper lower bound can avoid to detect some line features by chance,
                                        # which are not contributed from point sources but resides in the small-FLUX_RATIO region.
                                        # NOTE: Recommended values of Hough_FRLowerLimit: 0.1 ~ 1.0
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE

        -Hough_peak_clip [0.7]          # It determines the lower bound of the sensitivity of the line feature detection.
                                        # NOTE: When the point-source-belt is not very pronounced (e.g., in galaxy dominated fields),
                                        #       one may consider to reduce the parameter from default 0.7 to, says, ~ 0.4.
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE

        -BeltHW [0.2]                   # The half-width of point-source-belt detected by Hough Transformation.
                                        # Remarks: if you want to tune this parameter, it is helpful to draw 
                                        #          a figure of MAG_AUTO against FLUX_RADIUS.
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: determine point-source-belt

        -PS_ELLIPThresh [0.3]           # An additiona Restriction on ELLIPTICITY (ELLIPTICITY < PS_ELLIPThresh) for point sources.
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: determine point-sources
                                    
        -MatchTol [None]                # Given separation tolerance (pix) for source matching 
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: Cross-Match between REF & SCI
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: Cross-Match between REF & SCI AND
                                        #       Cross-Match between the prior selection and mean image coordinates of matched REF-SCI sources.

        -MatchTolFactor [3.0]           # The separation tolerance (pix) for source matching calculated by FWHM,
                                        # MatchTol = np.sqrt((FWHM_REF/MatchTolFactor)**2 + (FWHM_SCI/MatchTolFactor)**2)
                                        # @ Given precise WCS, one can use a high MatchTolFactor ~3.0
                                        # @ For very sparse fields where WCS can be inaccurate, 
                                        #   one can loosen the tolerance with a low MatchTolFactor ~1.0
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: Cross-Match between REF & SCI
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: Cross-Match between REF & SCI AND
                                        #       Cross-Match between the prior selection and mean image coordinates of matched REF-SCI sources.

        -COARSE_VAR_REJECTION [True]    # Activate Coarse Variable Rejection (CVREJ) or not.
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE 

        -CVREJ_MAGD_THRESH [0.12]       # CVREJ kicks out the obvious variables by the difference of 
                                        # instrument magnitudes (MAG_AUTO) measured on reference and science, i.e., 
                                        # MAG_AUTO_SCI - MAG_AUTO_REF. A source with difference highly deviated from
                                        # the median level of the field stars will be rejected from the source selection.
                                        # -CVREJ_MAGD_THRESH is the deviation magnitude threshold. 
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE 
        
        -ELABO_VAR_REJECTION [True]     # WARNING: EVREJ REQUIRES CORRECT GAIN IN FITS HEADER!
                                        # Activate Elaborate Variable Rejection (EVREJ) or not.
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE 

        -EVREJ_RATIO_THREH [5.0]        # EVREJ kicks out the variables by the photometric uncertainties
                                        # (SExtractor FLUX_AUTO) measured on reference and science.
                                        # The flux change of a stationary object from reference to science 
                                        # is a predictable probability ditribution. EVREJ will reject the 
                                        # -EVREJ_RATIO_THREH * SIGMA outliers of the probability distribution.
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE 
        
        -EVREJ_SAFE_MAGDEV [0.04]       # A protection to avoid EVREJ overkilling. EVREJ would not reject
                                        # a source if the difference of instrument magnitudes is deviated
                                        # from the median level within -EVREJ_SAFE_MAGDEV magnitude.
                                        # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE 
                                        #
                                        # @ Why does EVREJ overkill?
                                        # i) The calculated flux change is entangled with the error of 
                                        #    the constant photometric scaling used by EVREJ.
                                        # ii) SExtractor FLUXERR can be in accurate, especially at the bright end.
                                        # see more details in sfft.Auto_SparsePrep.HoughAutoMask

        -StarExt_iter [4]               # Our image mask is determined by the SExtractor check image SEGMENTATION 
                                        # of the selected sources. note that some pixels (e.g., outskirt region of a galaxy) 
                                        # harbouring signal may be not SExtractor-detectable due to the nature of 
                                        # the thresholding-based detection method (see NoiseChisel paper for more details). 
                                        # we want to include the missing light at outskirt region to contribute to
                                        # parameter-solving process, then a simple mask dilation is introduced.
                                        # -StarExt_iter means the iteration times of the dilation process.
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: mask dilation
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: mask dilation

        -XY_PriorBan_Queue [[]]         # A queue of -XY_PriorBan in sfft.Easy_SparsePacket. 
                                        # Recall -XY_PriorBan: 
                                        # a Numpy array of pixels coordinates, with shape (N, 2) (e.g., [[x0, y0], [x1, y1], ...])
                                        # this allows us to feed the prior knowledge about the varibility cross the field.
                                        # if you already get a list of variables (transients) and would like SFFT not to select
                                        # them to solve the subtraction, you can tell SFFT their coordinates through -XY_PriorBan.
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: mask refinement
                                        # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: mask refinement     

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv_Queue [[]]               # A queue of -ForceConv in sfft.Easy_SparsePacket.
                                            # Recall -ForceConv:
                                            # it determines which image will be convolved, can be 'REF', 'SCI' and 'AUTO'.
                                            # -ForceConv = 'AUTO' means SFFT will determine the convolution direction according to 
                                            # FWHM_SCI and FWHM_REF: the image with better seeing will be convolved to avoid deconvolution.

        -GKerHW_Queue [[]]                  # A queue of -GKerHW in sfft.Easy_SparsePacket.
                                            # Recall -GKerHW:
                                            # The given kernel half-width, None means the kernel size will be 
                                            # automatically determined by -KerHWRatio (to be seeing-related). 

        -KerHWRatio [2.0]                   # The ratio between FWHM and the kernel half-width
                                            # KerHW = int(KerHWRatio * Max(FWHM_REF, FWHM_SCI))

        -KerHWLimit [(2, 20)]               # The lower & upper bounds for kernel half-width 
                                            # KerHW is updated as np.clip(KerHW, KerHWLimit[0], KerHWLimit[1]) 
                                            # Remarks: this is useful for a survey since it can constrain the peak GPU memory usage.

        -KerPolyOrder [2]                   # Polynomial degree of kernel spatial variation, can be [0,1,2,3].

        -BGPolyOrder [0]                    # Polynomial degree of differential-background spatial variation, can be [0,1,2,3].
                                            # NOTE: this argument is trivial for Sparse-Flavor SFFT as input images have been sky subtracted.
                                            #       simply use a flat differential background (it will very close to 0 in the sfft solution). 

        -ConstPhotRatio [True]              # Constant photometric ratio between images? can be True or False
                                            # ConstPhotRatio = True: the sum of convolution kernel is restricted 
                                            #   to be a constant across the field. 
                                            # ConstPhotRatio = False: the flux scaling between images is modeled
                                            #   by a polynomial with degree -KerPolyOrder.

        -MaskSatContam [False]              # Mask saturation-contaminated regions on difference image ? can be True or False
                                            # NOTE the pixels enclosed in the regions are replaced by NaN.
        
        # ----------------------------- Post Subtraction --------------------------------- #

        -PostAnomalyCheck [False]           # WARNING: PAC REQUIRES CORRECT GAIN IN FITS HEADER!
                                            # Post Anomaly Check (PAC) is to find the missing variables in the source selection 
                                            # on the difference image. Like EVREJ, it depends on the photometric uncertainties 
                                            # given by SExtractor FLUXERR_AUTO of reference and science image.
                                            # As image subtraction can get a more accurate photometric scaling between reference 
                                            # and science than that in EVREJ, PA is a even more 'elaborate' way to identify variables.
                                            # NOTE: one can make use of the Post-Anomalies as the Prior-Ban-Sources 
                                            #       to refine the subtraction (run second-time subtraction).

        -PAC_RATIO_THRESH [3.0]             # PA will identify the -PAC_RATIO_THRESH * SIGMA outliers of the 
                                            # probability distribution as missing variables.

        # ----------------------------- Input & Output --------------------------------- #

        -FITS_REF_Queue [[]]                # A queue of -FITS_REF in sfft.Easy_SparsePacket.
                                            # Recall -FITS_REF: File path of input reference image.

        -FITS_SCI_Queue [[]]                # A queue of -FITS_SCI in sfft.Easy_SparsePacket.
                                            # Recall -FITS_SCI: File path of input science image.

        -FITS_DIFF_Queue [[]]               # A queue of -FITS_DIFF in sfft.Easy_SparsePacket.
                                            # Recall -FITS_DIFF:File path of output difference image.

        -FITS_Solution_Queue [[]]           # A queue of -FITS_Solution in sfft.Easy_SparsePacket.
                                            # Recall -FITS_Solution: File path of the solution of the linear system. 
                                            #   it is an array of (..., a_ijab, ... b_pq, ...).

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
        self.NUM_TASK = len(FITS_SCI_Queue)

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

        self.Hough_FRLowerLimit = Hough_FRLowerLimit
        self.Hough_peak_clip = Hough_peak_clip
        self.BeltHW = BeltHW
        self.PS_ELLIPThresh = PS_ELLIPThresh

        self.MatchTol = MatchTol
        self.MatchTolFactor = MatchTolFactor

        self.COARSE_VAR_REJECTION = COARSE_VAR_REJECTION
        self.CVREJ_MAGD_THRESH = CVREJ_MAGD_THRESH
        self.ELABO_VAR_REJECTION = ELABO_VAR_REJECTION
        self.EVREJ_RATIO_THREH = EVREJ_RATIO_THREH
        self.EVREJ_SAFE_MAGDEV = EVREJ_SAFE_MAGDEV
        
        self.StarExt_iter = StarExt_iter

        self.PostAnomalyCheck = PostAnomalyCheck
        self.PAC_RATIO_THRESH = PAC_RATIO_THRESH
        
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

        if XY_PriorSelect_Queue == []:
            _warn_message = 'Argument XY_PriorSelect_Queue is EMPTY then Use default [...None...] instead!'
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
            self.XY_PriorSelect_Queue = [None] * self.NUM_TASK
        else: self.XY_PriorSelect_Queue = XY_PriorSelect_Queue

        if XY_PriorBan_Queue == []:
            _warn_message = 'Argument XY_PriorBan_Queue is EMPTY then Use default [...None...] instead!'
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
            self.XY_PriorBan_Queue = [None] * self.NUM_TASK
        else: self.XY_PriorBan_Queue = XY_PriorBan_Queue

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

        # * define the function for preprocessing (see sfft.EasySparsePacket)
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
                    XY_PriorBan = self.XY_PriorBan_Queue[taskidx_acquired]
                    XY_PriorSelect = self.XY_PriorSelect_Queue[taskidx_acquired]

                    try:
                        with TimeoutAfter(timeout=TIMEOUT_4PREPROC_EACHTASK, exception=TimeoutError):
                            warnings.warn('\nMeLOn WARNING: Input images for sparse-flavor sfft should be SKY-SUBTRACTED !!!')

                            # ** perform auto sparse-prep
                            _ASP = Auto_SparsePrep(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, GAIN_KEY=self.GAIN_KEY, \
                                SATUR_KEY=self.SATUR_KEY, BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, \
                                BACK_SIZE=self.BACK_SIZE, BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, \
                                DETECT_MINAREA=self.DETECT_MINAREA, DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, \
                                BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, ONLY_FLAGS=self.ONLY_FLAGS, BoundarySIZE=self.BoundarySIZE)

                            if XY_PriorSelect is None:
                                IMAGE_MASK_METHOD = 'HOUGH-AUTO'
                                print('MeLOn CheckPoint: TRIGGER Auto-Sparse-Prep [%s] MODE for task-[%d]!' %(IMAGE_MASK_METHOD, taskidx_acquired))
                                SFFTPrepDict = _ASP.HoughAutoMask(Hough_FRLowerLimit=self.Hough_FRLowerLimit, \
                                    Hough_peak_clip=self.Hough_peak_clip, BeltHW=self.BeltHW, PS_ELLIPThresh=self.PS_ELLIPThresh, \
                                    MatchTol=self.MatchTol, MatchTolFactor=self.MatchTolFactor, ELABO_VAR_REJECTION=self.ELABO_VAR_REJECTION, \
                                    EVREJ_RATIO_THREH=self.EVREJ_RATIO_THREH, EVREJ_SAFE_MAGDEV=self.EVREJ_SAFE_MAGDEV, \
                                    StarExt_iter=self.StarExt_iter, XY_PriorBan=XY_PriorBan)
                            
                            if XY_PriorSelect is not None:
                                IMAGE_MASK_METHOD = 'SEMI-AUTO'
                                print('MeLOn CheckPoint: TRIGGER Auto-Sparse-Prep [%s] MODE for task-[%d]!' %(IMAGE_MASK_METHOD, taskidx_acquired))
                                SFFTPrepDict = _ASP.SemiAutoMask(XY_PriorSelect=XY_PriorSelect, MatchTol=self.MatchTol, \
                                    MatchTolFactor=self.MatchTolFactor, StarExt_iter=self.StarExt_iter)

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

        # * define the function for sfft subtraction (see sfft.EasySparsePacket)
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

                            if self.PostAnomalyCheck:
                                warnings.warn('\nMeLOn WARNING: Post-Anomaly Check requires correct GAIN values!')
                                # TODO: Incorporate weight-maps of the input image pair for more accurate FLUXERR.

                                """
                                # Remarks on the Post Anomaly Check (PAC)
                                # [1] Recall that Elaborate Variable Rejection (EVREJ) also use SExtractor photometric errors to identify variables.
                                #     I would say, PAC should be more accurate than EVREJ.
                                #     NOTE: Bright transients (compared with its host) can be also identified by PAC.
                                #
                                # [2] distribution mean: image subtraction is expected to have better accuracy on the determination 
                                #     of the photometric scaling, compared with the constant MAG_OFFSET in EVREJ. As a result,
                                #     the flux change counted on the difference image is likely more 'real' than that in EVREJ.
                                #
                                # [3] distribution variance: i) inaccurate SExtractor FLUXERR especially at the bright end
                                #     and ii) the error of MAG_OFFSET affect one FLUXERR scaling.
                                #     NOTE: The imperfections on distribution variance are just same as EVREJ.
                                #     NOTE: Using SFFT kernel sum would be better than MAG_OFFSET for scaling FLUXERR. 
                                #           For simplicity, howerver, we use MAG_OFFSET (with error << 5%) which is good enough here.
                                #
                                """
                                
                                AstSEx_SS = SFFTPrepDict['SExCatalog-SubSource']
                                SFFTLmap = SFFTPrepDict['SFFT-LabelMap']
                                
                                # ** Only consider the Non-Prior-Banned SubSources
                                if 'MASK_PriorBan' in AstSEx_SS.colnames:
                                    nPBMASK_SS = ~np.array(AstSEx_SS['MASK_PriorBan'])
                                    AstSEx_vSS = AstSEx_SS[nPBMASK_SS]
                                else: AstSEx_vSS = AstSEx_SS
                                
                                # ** Estimate expected variance of Non-Prior-Banned SubSources on the difference
                                MAG_OFFSET = SFFTPrepDict['MAG_OFFSET']
                                FLUX_SCAL = 10**(MAG_OFFSET/-2.5)  # FLUX_SCI/FLUX_REF

                                FLUXERR_vSSr = np.array(AstSEx_vSS['FLUXERR_AUTO_REF'])
                                FLUXERR_vSSs = np.array(AstSEx_vSS['FLUXERR_AUTO_SCI'])

                                if ConvdSide == 'REF':
                                    sFLUXERR_vSSr = FLUXERR_vSSr * FLUX_SCAL
                                    ExpDVAR_vSS = sFLUXERR_vSSr**2 + FLUXERR_vSSs**2
                                if ConvdSide == 'SCI':
                                    sFLUXERR_vSSs = FLUXERR_vSSs / FLUX_SCAL
                                    ExpDVAR_vSS = FLUXERR_vSSr**2 + sFLUXERR_vSSs**2

                                # ** Measure the ratios of Non-Prior-Banned SubSources on the difference for PAC
                                SEGL_vSS = np.array(AstSEx_vSS['SEGLABEL']).astype(int)
                                DFSUM_vSS = ndimage.labeled_comprehension(PixA_DIFF, SFFTLmap, SEGL_vSS, np.sum, float, 0.0)
                                RATIO_vSS = DFSUM_vSS / np.clip(np.sqrt(ExpDVAR_vSS), a_min=1e-8, a_max=None)
                                PAMASK_vSS = np.abs(RATIO_vSS) > self.PAC_RATIO_THRESH

                                _message = 'FIND [%d] PostAnomaly SubSources [> %.2f sigma] ' %(np.sum(PAMASK_vSS), self.PAC_RATIO_THRESH)
                                _message += 'out of [%d] Non-Prior-Banned SubSources!\n' %(len(AstSEx_vSS))
                                _message += 'P.S. there are [%d] Prior-Banned SubSources!' %(len(AstSEx_SS) - len(AstSEx_vSS))
                                print('\nMeLOn CheckPoint: %s' %_message)
                                
                                # ** Record the results (decorate AstSEx_SS in SFFTPrepDict)
                                if 'MASK_PriorBan' in AstSEx_SS.colnames:
                                    ExpDVAR_SS = np.nan * np.ones(len(AstSEx_SS))       # NOTE Prior-Ban is trivial NaN
                                    DFSUM_SS = np.nan * np.ones(len(AstSEx_SS))         # NOTE Prior-Ban is trivial NaN
                                    RATIO_SS = np.nan * np.ones(len(AstSEx_SS))         # NOTE Prior-Ban is trivial NaN
                                    PAMASK_SS = np.zeros(len(AstSEx_SS)).astype(bool)   # NOTE Prior-Ban is trivial False
                                    
                                    ExpDVAR_SS[nPBMASK_SS] = ExpDVAR_vSS
                                    DFSUM_SS[nPBMASK_SS] = DFSUM_vSS
                                    RATIO_SS[nPBMASK_SS] = RATIO_vSS
                                    PAMASK_SS[nPBMASK_SS] = PAMASK_vSS
                                else: 
                                    ExpDVAR_SS = ExpDVAR_vSS
                                    DFSUM_SS = DFSUM_vSS
                                    RATIO_SS = RATIO_vSS
                                    PAMASK_SS = PAMASK_vSS

                                AstSEx_SS.add_column(Column(ExpDVAR_SS, name='ExpDVAR_PostAnomaly'))
                                AstSEx_SS.add_column(Column(DFSUM_SS, name='DFSUM_PostAnomaly'))
                                AstSEx_SS.add_column(Column(RATIO_SS, name='RATIO_PostAnomaly'))
                                AstSEx_SS.add_column(Column(PAMASK_SS, name='MASK_PostAnomaly'))

                            # * Modifications on the difference image
                            #   a) when REF is convolved, DIFF = SCI - Conv(REF)
                            #      PSF_DIFF is coincident with SCI, transients on SCI are positive signal in DIFF.
                            #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
                            #      PSF_DIFF is coincident with REF, transients on SCI are still positive signal in DIFF.

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
