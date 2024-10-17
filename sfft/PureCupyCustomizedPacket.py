import time
import numpy as np
import importlib.util
from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract_PureCupy
from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure
if importlib.util.find_spec('cupy') is not None:
    import cupy as cp

__last_update__ = "2024-09-21"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.6.0"

#########################################################################
#||                                                                   ||#
#||                           Acknowledgment                          ||#
#||                                                                   ||#
#########################################################################
#||                                                                   ||#
#||   This code was developed during the NASA GPU Hackathon 2024      ||#
#||   (https://www.nas.nasa.gov/hackathon/#home) as part of the       ||#
#||   pipeline optimization efforts for the Roman Supernova PIT       ||#
#||   team (https://github.com/Roman-Supernova-PIT). I extend my      ||#
#||   sincere appreciation to my team members and our Hackathon       ||#
#||   mentors for their valuable support and insightful               ||#
#||   suggestions.                                                    ||#
#||                                                                   ||#
#||   I would like to specifically acknowledge the contributions      ||#
#||   of the following team members: Lauren Aldoroty (Duke),          ||#
#||   Robert Knop (LBNL), Shu Liu (Pitt), and Michael Wood-Vasey      ||#
#||   (Pitt). Additionally, I am grateful for the guidance provided   ||#
#||   by our mentors, Marcus Manos and Lucas Erlandson from NVIDIA.   ||#
#||                                                                   ||#
#||   The collaborative environment fostered by the NASA GPU          ||#
#||   Hackathon and the Roman Supernova PIT team was essential for    ||#
#||   the development and optimization of this code.                  ||#
#||                                                                   ||#
#########################################################################

class PureCupy_Customized_Packet:
    @staticmethod
    def PCCP(PixA_REF_GPU, PixA_SCI_GPU, PixA_mREF_GPU, PixA_mSCI_GPU, 
        ForceConv, GKerHW, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
        CUDA_DEVICE_4SUBTRACT='0', VERBOSE_LEVEL=2):
        """
        * Parameters for Cupy Customized SFFT (Pure Cupy Version)
        # @ Customized: Skip built-in preprocessing (automatic image-masking) by a customized masked image-pair.
        #
        # ----------------------------- Computing Enviornment --------------------------------- #

        -CUDA_DEVICE_4SUBTRACT ['0']        # it specifies certain GPU device (index) to conduct the subtraction task.
                                            # the GPU devices are usually numbered 0 to N-1 (you may use command nvidia-smi to check).
                                            # this argument becomes trivial for Numpy backend.

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv []                       # it determines which image will be convolved, can be 'REF' or 'SCI'.
                                            # here ForceConv CANNOT be 'AUTO'!

        -GKerHW []                          # the given kernel half-width. 

        -KerPolyOrder [2]                   # Polynomial degree of kernel spatial variation.

        -BGPolyOrder [2]                    # Polynomial degree of background spatial variation.

        -ConstPhotRatio [True]              # Constant photometric ratio between images ? can be True or False
                                            # ConstPhotRatio = True: the sum of convolution kernel is restricted to be a 
                                            #                constant across the field. 
                                            # ConstPhotRatio = False: the flux scaling between images is modeled by a 
                                            #                polynomial with degree -KerPolyOrder.

        # ----------------------------- Input & Output --------------------------------- #

        -PixA_REF_GPU []                    # Cupy ndarray for input reference image.

        -PixA_SCI_GPU []                    # Cupy ndarray for input science image.

        -PixA_mREF_GPU []                   # Cupy ndarray for input masked reference image (NaN-free).

        -PixA_mSCI_GPU []                   # Cupy ndarray for input masked science image (NaN-free).

        -PixA_DIFF_GPU [None]               # Cupy ndarray for output difference image.

        -PixA_Solution_GPU [None]           # Cupy ndarray for the solution of the linear system.
                                            # an array of (..., a_ijab, ... b_pq, ...).

        # ----------------------------- Miscellaneous --------------------------------- #
        
        -VERBOSE_LEVEL [2]                  # The level of verbosity, can be [0, 1, 2]
                                            # 0/1/2: QUIET/NORMAL/FULL mode

        # Important Notice:
        #
        # a): if reference is convolved in SFFT (-ForceConv='REF'), then DIFF = SCI - Convolved_REF.
        #     [the difference image is expected to have PSF & flux zero-point consistent with science image]
        #
        # b): if science is convolved in SFFT (-ForceConv='SCI'), then DIFF = Convolved_SCI - REF
        #     [the difference image is expected to have PSF & flux zero-point consistent with reference image]
        #
        # Remarks: this convention is to guarantee that transients emerge on science image always 
        #          show a positive signal on difference images.

        """

        # * assertions
        assert cp.sum(cp.isnan(PixA_mREF_GPU)) == 0, "The masked reference image contains NaNs!"
        assert cp.sum(cp.isnan(PixA_mSCI_GPU)) == 0, "The masked science image contains NaNs!"

        assert PixA_REF_GPU.ndim == 2, "The input PixA_REF_GPU is not two-dimensional!"
        assert PixA_SCI_GPU.ndim == 2, "The input PixA_SCI_GPU is not two-dimensional!"
        assert PixA_mREF_GPU.ndim == 2, "The input PixA_mREF_GPU is not two-dimensional!"
        assert PixA_mSCI_GPU.ndim == 2, "The input PixA_mSCI_GPU is not two-dimensional!"

        assert PixA_REF_GPU.dtype == cp.float64, "The array does not have dtype cp.float64!"
        assert PixA_SCI_GPU.dtype == cp.float64, "The array does not have dtype cp.float64!"
        assert PixA_mREF_GPU.dtype == cp.float64, "The array does not have dtype cp.float64!"
        assert PixA_mSCI_GPU.dtype == cp.float64, "The array does not have dtype cp.float64!"

        assert ForceConv in ['REF', 'SCI']
        ConvdSide = ForceConv
        KerHW = GKerHW

        device = cp.cuda.Device(int(CUDA_DEVICE_4SUBTRACT))
        device.use()
        
        # * Create the union NaN mask
        NaNmask_REF_GPU = cp.isnan(PixA_REF_GPU)
        NaNmask_SCI_GPU = cp.isnan(PixA_SCI_GPU)

        NaNmask_GPU = None
        if NaNmask_REF_GPU.any() or NaNmask_SCI_GPU.any():
            NaNmask_GPU = cp.logical_or(NaNmask_REF_GPU, NaNmask_SCI_GPU)

        # * Compile Functions in SFFT Subtraction
        if VERBOSE_LEVEL in [0, 1, 2]:
            print('MeLOn CheckPoint: TRIGGER Function Compilations of SFFT-SUBTRACTION!')

        Tcomp_start = time.time()
        NX, NY = PixA_REF_GPU.shape
        SFFTConfig = SingleSFFTConfigure.SSC(NX=NX, NY=NY, KerHW=KerHW, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            BACKEND_4SUBTRACT="Cupy", VERBOSE_LEVEL=VERBOSE_LEVEL)

        if VERBOSE_LEVEL in [1, 2]:
            _message = 'Function Compilations of SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tcomp_start)
            print('\nMeLOn Report: %s' %_message)

        # * Perform SFFT Subtraction
        if ConvdSide == 'REF':
            PixA_mI_GPU, PixA_mJ_GPU = PixA_mREF_GPU, PixA_mSCI_GPU
            if NaNmask_GPU is not None:
                PixA_I_GPU, PixA_J_GPU = PixA_REF_GPU.copy(), PixA_SCI_GPU.copy()
                PixA_I_GPU[NaNmask_GPU] = PixA_mI_GPU[NaNmask_GPU]
                PixA_J_GPU[NaNmask_GPU] = PixA_mJ_GPU[NaNmask_GPU]
            else: PixA_I_GPU, PixA_J_GPU = PixA_REF_GPU, PixA_SCI_GPU

        if ConvdSide == 'SCI':
            PixA_mI_GPU, PixA_mJ_GPU = PixA_mSCI_GPU, PixA_mREF_GPU
            if NaNmask_GPU is not None:
                PixA_I_GPU, PixA_J_GPU = PixA_SCI_GPU.copy(), PixA_REF_GPU.copy()
                PixA_I_GPU[NaNmask_GPU] = PixA_mI_GPU[NaNmask_GPU]
                PixA_J_GPU[NaNmask_GPU] = PixA_mJ_GPU[NaNmask_GPU]
            else: PixA_I_GPU, PixA_J_GPU = PixA_SCI_GPU, PixA_REF_GPU

        Tsub_start = time.time()
        Solution_GPU, PixA_DIFF_GPU, _ = GeneralSFFTSubtract_PureCupy.GSS(
            PixA_I_GPU=PixA_I_GPU, PixA_J_GPU=PixA_J_GPU, PixA_mI_GPU=PixA_mI_GPU, PixA_mJ_GPU=PixA_mJ_GPU, \
            SFFTConfig=SFFTConfig, ContamMask_I_GPU=None, VERBOSE_LEVEL=VERBOSE_LEVEL
        )
        if VERBOSE_LEVEL in [1, 2]:
            _message = 'SFFT-SUBTRACTION TAKES [%.3f s]' %(time.time() - Tsub_start)
            print('\nMeLOn Report: %s' %_message)

        # * Modifications on the difference image
        #   a) when REF is convolved, DIFF = SCI - Conv(REF)
        #      PSF(DIFF) is coincident with PSF(SCI), transients on SCI are positive signal in DIFF.
        #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
        #      PSF(DIFF) is coincident with PSF(REF), transients on SCI are still positive signal in DIFF.

        if NaNmask_GPU is not None:
            # ** Mask Union-NaN region
            PixA_DIFF_GPU[NaNmask_GPU] = np.nan
        
        if ConvdSide == 'SCI': 
            # ** Flip difference when science is convolved
            PixA_DIFF_GPU *= -1.
            
        return Solution_GPU, PixA_DIFF_GPU
