import importlib
import numpy as np
from sfft.utils.PureCupyFFTKits import PureCupy_FFTKits
if importlib.util.find_spec('cupy') is not None:
    import cupy as cp

__last_update__ = "2024-09-22"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.6.1"

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
#||   The initial cupy version of the DeCorrelation Calculator was    ||#
#||   developed by Lauren Aldoroty (Duke) and Robert Knop (LBNL) with ||#
#||   our mentors during the hackathon. The code can be found at:     ||#
#||   https://github.com/laldoroty/sfft/blob/nasa_hackathon/sfft      ||#
#||   /utils/DeCorrelationCalculator.py.                              ||#
#||                                                                   ||#
#########################################################################

class PureCupy_DeCorrelation_Calculator:
    @staticmethod
    def PCDC(NX_IMG, NY_IMG, KERNEL_GPU_JQueue, BKGSIG_JQueue, KERNEL_GPU_IQueue=[], BKGSIG_IQueue=[], 
        MATCH_KERNEL_GPU=None, REAL_OUTPUT=False, REAL_OUTPUT_SIZE=None, NORMALIZE_OUTPUT=True, VERBOSE_LEVEL=2):
        """Decorrelation Kernel Calculation in Pure Cupy"""
        NUM_I, NUM_J = len(KERNEL_GPU_IQueue), len(KERNEL_GPU_JQueue)
        assert NUM_J > 0
        if NUM_I == 0:
            MODE = 'IMAGE-STACKING'
            if NUM_J < 2:
                _error_message = 'IMAGE-STACKING MODE Requires at least 2 J-IMAGE!'
                raise Exception('MeLOn ERROR: %s' %_error_message)
            if np.sum([KERNEL_GPU is not None for KERNEL_GPU in KERNEL_GPU_JQueue]) == 0:
                _error_message = 'IMAGE-STACKING MODE Requires at least 1 non-None J-KERNEL!'
                raise Exception('MeLOn ERROR: %s' %_error_message)

        if NUM_I >= 1:
            MODE = 'IMAGE-SUBTRACTION'
            if NUM_J == 0: 
                _error_message = 'IMAGE-SUBTRACTION MODE Requires at least 1 I-IMAGE & 1 J-IMAGE!'
                raise Exception('MeLOn ERROR: %s' %_error_message)
            _Q = KERNEL_GPU_JQueue + KERNEL_GPU_IQueue + [MATCH_KERNEL_GPU]
            if np.sum([KERNEL_GPU is not None for KERNEL_GPU in _Q]) == 0:
                _error_message = 'IMAGE-SUBTRACTION MODE Requires at least 1 non-None J/I/MATCH-KERNEL!'
                raise Exception('MeLOn ERROR: %s' %_error_message)

        DELTA_KERNEL_GPU = cp.array([
            [0, 0, 0], 
            [0, 1, 0], 
            [0, 0, 0]], dtype=cp.float64
        )
        
        FDENO_GPU = None
        for KERNEL_GPU, BKGSIG in zip(KERNEL_GPU_JQueue, BKGSIG_JQueue):
            if KERNEL_GPU is not None:
                K_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=KERNEL_GPU, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            else:
                K_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=DELTA_KERNEL_GPU, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            FK_CSZ_GPU = cp.fft.fft2(K_CSZ_GPU)
            FK2_CSZ_GPU = (cp.conj(FK_CSZ_GPU) * FK_CSZ_GPU).real
            if FDENO_GPU is None:
                FDENO_GPU = (BKGSIG**2 * FK2_CSZ_GPU) / NUM_J**2
            else: 
                FDENO_GPU += (BKGSIG**2 * FK2_CSZ_GPU) / NUM_J**2
        
        if MATCH_KERNEL_GPU is not None:
            MK_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=MATCH_KERNEL_GPU, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
        else:
            MK_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=DELTA_KERNEL_GPU, NX_IMG=NX_IMG, NY_IMG=NY_IMG)

        FMK_CSZ_GPU = cp.fft.fft2(MK_CSZ_GPU)
        FMK2_CSZ_GPU = (cp.conj(FMK_CSZ_GPU) * FMK_CSZ_GPU).real
        
        for KERNEL_GPU, BKGSIG in zip(KERNEL_GPU_IQueue, BKGSIG_IQueue):
            if KERNEL_GPU is not None:
                K_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=KERNEL_GPU, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            else:
                K_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=DELTA_KERNEL_GPU, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            FK_CSZ_GPU = cp.fft.fft2(K_CSZ_GPU)
            FK2_CSZ_GPU = (cp.conj(FK_CSZ_GPU) * FK_CSZ_GPU).real
            FDENO_GPU += (BKGSIG**2 * FK2_CSZ_GPU * FMK2_CSZ_GPU) / NUM_I**2

        FDENO_GPU = cp.sqrt(FDENO_GPU)
        FKDECO_GPU = 1. / FDENO_GPU

        if not REAL_OUTPUT:
            if NORMALIZE_OUTPUT:
                NORMALIZE_FACTOR = 1./FKDECO_GPU[0, 0]
                FKDECO_GPU *= NORMALIZE_FACTOR
            return FKDECO_GPU
    
        if REAL_OUTPUT:
            KDECO_GPU = cp.fft.ifft2(FKDECO_GPU).real
            KDECO_GPU = PureCupy_FFTKits.KERNEL_CSZ_INV(
                KIMG_GPU=KDECO_GPU, NX_KERN=REAL_OUTPUT_SIZE[0], NY_KERN=REAL_OUTPUT_SIZE[1],
                VERBOSE_LEVEL=VERBOSE_LEVEL
            )
            
            if NORMALIZE_OUTPUT:
                assert REAL_OUTPUT_SIZE is not None
                NORMALIZE_FACTOR = 1./cp.sum(KDECO_GPU)
                KDECO_GPU *= NORMALIZE_FACTOR
            return KDECO_GPU
