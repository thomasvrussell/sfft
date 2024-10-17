import importlib
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
#########################################################################

class PureCupy_FFTKits:
    @staticmethod
    def KERNEL_CSZ(KERNEL_GPU, NX_IMG, NY_IMG, NORMALIZE_KERNEL=False):
        """ Circular Shift the kernel and extend to the target size """
        N0, N1 = NX_IMG, NY_IMG
        L0, L1 = KERNEL_GPU.shape
        W0, W1 = (L0 - 1)//2, (L1 - 1)//2
        # Note: currently only support odd-sized kernel
        assert L0 % 2 == 1 and L1 % 2 == 1   
        
        if NORMALIZE_KERNEL:
            KERNEL_TZP_GPU = cp.pad(KERNEL_GPU / cp.sum(KERNEL_GPU), \
                pad_width=((0, N0 - L0), (0, N1 - L1)), mode='constant', constant_values=0.)
        else:   
            KERNEL_TZP_GPU = cp.pad(KERNEL_GPU, pad_width=((0, N0 - L0), (0, N1 - L1)), \
                mode='constant', constant_values=0.)
        KIMG_CSZ_GPU = cp.roll(cp.roll(KERNEL_TZP_GPU, -W0, axis=0), -W1, axis=1)
        return KIMG_CSZ_GPU

    @staticmethod
    def KERNEL_CSZ_INV(KIMG_GPU, NX_KERN, NY_KERN, VERBOSE_LEVEL=2):
        """ Inverse Circular Shift the kernel and truncate to the target size """
        L0, L1 = NX_KERN, NY_KERN
        W0, W1 = (L0 - 1)//2, (L1 - 1)//2
        # Note: currently only support odd-sized kernel
        assert L0 % 2 == 1 and L1 % 2 == 1
        
        KIMG_iCSZ_GPU = cp.roll(cp.roll(KIMG_GPU, W1, axis=1), W0, axis=0)
        KERNEL_GPU = KIMG_iCSZ_GPU[:L0, :L1]
        if VERBOSE_LEVEL in [1, 2]:
            LOSE_RATIO = 1. - cp.sum(cp.abs(KERNEL_GPU)) / cp.sum(cp.abs(KIMG_iCSZ_GPU))
            _report_message = "Kernel Truncation Loses APE = [%.4f %s]" %(LOSE_RATIO*100, '%')
            print("MeLOn CheckPoint: %s " % _report_message)
        return KERNEL_GPU

    @staticmethod
    def FFT_CONVOLVE(PixA_Inp_GPU, KERNEL_GPU, PAD_FILL_VALUE=0., NAN_FILL_VALUE=0., 
        NORMALIZE_KERNEL=False, FORCE_OUTPUT_C_CONTIGUOUS=False, FFT_BACKEND="Cupy"):
        """ FFT Convolition """
        N0, N1 = PixA_Inp_GPU.shape
        L0, L1 = KERNEL_GPU.shape
        assert L0 % 2 == 1 and L1 % 2 == 1

        W0, W1 = (L0 - 1)//2, (L1 - 1)//2 
        NX_IMG, NY_IMG = N0 + 2*W0, N1 + 2*W1

        # zero padding on input image
        PixA_EInp_GPU = cp.pad(
            array=PixA_Inp_GPU, 
            pad_width=((W0, W0), (W1, W1)), 
            mode='constant', 
            constant_values=PAD_FILL_VALUE
        )

        if NAN_FILL_VALUE is not None:
            PixA_EInp_GPU[cp.isnan(PixA_EInp_GPU)] = NAN_FILL_VALUE
        
        # circular shift the kernel to the center and extend to the target size
        KIMG_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL_GPU=KERNEL_GPU, 
            NX_IMG=NX_IMG, NY_IMG=NY_IMG, NORMALIZE_KERNEL=NORMALIZE_KERNEL)

        # perform convolution in Fourier domain
        if FFT_BACKEND == "Cupy":
            PixA_Out_GPU = cp.fft.ifft2(
                cp.fft.fft2(PixA_EInp_GPU) * cp.fft.fft2(KIMG_CSZ_GPU)
            ).real[W0: -W0, W1: -W1]

        if FORCE_OUTPUT_C_CONTIGUOUS:
            if not PixA_Out_GPU.flags['C_CONTIGUOUS']:
                PixA_Out_GPU = cp.ascontiguousarray(PixA_Out_GPU)
        return PixA_Out_GPU
