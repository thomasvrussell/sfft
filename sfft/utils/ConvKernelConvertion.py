import numpy as np
import cupy

__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.0"

# * Remarks on Convolution Theorem
#    NOTE on Symbols: L theoretically can be even / odd, recall w = (L-1)//2 & w' = L//2 and w+w' = L-1
#    a. To make use of FFT with convolution theorem, it is necessary to convert convolution-kernel 
#        with relatively small size by Circular-Shift & tail-Zero-padding (CSZ) to align the Image-Size.
#    b. If we have an image elementwise-multiplication FI * FK, where FI and FK are conjugate symmetric,
#        Now we want to get its convolution counterpart in real space, we can calculate the convolution kernel by iCSZ on IDFT(FK).
#        In this process, we have to check the lost weight by size truncation, if the lost weight is basically inappreciable, 
#        the equivalent convolution is a good approximation.

class ConvKernel_Convertion:
    def CSZ(ConvKernel, N0, N1):
        L0, L1 = ConvKernel.shape  
        w0, w1 = (L0-1) // 2, (L1-1) // 2      
        pd0, pd1 = N0 - L0, N1 - L1
        TailZP = cupy.pad(ConvKernel, ((0, pd0), (0, pd1)), 'constant', constant_values=(0, 0))    # Tail-Zero-Padding
        KIMG_CSZ = cupy.roll(np.roll(TailZP, -w0, axis=0), -w1, axis=1)    # Circular-Shift
        return KIMG_CSZ

    def iCSZ(KIMG, L0, L1):
        N0, N1 = KIMG.shape
        w0, w1 = (L0-1) // 2, (L1-1) // 2
        KIMG_iCSZ = cupy.roll(cupy.roll(KIMG, w1, axis=1), w0, axis=0)    # inverse Circular-Shift
        ConvKernel = KIMG_iCSZ[:L0, :L1]    # Tail-Truncation 
        lost_weight = 1.0 - cupy.sum(cupy.abs(ConvKernel)) / cupy.sum(cupy.abs(KIMG_iCSZ))
        print('MeLOn CheckPoint: Tail-Truncation Lost-Weight [%.4f %s] (Absolute Percentage Error) ' %(lost_weight*100, '%'))
        return ConvKernel

