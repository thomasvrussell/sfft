import sys
import numpy as np
from sfft.utils.ConvKernelConvertion import ConvKernel_Convertion

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class DeCorrelation_Calculator:
    @staticmethod
    def DCC(MK_JLst, SkySig_JLst, MK_ILst=[], SkySig_ILst=[], MK_Fin=None, KERatio=2.0):
        
        # * Remarks on Input
        #   i. Image-Stacking Mode: NumI = 0
        #      MK_Fin will not work, NumJ >= 2 and NOT all J's kernel are None.
        #   ii. Image-Subtraction Mode: NumI & NumJ >= 1
        #       NOT all (I / J / Fin) kernel are None

        # * Remarks on difference flip
        #   D = REF - SCI * K
        #   fD = SCI * K - REF = fREF - fSCI * K
        #   Since Var(REF) = Var(fREF) and Var(SCI) = Var(fSCI)
        #   fD and D have consistent decorrelation kernel. 
        
        N0, N1 = 256, 256   # trivial Image-Size, just typically larger than Kernel-Size
        NumI, NumJ = len(MK_ILst), len(MK_JLst)
        if NumI == 0: 
            Mode = 'Image-Stacking'
            if NumJ < 2: sys.exit('MeLOn ERROR: Image-Stacking Mode requires at least 2 J-images !')
            if np.sum([MKj is not None for MKj in MK_JLst]) == 0:
                sys.exit('MeLOn ERROR: Image-Stacking Mode requires at least 1 not-None J-kernel !')

        if NumI >= 1:
            Mode = 'Image-Subtraction'
            if NumJ == 0: sys.exit('MeLOn ERROR: Image-Subtraction Mode requires at least 1 I-image & 1 J-image !')
            if np.sum([MK is not None for MK in MK_JLst+MK_ILst+[MK_Fin]]) == 0:
                sys.exit('MeLOn ERROR: Image-Stacking Mode requires at least 1 not-None J/I/Fin-kernel !')
        
        # * Determine DeCorrelation Kernel-Size
        #   The DeCorrelation Kernel is derived in Fourier Space, but it is not proper to directly perform DeCorrelation
        #   in Fourier Space (equivalently, perform a convolution with Kernel-Size = Image-Size). 
        #   a. convolution (and resampling) process has very local effect, it is not necesseary to use large decorrelation kernel.
        #   b. if we use a large decorrelation kernel, you will find only very few engery distributed at outskirt regions, however 
        #      the low weight can degrade the decorrelation convolution by the remote saturated pixels.

        MK_Queue = MK_JLst.copy()
        if Mode == 'Image-Subtraction': MK_Queue += [MK_Fin] + MK_ILst
        L0_KDeCo = int(round(KERatio * np.max([MK.shape[0] for MK in MK_Queue if MK is not None])))
        L1_KDeCo = int(round(KERatio * np.max([MK.shape[1] for MK in MK_Queue if MK is not None])))
        if L0_KDeCo%2 == 0: L0_KDeCo += 1
        if L1_KDeCo%2 == 0: L1_KDeCo += 1
        print('MeLOn CheckPoint: DeCorrelation Kernel-Size [%d, %d]' %(L0_KDeCo, L1_KDeCo))

        # * Construct the DeNonimator (a real-positive map) in Fourier Space
        uMK = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype(float)
        def Get_JTerm(MKj, skysig):
            if MKj is not None: KIMG_CSZ = ConvKernel_Convertion.CSZ(MKj, N0, N1)
            if MKj is None: KIMG_CSZ = ConvKernel_Convertion.CSZ(uMK, N0, N1)
            kft = np.fft.fft2(KIMG_CSZ)
            kft2 = (np.conj(kft) * kft).real
            term = (skysig**2 * kft2) / NumJ**2
            return term
        
        if MK_Fin is not None: KIMG_CSZ = ConvKernel_Convertion.CSZ(MK_Fin, N0, N1)
        if MK_Fin is None: KIMG_CSZ = ConvKernel_Convertion.CSZ(uMK, N0, N1)
        kft = np.fft.fft2(KIMG_CSZ)
        kft2_Fin = (np.conj(kft) * kft).real

        def Get_ITerm(MKi, skysig):
            if MKi is not None: KIMG_CSZ = ConvKernel_Convertion.CSZ(MKi, N0, N1)
            if MKi is None: KIMG_CSZ = ConvKernel_Convertion.CSZ(uMK, N0, N1)
            kft = np.fft.fft2(KIMG_CSZ)
            kft2 = (np.conj(kft) * kft).real
            term = (skysig**2 * kft2 * kft2_Fin) / NumI**2 
            return term
        
        DeNo = 0.0
        for MKj, skysig in zip(MK_JLst, SkySig_JLst):
            DeNo += Get_JTerm(MKj, skysig)
        if Mode == 'Image-Subtraction':
            for MKi, skysig in zip(MK_ILst, SkySig_ILst):
                DeNo += Get_ITerm(MKi, skysig)

        FDeCo = np.sqrt(1.0 / DeNo)                # Real & Conjugate-Symmetric (from formula)
        DeCo = np.fft.ifft2(FDeCo).real            # NO Imaginary part
        KDeCo = ConvKernel_Convertion.iCSZ(DeCo, L0_KDeCo, L1_KDeCo)
        KDeCo = KDeCo / np.sum(KDeCo)     # NOTE Rescale to be with Unit Kernel-Sum

        return KDeCo

