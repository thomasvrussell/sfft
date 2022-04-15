import os
import sys
import time
import numpy as np

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class ElementalSFFTSubtract_Numpy:
    @staticmethod
    def ESSN(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, NUM_CPU_THREADS=8):

        import pyfftw
        pyfftw.config.NUM_THREADS = NUM_CPU_THREADS
        pyfftw.interfaces.cache.enable()
        
        ta = time.time()
        # * Read SFFT parameters
        print('\n  --||--||--||--||-- TRIGGER SFFT SUBTRACTION --||--||--||--||-- ')
        print('\n  ---||--- KerPolyOrder %d | BGPolyOrder %d | KerHW [%d] ---||--- '\
            %(SFFTConfig[0]['DK'], SFFTConfig[0]['DB'], SFFTConfig[0]['w0']))

        SFFTParam_dict, SFFTModule_dict = SFFTConfig
        N0, N1 = SFFTParam_dict['N0'], SFFTParam_dict['N1']
        w0, w1 = SFFTParam_dict['w0'], SFFTParam_dict['w1']
        DK, DB = SFFTParam_dict['DK'], SFFTParam_dict['DB']

        if PixA_I.shape != (N0, N1) or PixA_J.shape != (N0, N1):
            sys.exit('MeLOn ERROR: Inconsistent Shape of Input Images I & J, required [%d, %d] !' %(N0, N1))

        ConstPhotRatio = SFFTParam_dict['ConstPhotRatio']
        MaxThreadPerB = SFFTParam_dict['MaxThreadPerB']

        L0, L1 = SFFTParam_dict['L0'], SFFTParam_dict['L1']
        Fab, Fij, Fpq = SFFTParam_dict['Fab'], SFFTParam_dict['Fij'], SFFTParam_dict['Fpq']
        SCALE, SCALE_L = SFFTParam_dict['SCALE'], SFFTParam_dict['SCALE_L']

        NEQ, Fijab = SFFTParam_dict['NEQ'], SFFTParam_dict['Fijab']
        NEQ_FSfree = SFFTParam_dict['NEQ_FSfree']

        FOMG, FGAM = SFFTParam_dict['FOMG'], SFFTParam_dict['FGAM']
        FTHE, FPSI = SFFTParam_dict['FTHE'], SFFTParam_dict['FPSI']
        FPHI, FDEL = SFFTParam_dict['FPHI'], SFFTParam_dict['FDEL']

        # * Define First-order MultiIndex Reference
        #REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(np.int32)
        #REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        REF_ij = np.array([(i, j) for i in range(6) for j in range(6)]).astype(np.int32)
        REF_ab = np.array([(a_pos-w0, b_pos-w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(np.int32)

        # * Define Second-order MultiIndex Reference
        SREF_iji0j0 = np.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)]).astype(np.int32)
        SREF_pqp0q0 = np.array([(pq, p0q0) for pq in range(Fpq) for p0q0 in range(Fpq)]).astype(np.int32)
        SREF_ijpq = np.array([(ij, pq) for ij in range(Fij) for pq in range(Fpq)]).astype(np.int32)
        SREF_pqij = np.array([(pq, ij) for pq in range(Fpq) for ij in range(Fij)]).astype(np.int32)
        SREF_ijab = np.array([(ij, ab) for ij in range(Fij) for ab in range(Fab)]).astype(np.int32)

        # * Define the Forbidden ij-ab Rule
        ij00 = np.arange(w0 * L1 + w1, Fijab, Fab).astype(np.int32)
        if ConstPhotRatio:
            FBijab = ij00[1:]
            MASK_nFS = np.ones(NEQ).astype(bool)
            MASK_nFS[FBijab] = False            
            IDX_nFS = np.where(MASK_nFS)[0].astype(np.int32)
            NEQ_FSfree = len(IDX_nFS)

        t0 = time.time()
        # * Read input images as C-order arrays
        if not PixA_I.flags['C_CONTIGUOUS']:
            PixA_I = np.ascontiguousarray(PixA_I, np.float64)
        else: PixA_I = PixA_I.astype(np.float64)

        if not PixA_J.flags['C_CONTIGUOUS']:
            PixA_J = np.ascontiguousarray(PixA_J, np.float64)
        else: PixA_J = PixA_J.astype(np.float64)
        dt0 = time.time() - t0

        # * Symbol Convention NOTE
        #    X (x) / Y (y) ----- pixel row / column index
        #    CX (cx) / CY (cy) ----- ScaledFortranCoor of pixel (x, y) center   
        #    e.g. pixel (x, y) = (3, 5) corresponds (cx, cy) = (4.0/N0, 6.0/N1)
        #    NOTE cx / cy is literally \mathtt{x} / \mathtt{y} in sfft paper.
        #    NOTE Without special definition, MeLOn convention refers to X (x) / Y (y) as FortranCoor.

        # * Get Spatial Coordinates
        t1 = time.time()
        PixA_X = np.zeros((N0, N1), dtype=np.int32)      # row index, [0, N0)
        PixA_Y = np.zeros((N0, N1), dtype=np.int32)      # column index, [0, N1)
        PixA_CX = np.zeros((N0, N1), dtype=np.float64)   # coordinate.x
        PixA_CY = np.zeros((N0, N1), dtype=np.float64)   # coordinate.y 

        _func = SFFTModule_dict['SpatialCoor']
        _func(PixA_X=PixA_X, PixA_Y=PixA_Y, PixA_CX=PixA_CX, PixA_CY=PixA_CY)

        # * Spatial Polynomial terms Tij, Iij, Tpq
        SPixA_Tij = np.zeros((Fij, N0, N1), dtype=np.float64)
        SPixA_Iij = np.zeros((Fij, N0, N1), dtype=np.float64)
        SPixA_Tpq = np.zeros((Fpq, N0, N1), dtype=np.float64)

        _func = SFFTModule_dict['SpatialPoly']
        _func(REF_ij=REF_ij, REF_pq=REF_pq, PixA_CX=PixA_CX, PixA_CY=PixA_CY, PixA_I=PixA_I, \
            SPixA_Tij=SPixA_Tij, SPixA_Iij=SPixA_Iij, SPixA_Tpq=SPixA_Tpq)
        dt1 = time.time() - t1

        t2 = time.time()
        Batchsize = 1 + Fij + Fpq
        _SPixA_FFT = np.empty((Batchsize, N0, N1), dtype=np.complex128)
        _SPixA_FFT[0, :, :] = PixA_J.astype(np.complex128)
        _SPixA_FFT[1: Fij+1, :, :] = SPixA_Iij.astype(np.complex128)
        _SPixA_FFT[Fij+1:, :, :] = SPixA_Tpq.astype(np.complex128)

        #_SPixA_FFT = SCALE * np.fft.fft2(_SPixA_FFT)
        _SPixA_FFT = SCALE * pyfftw.interfaces.numpy_fft.fft2(_SPixA_FFT)

        PixA_FJ = _SPixA_FFT[0, :, :]
        SPixA_FIij = _SPixA_FFT[1: Fij+1, :, :]
        SPixA_FTpq = _SPixA_FFT[Fij+1:, :, :]

        PixA_CFJ = np.conj(PixA_FJ)
        SPixA_CFIij = np.conj(SPixA_FIij)
        SPixA_CFTpq = np.conj(SPixA_FTpq)
        dt2 = time.time() - t2
        dta = time.time() - ta

        print('\nMeLOn CheckPoint: Priminary Time     [%.4fs]' %dta)
        print('/////   a   ///// Read Input Images  (%.4fs)' %dt0)
        print('/////   b   ///// Spatial Polynomial (%.4fs)' %dt1)
        print('/////   c   ///// DFT-%d             (%.4fs)' %(1 + Fij + Fpq, dt2))

        # * Consider The Major Sources of the Linear System 
        #     𝛀_i8j8ij     &    𝜦_i8j8pq    ||     𝚯_i8j8
        #     𝜳_p8q8ij     &    𝚽_p8q8pq    ||     𝚫_p8q8   
        #
        # * Remarks
        #    a. They have consistent form: Greek(rho, eps) = PreGreek(Mod_N0(rho), Mod_N1(eps))
        #    b. PreGreek = s * Re[DFT(HpGreek)], where HpGreek = GLH * GRH and s is some real-scale.
        #    c. Considering the subscripted variables, HpGreek / PreGreek is Complex / Real 3D with shape (F_Greek, N0, N1).

        if SFFTSolution is not None:
            Solution = SFFTSolution.astype(np.float64)
            a_ijab = Solution[: Fijab]
            b_pq = Solution[Fijab: ]
            
        if SFFTSolution is None:
            tb = time.time()    
            
            LHMAT = np.empty((NEQ, NEQ), dtype=np.float64)
            RHb = np.empty(NEQ, dtype=np.float64)
            t3 = time.time()
            
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝛀  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for 𝛀 [HpOMG]
            _func = SFFTModule_dict['HadProd_OMG']
            HpOMG = np.empty((FOMG, N0, N1), dtype=np.complex128)
            HpOMG = _func(SREF_iji0j0=SREF_iji0j0, SPixA_FIij=SPixA_FIij, SPixA_CFIij=SPixA_CFIij, HpOMG=HpOMG)
            
            # b. PreOMG = SCALE * Re[DFT(HpOMG)]
            HpOMG = SCALE * pyfftw.interfaces.numpy_fft.fft2(HpOMG)
            
            HpOMG = HpOMG.real
            HpOMG *= SCALE
            PreOMG = HpOMG
            
            # c. Fill Linear System with PreOMG
            _func = SFFTModule_dict['FillLS_OMG']
            LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreOMG=PreOMG, LHMAT=LHMAT)
            dt3 = time.time() - t3

            t4 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝜦  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for 𝜦 [HpGAM]
            _func = SFFTModule_dict['HadProd_GAM']
            HpGAM = np.empty((FGAM, N0, N1), dtype=np.complex128)
            HpGAM = _func(SREF_ijpq=SREF_ijpq, SPixA_FIij=SPixA_FIij, SPixA_CFTpq=SPixA_CFTpq, HpGAM=HpGAM)

            # b. PreGAM = 1 * Re[DFT(HpGAM)]
            HpGAM = SCALE * pyfftw.interfaces.numpy_fft.fft2(HpGAM)
            
            HpGAM = HpGAM.real 
            PreGAM = HpGAM

            # c. Fill Linear System with PreGAM
            _func = SFFTModule_dict['FillLS_GAM']
            LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreGAM=PreGAM, LHMAT=LHMAT)
            dt4 = time.time() - t4

            t5 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝜳  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for 𝜳  [HpPSI]
            _func = SFFTModule_dict['HadProd_PSI']
            HpPSI = np.empty((FPSI, N0, N1), dtype=np.complex128)
            HpPSI = _func(SREF_pqij=SREF_pqij, SPixA_CFIij=SPixA_CFIij, SPixA_FTpq=SPixA_FTpq, HpPSI=HpPSI)
            
            # b. PrePSI = 1 * Re[DFT(HpPSI)]
            HpPSI = SCALE * pyfftw.interfaces.numpy_fft.fft2(HpPSI)

            HpPSI = HpPSI.real
            PrePSI = HpPSI

            # c. Fill Linear System with PrePSI
            _func = SFFTModule_dict['FillLS_PSI']
            LHMAT = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PrePSI=PrePSI, LHMAT=LHMAT)
            dt5 = time.time() - t5

            t6 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝚽  -- -- -- -- -- -- -- -- *
            
            # a. Hadamard Product for 𝚽  [HpPHI]
            _func = SFFTModule_dict['HadProd_PHI']
            HpPHI = np.empty((FPHI, N0, N1), dtype=np.complex128)
            HpPHI = _func(SREF_pqp0q0=SREF_pqp0q0, SPixA_FTpq=SPixA_FTpq, SPixA_CFTpq=SPixA_CFTpq, HpPHI=HpPHI)

            # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
            HpPHI = SCALE * pyfftw.interfaces.numpy_fft.fft2(HpPHI)
            
            HpPHI = HpPHI.real
            HpPHI *= SCALE_L
            PrePHI = HpPHI

            # c. Fill Linear System with PrePHI
            _func = SFFTModule_dict['FillLS_PHI']
            LHMAT = _func(PrePHI=PrePHI, LHMAT=LHMAT)

            dt6 = time.time() - t6
            
            t7 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝚯 & 𝚫  -- -- -- -- -- -- -- -- *

            # a1. Hadamard Product for 𝚯  [HpTHE]
            _func = SFFTModule_dict['HadProd_THE']
            HpTHE = np.empty((FTHE, N0, N1), dtype=np.complex128)
            HpTHE = _func(SPixA_FIij=SPixA_FIij, PixA_CFJ=PixA_CFJ, HpTHE=HpTHE)

            # a2. Hadamard Product for 𝚫  [HpDEL]
            _func = SFFTModule_dict['HadProd_DEL']
            HpDEL = np.empty((FDEL, N0, N1), dtype=np.complex128)
            HpDEL = _func(SPixA_FTpq=SPixA_FTpq, PixA_CFJ=PixA_CFJ, HpDEL=HpDEL)
            
            # b1. PreTHE = 1 * Re[DFT(HpTHE)]
            # b2. PreDEL = SCALE_L * Re[DFT(HpDEL)]
            HpTHE = SCALE * pyfftw.interfaces.numpy_fft.fft2(HpTHE)
            HpDEL = SCALE * pyfftw.interfaces.numpy_fft.fft2(HpDEL)
            
            HpTHE = HpTHE.real
            PreTHE = HpTHE

            HpDEL = HpDEL.real
            HpDEL *= SCALE_L
            PreDEL = HpDEL
            
            # c1. Fill Linear System with PreTHE
            _func = SFFTModule_dict['FillLS_THE']
            RHb = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, PreTHE=PreTHE, RHb=RHb)

            # c2. Fill Linear System with PreDEL
            _func = SFFTModule_dict['FillLS_DEL']
            RHb = _func(PreDEL=PreDEL, RHb=RHb)
            dt7 = time.time() - t7
            
            t8 = time.time()
            # * -- -- -- -- -- -- -- -- Remove Forbidden Stripes  -- -- -- -- -- -- -- -- *
            if not ConstPhotRatio: pass
            if ConstPhotRatio:
                RHb_FSfree = RHb[IDX_nFS]
                _func = SFFTModule_dict['Remove_LSFStripes']
                LHMAT_FSfree = np.empty((NEQ_FSfree, NEQ_FSfree), dtype=np.float64)
                LHMAT_FSfree = _func(LHMAT=LHMAT, IDX_nFS=IDX_nFS, LHMAT_FSfree=LHMAT_FSfree)

            t8 = time.time()
            # * -- -- -- -- -- -- -- -- Solve Linear System  -- -- -- -- -- -- -- -- *
            if not ConstPhotRatio:
                Solution = np.linalg.solve(LHMAT, RHb).astype(np.float64)

            if ConstPhotRatio:
                Solution_FSfree = np.linalg.solve(LHMAT_FSfree, RHb_FSfree).astype(np.float64)

                # Extend the solution to be consistent form
                _func = SFFTModule_dict['Extend_Solution']
                Solution = np.zeros(NEQ, dtype=np.float64)
                Solution = _func(Solution_FSfree=Solution_FSfree, IDX_nFS=IDX_nFS, Solution=Solution)

            a_ijab = Solution[: Fijab]
            b_pq = Solution[Fijab: ]

            dt8 = time.time() - t8
            dtb = time.time() - tb

            print('\nMeLOn CheckPoint: Establish & Solve Linear System     [%.4fs]' %dtb)
            print('/////   d   ///// Establish OMG                       (%.4fs)' %dt3)
            print('/////   e   ///// Establish GAM                       (%.4fs)' %dt4)
            print('/////   f   ///// Establish PSI                       (%.4fs)' %dt5)
            print('/////   g   ///// Establish PHI                       (%.4fs)' %dt6)
            print('/////   h   ///// Establish THE & DEL                 (%.4fs)' %dt7)
            print('/////   i   ///// Solve                               (%.4fs)' %dt8)
        
        # * Perform Subtraction
        PixA_DIFF = None
        if Subtract:

            tc = time.time()
            t9 = time.time()
            # Calculate Kab components
            Wl = np.exp((-2j*np.pi/N0) * PixA_X.astype(np.float64))    # row index l, [0, N0)
            Wm = np.exp((-2j*np.pi/N1) * PixA_Y.astype(np.float64))    # column index m, [0, N1)
            Kab_Wla = np.empty((L0, N0, N1), dtype=np.complex128)
            Kab_Wmb = np.empty((L1, N0, N1), dtype=np.complex128)

            if w0 == w1:
                wx = w0   # a little bit faster
                for aob in range(-wx, wx+1):
                    Kab_Wla[aob + wx] = Wl ** aob    # offset 
                    Kab_Wmb[aob + wx] = Wm ** aob    # offset 
            else:
                for a in range(-w0, w0+1): 
                    Kab_Wla[a + w0] = Wl ** a      # offset 
                for b in range(-w1, w1+1): 
                    Kab_Wmb[b + w1] = Wm ** b      # offset 
            dt9 = time.time() - t9

            t10 = time.time()
            # Construct Difference in Fourier Space
            _func = SFFTModule_dict['Construct_FDIFF']   
            PixA_FDIFF = np.empty((N0, N1), dtype=np.complex128)
            PixA_FDIFF = _func(SREF_ijab=SREF_ijab, REF_ab=REF_ab, a_ijab=a_ijab.astype(np.complex128), \
                SPixA_FIij=SPixA_FIij, Kab_Wla=Kab_Wla, Kab_Wmb=Kab_Wmb, b_pq=b_pq.astype(np.complex128), \
                SPixA_FTpq=SPixA_FTpq, PixA_FJ=PixA_FJ, PixA_FDIFF=PixA_FDIFF)

            # Get Difference & Reconstructed Images
            PixA_DIFF = np.empty_like(PixA_FDIFF)
            PixA_DIFF[:, :] = SCALE_L * pyfftw.interfaces.numpy_fft.ifft2(PixA_FDIFF)
            
            PixA_DIFF = PixA_DIFF.real
            dt10 = time.time() - t10
            dtc = time.time() - tc

            print('\nMeLOn CheckPoint: Perform Subtraction     [%.4fs]' %dtc)
            print('/////   j   ///// Calculate Kab           (%.4fs)' %dt9)
            print('/////   k   ///// Construct DIFF        (%.4fs)' %dt10)
        
        print('\n  --||--||--||--||-- EXIT SFFT SUBTRACTION --||--||--||--||-- ')

        return Solution, PixA_DIFF

class ElementalSFFTSubtract:
    @staticmethod
    def ESS(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8):
        
        if backend == 'Numpy':
            Solution, PixA_DIFF = ElementalSFFTSubtract_Numpy.ESSN(PixA_I=PixA_I, PixA_J=PixA_J, SFFTConfig=SFFTConfig, SFFTSolution=SFFTSolution, Subtract=Subtract, NUM_CPU_THREADS=NUM_CPU_THREADS)
        
        return Solution, PixA_DIFF

class GeneralSFFTSubtract:
    @staticmethod
    def GSS(PixA_I, PixA_J, PixA_mI, PixA_mJ, SFFTConfig, ContamMask_I=None, backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8):

        """
        # Arguments:
        # a) PixA_I: Image I that will be convolved [NaN-Free]
        # b) PixA_J: Image J that won't be convolved [NaN-Free]
        # c) PixA_mI: Masked version of Image I. 'same' means it is identical with I [NaN-Free]
        # d) PixA_mJ: Masked version of Image J. 'same' means it is identical with J [NaN-Free]
        # e) SFFTConfig: Configuration of SFFT
        # f) ContamMask_I: Contaminate-Region in Image I (e.g., Saturation and Bad pixels).
        #               We will calculate the propagated Contaminate-Region on convolved I.
        # 
        # g) backend: Which backend would you like to perform SFFT on ?
        # h) CUDA_DEVICE (backend = Pycuda / Cupy): Which GPU device would you want to perform SFFT on ?
        # i) NUM_CPU_THREADS (backend = Numpy): How many CPU threads would you want to perform SFFT on ?
        #
        # NOTE: The SFFT parameter-solving is performed between mI & mJ, 
        #       then we apply the solution to input Images I & J.
        #
        """
        
        # * Size-Check Processes
        tmplst = [PixA_I.shape, PixA_J.shape]
        tmplst += [PixA_mI.shape, PixA_mI.shape]
        if len(set(tmplst)) > 1:
            sys.exit('MeLOn ERROR: Input images should have same size !')

        # * Subtraction Solution derived from input masked image-pair
        Solution = ElementalSFFTSubtract.ESS(PixA_I=PixA_mI, PixA_J=PixA_mJ, \
            SFFTConfig=SFFTConfig, SFFTSolution=None, Subtract=False, backend=backend, \
            CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)[0]
        
        # * Subtraction of the input image-pair (use above solution)
        PixA_DIFF = ElementalSFFTSubtract.ESS(PixA_I=PixA_I, PixA_J=PixA_J, \
            SFFTConfig=SFFTConfig, SFFTSolution=Solution, Subtract=True, backend=backend, \
            CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)[1]

        # * Identify propagated contamination region through convolving I
        ContamMask_CI = None
        if ContamMask_I is not None:
            tSolution = Solution.copy()
            DB = SFFTConfig[0]['DB']
            Fpq = int((DB+1)*(DB+2)/2)
            tSolution[-Fpq:] = 0.0
            _tmpI = ContamMask_I.astype(np.float64)
            _tmpJ = np.zeros(PixA_J.shape).astype(np.float64)

            # NOTE Convolved_I is inverse DIFF
            _tmpD = ElementalSFFTSubtract.ESS(PixA_I=_tmpI, PixA_J=_tmpJ, \
                SFFTConfig=SFFTConfig, SFFTSolution=tSolution, Subtract=True, backend=backend, \
                CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)[1]
            ContamMask_CI = _tmpD < -0.001     # FIXME Customizable
        
        return Solution, PixA_DIFF, ContamMask_CI
