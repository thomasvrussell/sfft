import os
import sys
import time
import numpy as np

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class ElementalSFFTSubtract_Pycuda:
    @staticmethod
    def ESSP(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, CUDA_DEVICE='0'):

        import warnings
        import pycuda.autoinit
        from pycuda.cumath import exp
        import pycuda.gpuarray as gpuarray
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import skcuda.fft as cu_fft
            import skcuda.linalg as sklinalg
            import skcuda.cusolver as sksolver
        os.environ["CUDA_DEVICE"] = CUDA_DEVICE

        def LSSolver(LHMAT_GPU, RHb_GPU):

            H, W = LHMAT_GPU.shape
            A_GPU = LHMAT_GPU.T.copy().astype(np.float64)
            b_GPU = RHb_GPU.copy().astype(np.float64)

            handle = sksolver.cusolverDnCreate()
            Lwork = sksolver.cusolverDnDgetrf_bufferSize(handle, H, W, A_GPU.gpudata, H)
            workspace_gpu = gpuarray.zeros(Lwork, dtype=np.float64)

            devipiv_GPU = gpuarray.zeros(min(A_GPU.shape), dtype=np.int64)
            devinfo_GPU = gpuarray.zeros(1, dtype=np.int64)

            # * LU decomposition 
            sksolver.cusolverDnDgetrf(handle=handle, m=H, n=W, \
                a=A_GPU.gpudata, lda=H, workspace=workspace_gpu.gpudata, \
                devIpiv=devipiv_GPU.gpudata, devInfo=devinfo_GPU.gpudata)

            # * cusolver-based (~200 times faster than np.linalg.solve)
            sksolver.cusolverDnDgetrs(handle=handle, trans="n", n=H, nrhs=1, a=A_GPU.gpudata, lda=H, \
                devIpiv=devipiv_GPU.gpudata, B=b_GPU.gpudata, ldb=H, devInfo=devinfo_GPU.gpudata)
            sksolver.cusolverDnDestroy(handle)
            Solution_GPU = b_GPU

            return Solution_GPU

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
        
        # * Grid-Block-Thread Managaement [Pixel-Level]
        GPUManage = lambda NT: ((NT-1)//MaxThreadPerB + 1, min(NT, MaxThreadPerB))
        BpG_PIX0, TpB_PIX0 = GPUManage(N0)
        BpG_PIX1, TpB_PIX1 = GPUManage(N1)
        BpG_PIX, TpB_PIX = (BpG_PIX0, BpG_PIX1), (TpB_PIX0, TpB_PIX1, 1)

        # * Define First-order MultiIndex Reference
        REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(np.int32)
        REF_ab = np.array([(a_pos-w0, b_pos-w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(np.int32)
        REF_pq_GPU = gpuarray.to_gpu(REF_pq)
        REF_ij_GPU = gpuarray.to_gpu(REF_ij)
        REF_ab_GPU = gpuarray.to_gpu(REF_ab)

        # * Define Second-order MultiIndex Reference
        SREF_iji0j0 = np.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)]).astype(np.int32)
        SREF_pqp0q0 = np.array([(pq, p0q0) for pq in range(Fpq) for p0q0 in range(Fpq)]).astype(np.int32)
        SREF_ijpq = np.array([(ij, pq) for ij in range(Fij) for pq in range(Fpq)]).astype(np.int32)
        SREF_pqij = np.array([(pq, ij) for pq in range(Fpq) for ij in range(Fij)]).astype(np.int32)
        SREF_ijab = np.array([(ij, ab) for ij in range(Fij) for ab in range(Fab)]).astype(np.int32)

        SREF_iji0j0_GPU = gpuarray.to_gpu(SREF_iji0j0)
        SREF_pqp0q0_GPU = gpuarray.to_gpu(SREF_pqp0q0)
        SREF_ijpq_GPU = gpuarray.to_gpu(SREF_ijpq)
        SREF_pqij_GPU = gpuarray.to_gpu(SREF_pqij)
        SREF_ijab_GPU = gpuarray.to_gpu(SREF_ijab)

        # * Define the Forbidden ij-ab Rule
        ij00 = np.arange(w0 * L1 + w1, Fijab, Fab).astype(np.int32)
        if ConstPhotRatio:
            FBijab = ij00[1:]
            MASK_nFS = np.ones(NEQ).astype(bool)
            MASK_nFS[FBijab] = False            
            IDX_nFS = np.where(MASK_nFS)[0].astype(np.int32)
            IDX_nFS_GPU = gpuarray.to_gpu(IDX_nFS)
            NEQ_FSfree = len(IDX_nFS)

        t0 = time.time()
        # * Read input images as C-order arrays
        if not PixA_I.flags['C_CONTIGUOUS']:
            PixA_I = np.ascontiguousarray(PixA_I, np.float64)
            PixA_I_GPU = gpuarray.to_gpu(PixA_I)
        else: PixA_I_GPU = gpuarray.to_gpu(PixA_I.astype(np.float64))
        
        if not PixA_J.flags['C_CONTIGUOUS']:
            PixA_J = np.ascontiguousarray(PixA_J, np.float64)
            PixA_J_GPU = gpuarray.to_gpu(PixA_J)
        else: PixA_J_GPU = gpuarray.to_gpu(PixA_J.astype(np.float64))
        dt0 = time.time() - t0

        # * Symbol Convention NOTE
        #    X (x) / Y (y) ----- pixel row / column index
        #    CX (cx) / CY (cy) ----- ScaledFortranCoor of pixel (x, y) center   
        #    e.g. pixel (x, y) = (3, 5) corresponds (cx, cy) = (4.0/N0, 6.0/N1)
        #    NOTE cx / cy is literally \mathtt{x} / \mathtt{y} in sfft paper.
        #    NOTE Without special definition, MeLOn convention refers to X (x) / Y (y) as FortranCoor.

        # * Get Spatial Coordinates
        t1 = time.time()
        PixA_X_GPU = gpuarray.zeros((N0, N1), dtype=np.int32)      # row index, [0, N0)
        PixA_Y_GPU = gpuarray.zeros((N0, N1), dtype=np.int32)      # column index, [0, N1)
        PixA_CX_GPU = gpuarray.zeros((N0, N1), dtype=np.float64)   # coordinate.x
        PixA_CY_GPU = gpuarray.zeros((N0, N1), dtype=np.float64)   # coordinate.y 

        _module = SFFTModule_dict['SpatialCoor']
        _func = _module.get_function('kmain')
        _func(PixA_X_GPU, PixA_Y_GPU, PixA_CX_GPU, PixA_CY_GPU, block=TpB_PIX, grid=BpG_PIX)

        # * Spatial Polynomial terms Tij, Iij, Tpq
        SPixA_Tij_GPU = gpuarray.zeros((Fij, N0, N1), dtype=np.float64)
        SPixA_Iij_GPU = gpuarray.zeros((Fij, N0, N1), dtype=np.float64)
        SPixA_Tpq_GPU = gpuarray.zeros((Fpq, N0, N1), dtype=np.float64)

        _module = SFFTModule_dict['SpatialPoly']
        _func = _module.get_function('kmain')
        _func(REF_ij_GPU, REF_pq_GPU, PixA_CX_GPU, PixA_CY_GPU, PixA_I_GPU, \
            SPixA_Tij_GPU, SPixA_Iij_GPU, SPixA_Tpq_GPU, block=TpB_PIX, grid=BpG_PIX)
        PixA_I_GPU.gpudata.free()
        dt1 = time.time() - t1

        t2 = time.time()
        # * Empirical Trick on FFT-Plan
        EmpFFTPlan = False
        plan_once = cu_fft.Plan((N0, N1), np.complex128, np.complex128)
        if DK == 2 and DB == 2:
            EmpFFTPlan = True
            plan_12 = cu_fft.Plan((N0, N1), np.complex128, np.complex128, 12)

        # * Make DFT of J, Iij, Tpq and their conjugates
        if EmpFFTPlan:
            PixA_FJ_GPU = PixA_J_GPU.astype(np.complex128)
            cu_fft.fft(PixA_FJ_GPU, PixA_FJ_GPU, plan_once, scale=True)

            _SPixA_FFT_GPU = gpuarray.empty((12, N0, N1), dtype=np.complex128)
            _SPixA_FFT_GPU[:6, :] = SPixA_Iij_GPU.astype(np.complex128)
            _SPixA_FFT_GPU[6:, :] = SPixA_Tpq_GPU.astype(np.complex128)

            cu_fft.fft(_SPixA_FFT_GPU, _SPixA_FFT_GPU, plan_12, scale=True)
            SPixA_FIij_GPU = _SPixA_FFT_GPU[:6, :]
            SPixA_FTpq_GPU = _SPixA_FFT_GPU[6:, :]
            
        if not EmpFFTPlan:
            Batchsize = 1 + Fij + Fpq
            _SPixA_FFT_GPU = gpuarray.empty((Batchsize, N0, N1), dtype=np.complex128)
            _SPixA_FFT_GPU[0, :, :] = PixA_J_GPU.astype(np.complex128)
            _SPixA_FFT_GPU[1: Fij+1, :, :] = SPixA_Iij_GPU.astype(np.complex128)
            _SPixA_FFT_GPU[Fij+1:, :, :] = SPixA_Tpq_GPU.astype(np.complex128)

            plan = cu_fft.Plan((N0, N1), np.complex128, np.complex128, Batchsize)
            cu_fft.fft(_SPixA_FFT_GPU, _SPixA_FFT_GPU, plan, scale=True)
            PixA_FJ_GPU = _SPixA_FFT_GPU[0, :, :]
            SPixA_FIij_GPU = _SPixA_FFT_GPU[1: Fij+1, :, :]
            SPixA_FTpq_GPU = _SPixA_FFT_GPU[Fij+1:, :, :]

        SPixA_Iij_GPU.gpudata.free()
        SPixA_Tpq_GPU.gpudata.free()

        PixA_CFJ_GPU = sklinalg.conj(PixA_FJ_GPU)
        SPixA_CFIij_GPU = sklinalg.conj(SPixA_FIij_GPU)
        SPixA_CFTpq_GPU = sklinalg.conj(SPixA_FTpq_GPU)
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
            Solution = SFFTSolution 
            Solution_GPU = gpuarray.to_gpu(Solution.astype(np.float64))
            a_ijab_GPU = Solution_GPU[: Fijab]
            b_pq_GPU = Solution_GPU[Fijab: ]

        if SFFTSolution is None:
            tb = time.time()
            # * Grid-Block-Thread Managaement [for Greek]
            BpG_OMG0, TpB_OMG0 = GPUManage(Fijab)
            BpG_OMG1, TpB_OMG1 = GPUManage(Fijab)
            BpG_OMG, TpB_OMG = (BpG_OMG0, BpG_OMG1), (TpB_OMG0, TpB_OMG1, 1)

            BpG_GAM0, TpB_GAM0 = GPUManage(Fijab)
            BpG_GAM1, TpB_GAM1 = GPUManage(Fpq)
            BpG_GAM, TpB_GAM = (BpG_GAM0, BpG_GAM1), (TpB_GAM0, TpB_GAM1, 1)

            BpG_THE0, TpB_THE0 = GPUManage(Fijab)
            BpG_THE, TpB_THE = (BpG_THE0, 1), (TpB_THE0, 1, 1)

            BpG_PSI0, TpB_PSI0 = GPUManage(Fpq)
            BpG_PSI1, TpB_PSI1 = GPUManage(Fijab)
            BpG_PSI, TpB_PSI = (BpG_PSI0, BpG_PSI1), (TpB_PSI0, TpB_PSI1, 1)

            BpG_PHI0, TpB_PHI0 = GPUManage(Fpq)
            BpG_PHI1, TpB_PHI1 = GPUManage(Fpq)
            BpG_PHI, TpB_PHI = (BpG_PHI0, BpG_PHI1), (TpB_PHI0, TpB_PHI1, 1)

            BpG_DEL0, TpB_DEL0 = GPUManage(Fpq)
            BpG_DEL, TpB_DEL = (BpG_DEL0, 1), (TpB_DEL0, 1, 1)

            LHMAT_GPU = gpuarray.empty((NEQ, NEQ), dtype=np.float64)
            RHb_GPU = gpuarray.empty(NEQ, dtype=np.float64)
            t3 = time.time()

            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝛀  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for 𝛀 [HpOMG]
            _module = SFFTModule_dict['HadProd_OMG']
            _func = _module.get_function('kmain')
            HpOMG_GPU = gpuarray.empty((FOMG, N0, N1), dtype=np.complex128)
            _func(SREF_iji0j0_GPU, SPixA_FIij_GPU, SPixA_CFIij_GPU, HpOMG_GPU, \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PreOMG = SCALE * Re[DFT(HpOMG)]
            if EmpFFTPlan:
                cu_fft.fft(HpOMG_GPU[:12], HpOMG_GPU[:12], plan_12, scale=True)
                cu_fft.fft(HpOMG_GPU[12: 24], HpOMG_GPU[12: 24], plan_12, scale=True)
                cu_fft.fft(HpOMG_GPU[24: 36], HpOMG_GPU[24: 36], plan_12, scale=True)

            if not EmpFFTPlan:
                plan = cu_fft.Plan((N0, N1), np.complex128, np.complex128, FOMG)
                cu_fft.fft(HpOMG_GPU, HpOMG_GPU, plan, scale=True)

            HpOMG_GPU = HpOMG_GPU.real 
            HpOMG_GPU *= SCALE 
            PreOMG_GPU = HpOMG_GPU

            # c. Fill Linear System with PreOMG
            _module = SFFTModule_dict['FillLS_OMG']
            _func = _module.get_function('kmain')
            _func(SREF_ijab_GPU, REF_ab_GPU, PreOMG_GPU, LHMAT_GPU, \
                block=TpB_OMG, grid=BpG_OMG)

            PreOMG_GPU.gpudata.free()
            dt3 = time.time() - t3

            t4 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝜦  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for 𝜦 [HpGAM]
            _module = SFFTModule_dict['HadProd_GAM']
            _func = _module.get_function('kmain')
            HpGAM_GPU = gpuarray.empty((FGAM, N0, N1), dtype=np.complex128)
            _func(SREF_ijpq_GPU, SPixA_FIij_GPU, SPixA_CFTpq_GPU, HpGAM_GPU, \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PreGAM = 1 * Re[DFT(HpGAM)]
            if EmpFFTPlan:
                cu_fft.fft(HpGAM_GPU[:12], HpGAM_GPU[:12], plan_12, scale=True)
                cu_fft.fft(HpGAM_GPU[12: 24], HpGAM_GPU[12: 24], plan_12, scale=True)
                cu_fft.fft(HpGAM_GPU[24: 36], HpGAM_GPU[24: 36], plan_12, scale=True)

            if not EmpFFTPlan:
                plan = cu_fft.Plan((N0, N1), np.complex128, np.complex128, FGAM)
                cu_fft.fft(HpGAM_GPU, HpGAM_GPU, plan, scale=True)

            HpGAM_GPU = HpGAM_GPU.real 
            PreGAM_GPU = HpGAM_GPU

            # c. Fill Linear System with PreGAM
            _module = SFFTModule_dict['FillLS_GAM']
            _func = _module.get_function('kmain')
            _func(SREF_ijab_GPU, REF_ab_GPU, PreGAM_GPU, LHMAT_GPU, \
                block=TpB_GAM, grid=BpG_GAM)

            PreGAM_GPU.gpudata.free()
            dt4 = time.time() - t4

            t5 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝜳  -- -- -- -- -- -- -- -- *
            
            # a. Hadamard Product for 𝜳  [HpPSI]
            _module = SFFTModule_dict['HadProd_PSI']
            _func = _module.get_function('kmain')
            HpPSI_GPU = gpuarray.empty((FPSI, N0, N1), dtype=np.complex128)
            _func(SREF_pqij_GPU, SPixA_CFIij_GPU, SPixA_FTpq_GPU, HpPSI_GPU, \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PrePSI = 1 * Re[DFT(HpPSI)]
            if EmpFFTPlan:
                cu_fft.fft(HpPSI_GPU[:12], HpPSI_GPU[:12], plan_12, scale=True)
                cu_fft.fft(HpPSI_GPU[12: 24], HpPSI_GPU[12: 24], plan_12, scale=True)
                cu_fft.fft(HpPSI_GPU[24: 36], HpPSI_GPU[24: 36], plan_12, scale=True)

            if not EmpFFTPlan:
                plan = cu_fft.Plan((N0, N1), np.complex128, np.complex128, FPSI)
                cu_fft.fft(HpPSI_GPU, HpPSI_GPU, plan, scale=True)

            HpPSI_GPU = HpPSI_GPU.real
            PrePSI_GPU = HpPSI_GPU

            # c. Fill Linear System with PrePSI
            _module = SFFTModule_dict['FillLS_PSI']
            _func = _module.get_function('kmain')
            _func(SREF_ijab_GPU, REF_ab_GPU, PrePSI_GPU, LHMAT_GPU, \
                block=TpB_PSI, grid=BpG_PSI)

            PrePSI_GPU.gpudata.free()
            dt5 = time.time() - t5

            t6 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝚽  -- -- -- -- -- -- -- -- *
            
            
            # a. Hadamard Product for 𝚽  [HpPHI]
            _module = SFFTModule_dict['HadProd_PHI']
            _func = _module.get_function('kmain')
            HpPHI_GPU = gpuarray.empty((FPHI, N0, N1), dtype=np.complex128)
            _func(SREF_pqp0q0_GPU, SPixA_FTpq_GPU, SPixA_CFTpq_GPU, HpPHI_GPU, \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
            if EmpFFTPlan:
                cu_fft.fft(HpPHI_GPU[:12], HpPHI_GPU[:12], plan_12, scale=True)
                cu_fft.fft(HpPHI_GPU[12: 24], HpPHI_GPU[12: 24], plan_12, scale=True)
                cu_fft.fft(HpPHI_GPU[24: 36], HpPHI_GPU[24: 36], plan_12, scale=True)

            if not EmpFFTPlan:
                plan = cu_fft.Plan((N0, N1), np.complex128, np.complex128, FPHI)
                cu_fft.fft(HpPHI_GPU, HpPHI_GPU, plan, scale=True)

            HpPHI_GPU = HpPHI_GPU.real
            HpPHI_GPU *= SCALE_L
            PrePHI_GPU = HpPHI_GPU

            # c. Fill Linear System with PrePHI
            _module = SFFTModule_dict['FillLS_PHI']
            _func = _module.get_function('kmain')
            _func(PrePHI_GPU, LHMAT_GPU, block=TpB_PHI, grid=BpG_PHI)

            PrePHI_GPU.gpudata.free()
            dt6 = time.time() - t6

            t7 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝚯 & 𝚫  -- -- -- -- -- -- -- -- *

            # a1. Hadamard Product for 𝚯  [HpTHE]
            _module = SFFTModule_dict['HadProd_THE']
            _func = _module.get_function('kmain')
            HpTHE_GPU = gpuarray.empty((FTHE, N0, N1), dtype=np.complex128)
            _func(SPixA_FIij_GPU, PixA_CFJ_GPU, HpTHE_GPU, block=TpB_PIX, grid=BpG_PIX)

            # a2. Hadamard Product for 𝚫  [HpDEL]
            _module = SFFTModule_dict['HadProd_DEL']
            _func = _module.get_function('kmain')
            HpDEL_GPU = gpuarray.empty((FDEL, N0, N1), dtype=np.complex128)
            _func(SPixA_FTpq_GPU, PixA_CFJ_GPU, HpDEL_GPU, block=TpB_PIX, grid=BpG_PIX)

            # b1. PreTHE = 1 * Re[DFT(HpTHE)]
            # b2. PreDEL = SCALE_L * Re[DFT(HpDEL)]
            if EmpFFTPlan:
                SPixA_FFT_GPU = gpuarray.empty((12, N0, N1), dtype=np.complex128)
                SPixA_FFT_GPU[:6, :] = HpTHE_GPU
                SPixA_FFT_GPU[6:, :] = HpDEL_GPU
                HpTHE_GPU.gpudata.free()
                HpDEL_GPU.gpudata.free()
                
                cu_fft.fft(SPixA_FFT_GPU, SPixA_FFT_GPU, plan_12, scale=True)
                HpTHE_GPU = SPixA_FFT_GPU[:6, :]
                HpDEL_GPU = SPixA_FFT_GPU[6:, :]

            if not EmpFFTPlan:
                plan = cu_fft.Plan((N0, N1), np.complex128, np.complex128, FTHE)
                cu_fft.fft(HpTHE_GPU, HpTHE_GPU, plan, scale=True)

                plan = cu_fft.Plan((N0, N1), np.complex128, np.complex128, FDEL)
                cu_fft.fft(HpDEL_GPU, HpDEL_GPU, plan, scale=True)
            
            HpTHE_GPU = HpTHE_GPU.real
            PreTHE_GPU = HpTHE_GPU

            HpDEL_GPU = HpDEL_GPU.real
            HpDEL_GPU *= SCALE_L
            PreDEL_GPU = HpDEL_GPU

            # c1. Fill Linear System with PreTHE
            _module = SFFTModule_dict['FillLS_THE']
            _func = _module.get_function('kmain')
            _func(SREF_ijab_GPU, REF_ab_GPU, PreTHE_GPU, RHb_GPU, \
                block=TpB_THE, grid=BpG_THE)
            PreTHE_GPU.gpudata.free()

            # c2. Fill Linear System with PreDEL
            _module = SFFTModule_dict['FillLS_DEL']
            _func = _module.get_function('kmain')
            _func(PreDEL_GPU, RHb_GPU, block=TpB_DEL, grid=BpG_DEL)
            PreDEL_GPU.gpudata.free()
            dt7 = time.time() - t7

            # * -- -- -- -- -- -- -- -- Remove Forbidden Stripes  -- -- -- -- -- -- -- -- *
            if not ConstPhotRatio: pass
            if ConstPhotRatio:
                RHb_FSfree_GPU = gpuarray.take(RHb_GPU, IDX_nFS_GPU)
                _module = SFFTModule_dict['Remove_LSFStripes']
                _func = _module.get_function('kmain')
                BpG_FSfree_PA, TpB_FSfree_PA = GPUManage(NEQ_FSfree)     # Per Axis
                BpG_FSfree, TpB_FSfree = (BpG_FSfree_PA, BpG_FSfree_PA), (TpB_FSfree_PA, TpB_FSfree_PA, 1)
                LHMAT_FSfree_GPU = gpuarray.empty((NEQ_FSfree, NEQ_FSfree), dtype=np.float64)
                _func(LHMAT_GPU, IDX_nFS_GPU, LHMAT_FSfree_GPU, block=TpB_FSfree, grid=BpG_FSfree)

            t8 = time.time()
            # * -- -- -- -- -- -- -- -- Solve Linear System  -- -- -- -- -- -- -- -- *
            if not ConstPhotRatio: 
                Solution_GPU = LSSolver(LHMAT_GPU=LHMAT_GPU, RHb_GPU=RHb_GPU)

            if ConstPhotRatio:
                # Extend the solution to be consistent form
                Solution_FSfree_GPU = LSSolver(LHMAT_GPU=LHMAT_FSfree_GPU, RHb_GPU=RHb_FSfree_GPU)
                _module = SFFTModule_dict['Extend_Solution']
                _func = _module.get_function('kmain')
                BpG_ES, TpB_ES = (BpG_FSfree_PA, 1), (TpB_FSfree_PA, 1, 1)
                Solution_GPU = gpuarray.zeros(NEQ, dtype=np.float64)
                _func(Solution_FSfree_GPU, IDX_nFS_GPU, Solution_GPU, block=TpB_ES, grid=BpG_ES)
            
            a_ijab_GPU = Solution_GPU[: Fijab]
            b_pq_GPU = Solution_GPU[Fijab: ]
            Solution = Solution_GPU.get()
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
            Wl_GPU = exp((-2j*np.pi/N0) * PixA_X_GPU.astype(np.float64))    # row index l, [0, N0)
            Wm_GPU = exp((-2j*np.pi/N1) * PixA_Y_GPU.astype(np.float64))    # column index m, [0, N1)
            Kab_Wla_GPU = gpuarray.empty((L0, N0, N1), dtype=np.complex128)
            Kab_Wmb_GPU = gpuarray.empty((L1, N0, N1), dtype=np.complex128)

            if w0 == w1:
                wx = w0   # a little bit faster
                for aob in range(-wx, wx+1):
                    Kab_Wla_GPU[aob + wx] = Wl_GPU ** aob    # offset 
                    Kab_Wmb_GPU[aob + wx] = Wm_GPU ** aob    # offset 
            else:
                for a in range(-w0, w0+1): 
                    Kab_Wla_GPU[a + w0] = Wl_GPU ** a      # offset 
                for b in range(-w1, w1+1): 
                    Kab_Wmb_GPU[b + w1] = Wm_GPU ** b      # offset 
            dt9 = time.time() - t9

            t10 = time.time()
            # Construct Difference in Fourier Space
            _module = SFFTModule_dict['Construct_FDIFF']
            _func = _module.get_function('kmain')     
            PixA_FDIFF_GPU = gpuarray.empty((N0, N1), dtype=np.complex128)
            _func(SREF_ijab_GPU, REF_ab_GPU, a_ijab_GPU.astype(np.complex128), \
                SPixA_FIij_GPU, Kab_Wla_GPU, Kab_Wmb_GPU, b_pq_GPU.astype(np.complex128), \
                SPixA_FTpq_GPU, PixA_FJ_GPU, PixA_FDIFF_GPU, block=TpB_PIX, grid=BpG_PIX)
            
            # Get Difference & Reconstructed Images
            PixA_DIFF_GPU = gpuarray.empty_like(PixA_FDIFF_GPU)
            cu_fft.ifft(PixA_FDIFF_GPU, PixA_DIFF_GPU, plan_once, scale=False)
            PixA_DIFF_GPU = PixA_DIFF_GPU.real
            PixA_DIFF = PixA_DIFF_GPU.get()
            dt10 = time.time() - t10
            dtc = time.time() - tc

            print('\nMeLOn CheckPoint: Perform Subtraction     [%.4fs]' %dtc)
            print('/////   j   ///// Calculate Kab           (%.4fs)' %dt9)
            print('/////   k   ///// Construct DIFF        (%.4fs)' %dt10)
        
        print('\n  --||--||--||--||-- EXIT SFFT SUBTRACTION --||--||--||--||-- ')

        return Solution, PixA_DIFF

class ElementalSFFTSubtract_Cupy:
    @staticmethod
    def ESSC(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, CUDA_DEVICE='0'):

        import cupy as cp
        import cupyx.scipy.linalg as cpx_linalg
        os.environ["CUDA_DEVICE"] = CUDA_DEVICE

        def LSSolver(LHMAT_GPU, RHb_GPU):
            """
            # deprecated cupy version (moderately slow)
            Solution_GPU = cp.linalg.solve(LHMAT_GPU, RHb_GPU)
            # deprecated numpy version (very slow)
            Solution_GPU = cp.array(np.linalg.solve(cp.asnumpy(LHMAT_GPU), cp.asnumpy(RHb_GPU)), dtype=np.float64)
            """
            lu_piv_GPU = cpx_linalg.lu_factor(LHMAT_GPU, overwrite_a=False, check_finite=True)
            Solution_GPU = cpx_linalg.lu_solve(lu_piv_GPU, RHb_GPU)
            return Solution_GPU

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
        
        # * Grid-Block-Thread Managaement [Pixel-Level]
        GPUManage = lambda NT: ((NT-1)//MaxThreadPerB + 1, min(NT, MaxThreadPerB))
        BpG_PIX0, TpB_PIX0 = GPUManage(N0)
        BpG_PIX1, TpB_PIX1 = GPUManage(N1)
        BpG_PIX, TpB_PIX = (BpG_PIX0, BpG_PIX1), (TpB_PIX0, TpB_PIX1, 1)

        # * Define First-order MultiIndex Reference
        REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(np.int32)
        REF_ab = np.array([(a_pos-w0, b_pos-w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(np.int32)
        REF_pq_GPU = cp.array(REF_pq)
        REF_ij_GPU = cp.array(REF_ij)
        REF_ab_GPU = cp.array(REF_ab)

        # * Define Second-order MultiIndex Reference
        SREF_iji0j0 = np.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)]).astype(np.int32)
        SREF_pqp0q0 = np.array([(pq, p0q0) for pq in range(Fpq) for p0q0 in range(Fpq)]).astype(np.int32)
        SREF_ijpq = np.array([(ij, pq) for ij in range(Fij) for pq in range(Fpq)]).astype(np.int32)
        SREF_pqij = np.array([(pq, ij) for pq in range(Fpq) for ij in range(Fij)]).astype(np.int32)
        SREF_ijab = np.array([(ij, ab) for ij in range(Fij) for ab in range(Fab)]).astype(np.int32)

        SREF_iji0j0_GPU = cp.array(SREF_iji0j0)
        SREF_pqp0q0_GPU = cp.array(SREF_pqp0q0)
        SREF_ijpq_GPU = cp.array(SREF_ijpq)
        SREF_pqij_GPU = cp.array(SREF_pqij)
        SREF_ijab_GPU = cp.array(SREF_ijab)

        # * Define the Forbidden ij-ab Rule
        ij00 = np.arange(w0 * L1 + w1, Fijab, Fab).astype(np.int32)
        if ConstPhotRatio:
            FBijab = ij00[1:]
            MASK_nFS = np.ones(NEQ).astype(bool)
            MASK_nFS[FBijab] = False            
            IDX_nFS = np.where(MASK_nFS)[0].astype(np.int32)
            IDX_nFS_GPU = cp.array(IDX_nFS)
            NEQ_FSfree = len(IDX_nFS)

        t0 = time.time()
        # * Read input images as C-order arrays
        if not PixA_I.flags['C_CONTIGUOUS']:
            PixA_I = np.ascontiguousarray(PixA_I, np.float64)
            PixA_I_GPU = cp.array(PixA_I)
        else: PixA_I_GPU = cp.array(PixA_I.astype(np.float64))
        
        if not PixA_J.flags['C_CONTIGUOUS']:
            PixA_J = np.ascontiguousarray(PixA_J, np.float64)
            PixA_J_GPU = cp.array(PixA_J)
        else: PixA_J_GPU = cp.array(PixA_J.astype(np.float64))
        dt0 = time.time() - t0

        # * Symbol Convention NOTE
        #    X (x) / Y (y) ----- pixel row / column index
        #    CX (cx) / CY (cy) ----- ScaledFortranCoor of pixel (x, y) center   
        #    e.g. pixel (x, y) = (3, 5) corresponds (cx, cy) = (4.0/N0, 6.0/N1)
        #    NOTE cx / cy is literally \mathtt{x} / \mathtt{y} in sfft paper.
        #    NOTE Without special definition, MeLOn convention refers to X (x) / Y (y) as FortranCoor.

        # * Get Spatial Coordinates
        t1 = time.time()
        PixA_X_GPU = cp.zeros((N0, N1), dtype=np.int32)      # row index, [0, N0)
        PixA_Y_GPU = cp.zeros((N0, N1), dtype=np.int32)      # column index, [0, N1)
        PixA_CX_GPU = cp.zeros((N0, N1), dtype=np.float64)   # coordinate.x
        PixA_CY_GPU = cp.zeros((N0, N1), dtype=np.float64)   # coordinate.y 

        _module = SFFTModule_dict['SpatialCoor']
        _func = _module.get_function('kmain')
        _func(args=(PixA_X_GPU, PixA_Y_GPU, PixA_CX_GPU, PixA_CY_GPU), block=TpB_PIX, grid=BpG_PIX)

        # * Spatial Polynomial terms Tij, Iij, Tpq
        SPixA_Tij_GPU = cp.zeros((Fij, N0, N1), dtype=np.float64)
        SPixA_Iij_GPU = cp.zeros((Fij, N0, N1), dtype=np.float64)
        SPixA_Tpq_GPU = cp.zeros((Fpq, N0, N1), dtype=np.float64)

        _module = SFFTModule_dict['SpatialPoly']
        _func = _module.get_function('kmain')
        _func(args=(REF_ij_GPU, REF_pq_GPU, PixA_CX_GPU, PixA_CY_GPU, PixA_I_GPU, \
            SPixA_Tij_GPU, SPixA_Iij_GPU, SPixA_Tpq_GPU), block=TpB_PIX, grid=BpG_PIX)
        
        del PixA_I_GPU
        dt1 = time.time() - t1

        t2 = time.time()
        # * Empirical Trick on FFT-Plan
        EmpFFTPlan = False
        if DK == 2 and DB == 2:
            EmpFFTPlan = True

        """
        # may be alternative to perform FFT
        import cupyx as cpx
        plan = cpx.scipy.fftpack.get_fft_plan(x)
        y = cpx.scipy.fftpack.fftn(x, plan=plan)

        """

        # * Make DFT of J, Iij, Tpq and their conjugates
        if EmpFFTPlan:
            PixA_FJ_GPU = SCALE * cp.fft.fft2(PixA_J_GPU)
            _SPixA_FFT_GPU = cp.empty((12, N0, N1), dtype=np.complex128)
            _SPixA_FFT_GPU[:6, :] = SPixA_Iij_GPU.astype(np.complex128)
            _SPixA_FFT_GPU[6:, :] = SPixA_Tpq_GPU.astype(np.complex128)  
            _SPixA_FFT_GPU = SCALE * cp.fft.fft2(_SPixA_FFT_GPU)
            SPixA_FIij_GPU = _SPixA_FFT_GPU[:6, :]
            SPixA_FTpq_GPU = _SPixA_FFT_GPU[6:, :]

        if not EmpFFTPlan:
            Batchsize = 1 + Fij + Fpq
            _SPixA_FFT_GPU = cp.empty((Batchsize, N0, N1), dtype=np.complex128)
            _SPixA_FFT_GPU[0, :, :] = PixA_J_GPU.astype(np.complex128)
            _SPixA_FFT_GPU[1: Fij+1, :, :] = SPixA_Iij_GPU.astype(np.complex128)
            _SPixA_FFT_GPU[Fij+1:, :, :] = SPixA_Tpq_GPU.astype(np.complex128)
            
            _SPixA_FFT_GPU = SCALE * cp.fft.fft2(_SPixA_FFT_GPU)
            PixA_FJ_GPU = _SPixA_FFT_GPU[0, :, :]
            SPixA_FIij_GPU = _SPixA_FFT_GPU[1: Fij+1, :, :]
            SPixA_FTpq_GPU = _SPixA_FFT_GPU[Fij+1:, :, :]

        del SPixA_Iij_GPU
        del SPixA_Tpq_GPU

        PixA_CFJ_GPU = cp.conj(PixA_FJ_GPU)
        SPixA_CFIij_GPU = cp.conj(SPixA_FIij_GPU)
        SPixA_CFTpq_GPU = cp.conj(SPixA_FTpq_GPU)
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
            Solution = SFFTSolution 
            Solution_GPU = cp.array(Solution.astype(np.float64))
            a_ijab_GPU = Solution_GPU[: Fijab]
            b_pq_GPU = Solution_GPU[Fijab: ]

        if SFFTSolution is None:
            tb = time.time()
            # * Grid-Block-Thread Managaement [for Greek]
            BpG_OMG0, TpB_OMG0 = GPUManage(Fijab)
            BpG_OMG1, TpB_OMG1 = GPUManage(Fijab)
            BpG_OMG, TpB_OMG = (BpG_OMG0, BpG_OMG1), (TpB_OMG0, TpB_OMG1, 1)

            BpG_GAM0, TpB_GAM0 = GPUManage(Fijab)
            BpG_GAM1, TpB_GAM1 = GPUManage(Fpq)
            BpG_GAM, TpB_GAM = (BpG_GAM0, BpG_GAM1), (TpB_GAM0, TpB_GAM1, 1)

            BpG_THE0, TpB_THE0 = GPUManage(Fijab)
            BpG_THE, TpB_THE = (BpG_THE0, 1), (TpB_THE0, 1, 1)

            BpG_PSI0, TpB_PSI0 = GPUManage(Fpq)
            BpG_PSI1, TpB_PSI1 = GPUManage(Fijab)
            BpG_PSI, TpB_PSI = (BpG_PSI0, BpG_PSI1), (TpB_PSI0, TpB_PSI1, 1)

            BpG_PHI0, TpB_PHI0 = GPUManage(Fpq)
            BpG_PHI1, TpB_PHI1 = GPUManage(Fpq)
            BpG_PHI, TpB_PHI = (BpG_PHI0, BpG_PHI1), (TpB_PHI0, TpB_PHI1, 1)

            BpG_DEL0, TpB_DEL0 = GPUManage(Fpq)
            BpG_DEL, TpB_DEL = (BpG_DEL0, 1), (TpB_DEL0, 1, 1)

            LHMAT_GPU = cp.empty((NEQ, NEQ), dtype=np.float64)
            RHb_GPU = cp.empty(NEQ, dtype=np.float64)
            t3 = time.time()

            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝛀  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for 𝛀 [HpOMG]
            _module = SFFTModule_dict['HadProd_OMG']
            _func = _module.get_function('kmain')
            HpOMG_GPU = cp.empty((FOMG, N0, N1), dtype=np.complex128)
            _func(args=(SREF_iji0j0_GPU, SPixA_FIij_GPU, SPixA_CFIij_GPU, HpOMG_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PreOMG = SCALE * Re[DFT(HpOMG)]
            if EmpFFTPlan:
                HpOMG_GPU[:12] = SCALE * cp.fft.fft2(HpOMG_GPU[:12])
                HpOMG_GPU[12: 24] = SCALE * cp.fft.fft2(HpOMG_GPU[12: 24])
                HpOMG_GPU[24: 36] = SCALE * cp.fft.fft2(HpOMG_GPU[24: 36])

            if not EmpFFTPlan:
                HpOMG_GPU = SCALE * cp.fft.fft2(HpOMG_GPU)

            PreOMG_GPU = cp.array(SCALE * HpOMG_GPU.real, dtype=np.float64)
            del HpOMG_GPU

            # c. Fill Linear System with PreOMG
            _module = SFFTModule_dict['FillLS_OMG']
            _func = _module.get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreOMG_GPU, LHMAT_GPU), \
                block=TpB_OMG, grid=BpG_OMG)

            del PreOMG_GPU
            dt3 = time.time() - t3

            t4 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝜦  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for 𝜦 [HpGAM]
            _module = SFFTModule_dict['HadProd_GAM']
            _func = _module.get_function('kmain')
            HpGAM_GPU = cp.empty((FGAM, N0, N1), dtype=np.complex128)
            _func(args=(SREF_ijpq_GPU, SPixA_FIij_GPU, SPixA_CFTpq_GPU, HpGAM_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PreGAM = 1 * Re[DFT(HpGAM)]
            if EmpFFTPlan:
                HpGAM_GPU[:12] = SCALE * cp.fft.fft2(HpGAM_GPU[:12])
                HpGAM_GPU[12: 24] = SCALE * cp.fft.fft2(HpGAM_GPU[12: 24])
                HpGAM_GPU[24: 36] = SCALE * cp.fft.fft2(HpGAM_GPU[24: 36])

            if not EmpFFTPlan:
                HpGAM_GPU = SCALE * cp.fft.fft2(HpGAM_GPU)
            
            PreGAM_GPU = cp.array(HpGAM_GPU.real, dtype=np.float64)
            del HpGAM_GPU

            # c. Fill Linear System with PreGAM
            _module = SFFTModule_dict['FillLS_GAM']
            _func = _module.get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreGAM_GPU, LHMAT_GPU), \
                block=TpB_GAM, grid=BpG_GAM)

            del PreGAM_GPU
            dt4 = time.time() - t4

            t5 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝜳  -- -- -- -- -- -- -- -- *
            
            # a. Hadamard Product for 𝜳  [HpPSI]
            _module = SFFTModule_dict['HadProd_PSI']
            _func = _module.get_function('kmain')
            HpPSI_GPU = cp.empty((FPSI, N0, N1), dtype=np.complex128)
            _func(args=(SREF_pqij_GPU, SPixA_CFIij_GPU, SPixA_FTpq_GPU, HpPSI_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PrePSI = 1 * Re[DFT(HpPSI)]
            if EmpFFTPlan:
                HpPSI_GPU[:12] = SCALE * cp.fft.fft2(HpPSI_GPU[:12])
                HpPSI_GPU[12: 24] = SCALE * cp.fft.fft2(HpPSI_GPU[12: 24])
                HpPSI_GPU[24: 36] = SCALE * cp.fft.fft2(HpPSI_GPU[24: 36])

            if not EmpFFTPlan:
                HpPSI_GPU = SCALE * cp.fft.fft2(HpPSI_GPU)

            PrePSI_GPU = cp.array(HpPSI_GPU.real, dtype=np.float64)
            del HpPSI_GPU

            # c. Fill Linear System with PrePSI
            _module = SFFTModule_dict['FillLS_PSI']
            _func = _module.get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PrePSI_GPU, LHMAT_GPU), \
                block=TpB_PSI, grid=BpG_PSI)

            del PrePSI_GPU
            dt5 = time.time() - t5

            t6 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝚽  -- -- -- -- -- -- -- -- *
            
            # a. Hadamard Product for 𝚽  [HpPHI]
            _module = SFFTModule_dict['HadProd_PHI']
            _func = _module.get_function('kmain')
            HpPHI_GPU = cp.empty((FPHI, N0, N1), dtype=np.complex128)
            _func(args=(SREF_pqp0q0_GPU, SPixA_FTpq_GPU, SPixA_CFTpq_GPU, HpPHI_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
            if EmpFFTPlan:
                HpPHI_GPU[:12] = SCALE * cp.fft.fft2(HpPHI_GPU[:12])
                HpPHI_GPU[12: 24] = SCALE * cp.fft.fft2(HpPHI_GPU[12: 24])
                HpPHI_GPU[24: 36] = SCALE * cp.fft.fft2(HpPHI_GPU[24: 36])

            if not EmpFFTPlan:
                HpPHI_GPU = SCALE * cp.fft.fft2(HpPHI_GPU)

            PrePHI_GPU = cp.array(SCALE_L * HpPHI_GPU.real, dtype=np.float64)
            del HpPHI_GPU

            # c. Fill Linear System with PrePHI
            _module = SFFTModule_dict['FillLS_PHI']
            _func = _module.get_function('kmain')
            _func(args=(PrePHI_GPU, LHMAT_GPU), block=TpB_PHI, grid=BpG_PHI)

            del PrePHI_GPU
            dt6 = time.time() - t6

            t7 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through 𝚯 & 𝚫  -- -- -- -- -- -- -- -- *

            # a1. Hadamard Product for 𝚯  [HpTHE]
            _module = SFFTModule_dict['HadProd_THE']
            _func = _module.get_function('kmain')
            HpTHE_GPU = cp.empty((FTHE, N0, N1), dtype=np.complex128)
            _func(args=(SPixA_FIij_GPU, PixA_CFJ_GPU, HpTHE_GPU), block=TpB_PIX, grid=BpG_PIX)

            # a2. Hadamard Product for 𝚫  [HpDEL]
            _module = SFFTModule_dict['HadProd_DEL']
            _func = _module.get_function('kmain')
            HpDEL_GPU = cp.empty((FDEL, N0, N1), dtype=np.complex128)
            _func(args=(SPixA_FTpq_GPU, PixA_CFJ_GPU, HpDEL_GPU), block=TpB_PIX, grid=BpG_PIX)

            # b1. PreTHE = 1 * Re[DFT(HpTHE)]
            # b2. PreDEL = SCALE_L * Re[DFT(HpDEL)]
            if EmpFFTPlan:
                _SPixA_FFT_GPU = cp.empty((12, N0, N1), dtype=np.complex128)
                _SPixA_FFT_GPU[:6, :] = HpTHE_GPU
                _SPixA_FFT_GPU[6:, :] = HpDEL_GPU
                del HpTHE_GPU
                del HpDEL_GPU                
                _SPixA_FFT_GPU = SCALE * cp.fft.fft2(_SPixA_FFT_GPU)
                HpTHE_GPU = _SPixA_FFT_GPU[:6, :]
                HpDEL_GPU = _SPixA_FFT_GPU[6:, :]

            if not EmpFFTPlan:
                HpTHE_GPU = SCALE * cp.fft.fft2(HpTHE_GPU)
                HpDEL_GPU = SCALE * cp.fft.fft2(HpDEL_GPU)

            PreTHE_GPU = cp.array(HpTHE_GPU.real, dtype=np.float64)
            PreDEL_GPU = cp.array(SCALE_L * HpDEL_GPU.real, dtype=np.float64)
            del HpTHE_GPU
            del HpDEL_GPU

            # c1. Fill Linear System with PreTHE
            _module = SFFTModule_dict['FillLS_THE']
            _func = _module.get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreTHE_GPU, RHb_GPU), \
                block=TpB_THE, grid=BpG_THE)
            del PreTHE_GPU

            # c2. Fill Linear System with PreDEL
            _module = SFFTModule_dict['FillLS_DEL']
            _func = _module.get_function('kmain')
            _func(args=(PreDEL_GPU, RHb_GPU), block=TpB_DEL, grid=BpG_DEL)
            del PreDEL_GPU
            dt7 = time.time() - t7

            # * -- -- -- -- -- -- -- -- Remove Forbidden Stripes  -- -- -- -- -- -- -- -- *
            if not ConstPhotRatio: pass
            if ConstPhotRatio:
                RHb_FSfree_GPU = cp.take(RHb_GPU, IDX_nFS_GPU)
                _module = SFFTModule_dict['Remove_LSFStripes']
                _func = _module.get_function('kmain')
                BpG_FSfree_PA, TpB_FSfree_PA = GPUManage(NEQ_FSfree)     # Per Axis
                BpG_FSfree, TpB_FSfree = (BpG_FSfree_PA, BpG_FSfree_PA), (TpB_FSfree_PA, TpB_FSfree_PA, 1)
                LHMAT_FSfree_GPU = cp.empty((NEQ_FSfree, NEQ_FSfree), dtype=np.float64)
                _func(args=(LHMAT_GPU, IDX_nFS_GPU, LHMAT_FSfree_GPU), block=TpB_FSfree, grid=BpG_FSfree)

            t8 = time.time()
            # * -- -- -- -- -- -- -- -- Solve Linear System  -- -- -- -- -- -- -- -- *
            if not ConstPhotRatio: 
                Solution_GPU = LSSolver(LHMAT_GPU=LHMAT_GPU, RHb_GPU=RHb_GPU)

            if ConstPhotRatio:
                # Extend the solution to be consistent form
                Solution_FSfree_GPU = LSSolver(LHMAT_GPU=LHMAT_FSfree_GPU, RHb_GPU=RHb_FSfree_GPU)
                _module = SFFTModule_dict['Extend_Solution']
                _func = _module.get_function('kmain')
                BpG_ES, TpB_ES = (BpG_FSfree_PA, 1), (TpB_FSfree_PA, 1, 1)
                Solution_GPU = cp.zeros(NEQ, dtype=np.float64)
                _func(args=(Solution_FSfree_GPU, IDX_nFS_GPU, Solution_GPU), block=TpB_ES, grid=BpG_ES)
            
            a_ijab_GPU = Solution_GPU[: Fijab]
            b_pq_GPU = Solution_GPU[Fijab: ]
            Solution = cp.asnumpy(Solution_GPU)
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
            Wl_GPU = cp.exp((-2j*np.pi/N0) * PixA_X_GPU.astype(np.float64))    # row index l, [0, N0)
            Wm_GPU = cp.exp((-2j*np.pi/N1) * PixA_Y_GPU.astype(np.float64))    # column index m, [0, N1)
            Kab_Wla_GPU = cp.empty((L0, N0, N1), dtype=np.complex128)
            Kab_Wmb_GPU = cp.empty((L1, N0, N1), dtype=np.complex128)

            if w0 == w1:
                wx = w0   # a little bit faster
                for aob in range(-wx, wx+1):
                    Kab_Wla_GPU[aob + wx] = Wl_GPU ** aob    # offset 
                    Kab_Wmb_GPU[aob + wx] = Wm_GPU ** aob    # offset 
            else:
                for a in range(-w0, w0+1): 
                    Kab_Wla_GPU[a + w0] = Wl_GPU ** a        # offset 
                for b in range(-w1, w1+1): 
                    Kab_Wmb_GPU[b + w1] = Wm_GPU ** b        # offset 
            dt9 = time.time() - t9

            t10 = time.time()
            # Construct Difference in Fourier Space
            _module = SFFTModule_dict['Construct_FDIFF']
            _func = _module.get_function('kmain')     
            PixA_FDIFF_GPU = cp.empty((N0, N1), dtype=np.complex128)
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, a_ijab_GPU.astype(np.complex128), \
                SPixA_FIij_GPU, Kab_Wla_GPU, Kab_Wmb_GPU, b_pq_GPU.astype(np.complex128), \
                SPixA_FTpq_GPU, PixA_FJ_GPU, PixA_FDIFF_GPU), block=TpB_PIX, grid=BpG_PIX)
            
            # Get Difference & Reconstructed Images
            PixA_DIFF_GPU = SCALE_L * cp.fft.ifft2(PixA_FDIFF_GPU)
            PixA_DIFF = cp.asnumpy(PixA_DIFF_GPU.real)
            dt10 = time.time() - t10
            dtc = time.time() - tc

            print('\nMeLOn CheckPoint: Perform Subtraction     [%.4fs]' %dtc)
            print('/////   j   ///// Calculate Kab           (%.4fs)' %dt9)
            print('/////   k   ///// Construct DIFF        (%.4fs)' %dt10)
        
        print('\n  --||--||--||--||-- EXIT SFFT SUBTRACTION --||--||--||--||-- ')

        return Solution, PixA_DIFF    

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
        REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(np.int32)
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
 
        if backend == 'Pycuda':
            Solution, PixA_DIFF = ElementalSFFTSubtract_Pycuda.ESSP(PixA_I=PixA_I, PixA_J=PixA_J, SFFTConfig=SFFTConfig, SFFTSolution=SFFTSolution, Subtract=Subtract, CUDA_DEVICE=CUDA_DEVICE)
        
        if backend == 'Cupy':
            Solution, PixA_DIFF = ElementalSFFTSubtract_Cupy.ESSC(PixA_I=PixA_I, PixA_J=PixA_J, SFFTConfig=SFFTConfig, SFFTSolution=SFFTSolution, Subtract=Subtract, CUDA_DEVICE=CUDA_DEVICE)

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
