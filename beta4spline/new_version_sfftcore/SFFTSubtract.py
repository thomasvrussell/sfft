import time
import numpy as np
from scipy.interpolate import BSpline
# version: Apr 17, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class ElementalSFFTSubtract_Cupy:
    @staticmethod
    def ESSC(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, VERBOSE_LEVEL=2):
        
        import cupy as cp
        import cupyx.scipy.linalg as cpx_linalg

        def LSSolver(LHMAT_GPU, RHb_GPU):
            lu_piv_GPU = cpx_linalg.lu_factor(LHMAT_GPU, overwrite_a=False, check_finite=True)
            Solution_GPU = cpx_linalg.lu_solve(lu_piv_GPU, RHb_GPU)
            return Solution_GPU
        
        def Create_BSplineBasis(N, IntKnot, BSplineDegree):
            BSplineBasis = []
            PixCoord = (1.0+np.arange(N))/N
            Knot = np.concatenate(([0.5]*(BSplineDegree+1), IntKnot, [N+0.5]*(BSplineDegree+1)))/N
            Nc = len(IntKnot) + BSplineDegree + 1    # number of control points/coeffcients
            for idx in range(Nc):
                Coeff = (np.arange(Nc) == idx).astype(float)
                BaseFunc = BSpline(t=Knot, c=Coeff, k=BSplineDegree, extrapolate=False)
                BSplineBasis.append(BaseFunc(PixCoord))
            BSplineBasis = np.array(BSplineBasis)
            return BSplineBasis
        
        ta = time.time()
        # * Read SFFT parameters
        SFFTParam_dict, SFFTModule_dict = SFFTConfig

        KerHW = SFFTParam_dict['KerHW']
        KerSpType = SFFTParam_dict['KerSpType']
        KerSpDegree = SFFTParam_dict['KerSpDegree']
        KerIntKnotX = SFFTParam_dict['KerIntKnotX']
        KerIntKnotY = SFFTParam_dict['KerIntKnotY']
        BkgSpType = SFFTParam_dict['BkgSpType']
        BkgSpDegree = SFFTParam_dict['BkgSpDegree']
        BkgIntKnotX = SFFTParam_dict['BkgIntKnotX']
        BkgIntKnotY = SFFTParam_dict['BkgIntKnotY']
        ConstPhotRatio = SFFTParam_dict['ConstPhotRatio']

        N0 = SFFTParam_dict['N0']    # a.k.a, NX
        N1 = SFFTParam_dict['N1']    # a.k.a, NY
        w0 = SFFTParam_dict['w0']    # a.k.a, KerHW
        w1 = SFFTParam_dict['w1']    # a.k.a, KerHW
        DK = SFFTParam_dict['DK']    # a.k.a, KerSpDegree
        DB = SFFTParam_dict['DB']    # a.k.a, BkgSpDegree

        MaxThreadPerB = SFFTParam_dict['MaxThreadPerB']
        SCALE = SFFTParam_dict['SCALE']
        SCALE_L = SFFTParam_dict['SCALE_L']

        L0, L1, Fab = SFFTParam_dict['L0'], SFFTParam_dict['L1'], SFFTParam_dict['Fab']
        Fi, Fj, Fij = SFFTParam_dict['Fi'], SFFTParam_dict['Fj'], SFFTParam_dict['Fij']
        Fp, Fq, Fpq = SFFTParam_dict['Fp'], SFFTParam_dict['Fq'], SFFTParam_dict['Fpq']
        Fijab = SFFTParam_dict['Fijab']

        FOMG, FGAM = SFFTParam_dict['FOMG'], SFFTParam_dict['FGAM']
        FTHE, FPSI = SFFTParam_dict['FTHE'], SFFTParam_dict['FPSI']
        FPHI, FDEL = SFFTParam_dict['FPHI'], SFFTParam_dict['FDEL']
        NEQ, NEQt = SFFTParam_dict['NEQ'], SFFTParam_dict['NEQt']

        assert PixA_I.shape == (N0, N1) and PixA_J.shape == (N0, N1)
            
        if VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- TRIGGER SFFT SUBTRACTION [Cupy] --//--//--//--//-- ')
            if KerSpType == 'Polynomial':
                print('\n ---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(DK, w0))
            if KerSpType == 'B-Spline':
                print('\n ---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                    %(len(KerIntKnotX), len(KerIntKnotY), DK, w0))
            if BkgSpType == 'Polynomial':
                print('\n ---//--- Polynomial Background | BkgSpDegree %d ---//---' %DB)
            if BkgSpType == 'B-Spline':
                print('\n ---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), DB))

        # * Grid-Block-Thread Managaement [Pixel-Level]
        GPUManage = lambda NT: ((NT-1)//MaxThreadPerB + 1, min(NT, MaxThreadPerB))
        BpG_PIX0, TpB_PIX0 = GPUManage(N0)
        BpG_PIX1, TpB_PIX1 = GPUManage(N1)
        BpG_PIX, TpB_PIX = (BpG_PIX0, BpG_PIX1), (TpB_PIX0, TpB_PIX1, 1)

        # * Define First-order MultiIndex Reference
        if KerSpType == 'Polynomial':
            REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(np.int32)
        if KerSpType == 'B-Spline':
            REF_ij = np.array([(i, j) for i in range(Fi) for j in range(Fj)]).astype(np.int32)
        if BkgSpType == 'Polynomial':
            REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        if BkgSpType == 'B-Spline':
            REF_pq = np.array([(p, q) for p in range(Fp) for q in range(Fq)]).astype(np.int32)
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

        # * Indices related to Constant Scaling Case
        ij00 = np.arange(w0 * L1 + w1, Fijab, Fab).astype(np.int32)
        ij00_GPU = cp.array(ij00)

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
        
        # * Symbol Convention Notes
        #   X (x) / Y (y) ----- pixel row / column index
        #   CX (cx) / CY (cy) ----- ScaledFortranCoor of pixel (x, y) center   
        #   e.g. pixel (x, y) = (3, 5) corresponds (cx, cy) = (4.0/N0, 6.0/N1)
        #   NOTE cx / cy is literally \mathtt{x} / \mathtt{y} in SFFT paper.
        #   NOTE Without special definition, MeLOn convention refers to X (x) / Y (y) as FortranCoor.

        # * Get Spatial Coordinates
        t1 = time.time()
        PixA_X_GPU = cp.zeros((N0, N1), dtype=np.int32)      # row index, [0, N0)
        PixA_Y_GPU = cp.zeros((N0, N1), dtype=np.int32)      # column index, [0, N1)
        PixA_CX_GPU = cp.zeros((N0, N1), dtype=np.float64)   # coordinate.x
        PixA_CY_GPU = cp.zeros((N0, N1), dtype=np.float64)   # coordinate.y 

        _module = SFFTModule_dict['SpatialCoord']
        _func = _module.get_function('kmain')
        _func(args=(PixA_X_GPU, PixA_Y_GPU, PixA_CX_GPU, PixA_CY_GPU), block=TpB_PIX, grid=BpG_PIX)

        # * Spatial Variation Iij
        SPixA_Iij_GPU = cp.zeros((Fij, N0, N1), dtype=np.float64)
        if KerSpType == 'Polynomial':
            _module = SFFTModule_dict['KerSpatial']
            _func = _module.get_function('kmain')
            _func(args=(REF_ij_GPU, PixA_CX_GPU, PixA_CY_GPU, PixA_I_GPU, SPixA_Iij_GPU), \
                block=TpB_PIX, grid=BpG_PIX)
        
        if KerSpType == 'B-Spline':
            KerSplBasisX = Create_BSplineBasis(N=N0, IntKnot=KerIntKnotX, BSplineDegree=DK).astype(np.float64)
            KerSplBasisY = Create_BSplineBasis(N=N1, IntKnot=KerIntKnotY, BSplineDegree=DK).astype(np.float64)
            KerSplBasisX_GPU = cp.array(KerSplBasisX)
            KerSplBasisY_GPU = cp.array(KerSplBasisY)

            _module = SFFTModule_dict['KerSpatial']
            _func = _module.get_function('kmain')
            _func(args=(REF_ij_GPU, KerSplBasisX_GPU, KerSplBasisY_GPU, PixA_I_GPU, SPixA_Iij_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

        del PixA_I_GPU

        # * Spatial Variation Tpq
        SPixA_Tpq_GPU = cp.zeros((Fpq, N0, N1), dtype=np.float64)
        if BkgSpType == 'Polynomial':
            _module = SFFTModule_dict['BkgSpatial']
            _func = _module.get_function('kmain')
            _func(args=(REF_pq_GPU, PixA_CX_GPU, PixA_CY_GPU, SPixA_Tpq_GPU), \
                block=TpB_PIX, grid=BpG_PIX)
        
        if BkgSpType == 'B-Spline':
            BkgSplBasisX = Create_BSplineBasis(N=N0, IntKnot=BkgIntKnotX, BSplineDegree=DB).astype(np.float64)
            BkgSplBasisY = Create_BSplineBasis(N=N1, IntKnot=BkgIntKnotY, BSplineDegree=DB).astype(np.float64)
            BkgSplBasisX_GPU = cp.array(BkgSplBasisX)
            BkgSplBasisY_GPU = cp.array(BkgSplBasisY)

            _module = SFFTModule_dict['BkgSpatial']
            _func = _module.get_function('kmain')
            _func(args=(REF_pq_GPU, BkgSplBasisX_GPU, BkgSplBasisY_GPU, SPixA_Tpq_GPU), \
                block=TpB_PIX, grid=BpG_PIX)
        
        dt1 = time.time() - t1
        t2 = time.time()

        # * Make DFT of J, Iij, Tpq and their conjugates
        PixA_FJ_GPU = cp.empty((N0, N1), dtype=np.complex128)
        PixA_FJ_GPU[:, :] = PixA_J_GPU.astype(np.complex128)
        PixA_FJ_GPU[:, :] = cp.fft.fft2(PixA_FJ_GPU)
        PixA_FJ_GPU[:, :] *= SCALE

        SPixA_FIij_GPU = cp.empty((Fij, N0, N1), dtype=np.complex128)
        SPixA_FIij_GPU[:, :, :] = SPixA_Iij_GPU.astype(np.complex128)
        for k in range(Fij): 
            SPixA_FIij_GPU[k: k+1] = cp.fft.fft2(SPixA_FIij_GPU[k: k+1])
        SPixA_FIij_GPU[:, :] *= SCALE

        SPixA_FTpq_GPU = cp.empty((Fpq, N0, N1), dtype=np.complex128)
        SPixA_FTpq_GPU[:, :, :] = SPixA_Tpq_GPU.astype(np.complex128)
        for k in range(Fpq): 
            SPixA_FTpq_GPU[k: k+1] = cp.fft.fft2(SPixA_FTpq_GPU[k: k+1])
        SPixA_FTpq_GPU[:, :] *= SCALE

        del PixA_J_GPU
        del SPixA_Iij_GPU
        del SPixA_Tpq_GPU

        PixA_CFJ_GPU = cp.conj(PixA_FJ_GPU)
        SPixA_CFIij_GPU = cp.conj(SPixA_FIij_GPU)
        SPixA_CFTpq_GPU = cp.conj(SPixA_FTpq_GPU)
        dt2 = time.time() - t2
        dta = time.time() - ta

        if VERBOSE_LEVEL in [1, 2]:
            print('\nMeLOn CheckPoint: SFFT-SUBTRACTION Preliminary Steps takes [%.4fs]' %dta)

        if VERBOSE_LEVEL in [2]:
            print('/////   a   ///// Read Input Images  (%.4fs)' %dt0)
            print('/////   b   ///// Spatial Polynomial (%.4fs)' %dt1)
            print('/////   c   ///// DFT-%d             (%.4fs)' %(1 + Fij + Fpq, dt2))

        # * Consider The Major Sources of the Linear System 
        #    Omega_i8j8ij   &    DELTA_i8j8pq    ||   THETA_i8j8
        #    PSI_p8q8ij     &    PHI_p8q8pq      ||   DELTA_p8q8
        
        # - Remarks
        #    a. They have consistent form: Greek(rho, eps) = PreGreek(Mod_N0(rho), Mod_N1(eps))
        #    b. PreGreek = s * Re[DFT(HpGreek)], where HpGreek = GLH * GRH and s is some real-scale
        #    c. Considering the subscripted variables, HpGreek/PreGreek is Complex/Real 3D with shape (F_Greek, N0, N1)
        
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

            # * -- -- -- -- -- -- -- -- Establish Linear System through Omega  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for Omega [HpOMG]
            _module = SFFTModule_dict['HadProd_OMG']
            _func = _module.get_function('kmain')
            HpOMG_GPU = cp.empty((FOMG, N0, N1), dtype=np.complex128)
            _func(args=(SREF_iji0j0_GPU, SPixA_FIij_GPU, SPixA_CFIij_GPU, HpOMG_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PreOMG = SCALE * Re[DFT(HpOMG)]
            for k in range(FOMG):
                HpOMG_GPU[k: k+1] = cp.fft.fft2(HpOMG_GPU[k: k+1])
            HpOMG_GPU *= SCALE

            PreOMG_GPU = cp.empty((FOMG, N0, N1), dtype=np.float64)
            PreOMG_GPU[:, :, :] = HpOMG_GPU.real
            PreOMG_GPU[:, :, :] *= SCALE
            del HpOMG_GPU

            # c. Fill Linear System with PreOMG
            _module = SFFTModule_dict['FillLS_OMG']
            _func = _module.get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreOMG_GPU, LHMAT_GPU), \
                block=TpB_OMG, grid=BpG_OMG)

            del PreOMG_GPU
            dt3 = time.time() - t3

            t4 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through GAMMA  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for GAMMA [HpGAM]
            _module = SFFTModule_dict['HadProd_GAM']
            _func = _module.get_function('kmain')
            HpGAM_GPU = cp.empty((FGAM, N0, N1), dtype=np.complex128)
            _func(args=(SREF_ijpq_GPU, SPixA_FIij_GPU, SPixA_CFTpq_GPU, HpGAM_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PreGAM = 1 * Re[DFT(HpGAM)]
            for k in range(FGAM):
                HpGAM_GPU[k: k+1] = cp.fft.fft2(HpGAM_GPU[k: k+1])
            HpGAM_GPU *= SCALE

            PreGAM_GPU = cp.empty((FGAM, N0, N1), dtype=np.float64)
            PreGAM_GPU[:, :, :] = HpGAM_GPU.real
            del HpGAM_GPU

            # c. Fill Linear System with PreGAM
            _module = SFFTModule_dict['FillLS_GAM']
            _func = _module.get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PreGAM_GPU, LHMAT_GPU), \
                block=TpB_GAM, grid=BpG_GAM)

            del PreGAM_GPU
            dt4 = time.time() - t4

            t5 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through PSI  -- -- -- -- -- -- -- -- *
            
            # a. Hadamard Product for PSI  [HpPSI]
            _module = SFFTModule_dict['HadProd_PSI']
            _func = _module.get_function('kmain')
            HpPSI_GPU = cp.empty((FPSI, N0, N1), dtype=np.complex128)
            _func(args=(SREF_pqij_GPU, SPixA_CFIij_GPU, SPixA_FTpq_GPU, HpPSI_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PrePSI = 1 * Re[DFT(HpPSI)]
            for k in range(FPSI):
                HpPSI_GPU[k: k+1] = cp.fft.fft2(HpPSI_GPU[k: k+1])
            HpPSI_GPU *= SCALE

            PrePSI_GPU = cp.empty((FPSI, N0, N1), dtype=np.float64)
            PrePSI_GPU[:, :, :] = HpPSI_GPU.real
            del HpPSI_GPU

            # c. Fill Linear System with PrePSI
            _module = SFFTModule_dict['FillLS_PSI']
            _func = _module.get_function('kmain')
            _func(args=(SREF_ijab_GPU, REF_ab_GPU, PrePSI_GPU, LHMAT_GPU), \
                block=TpB_PSI, grid=BpG_PSI)

            del PrePSI_GPU
            dt5 = time.time() - t5

            t6 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through PHI  -- -- -- -- -- -- -- -- *
            
            # a. Hadamard Product for PHI  [HpPHI]
            _module = SFFTModule_dict['HadProd_PHI']
            _func = _module.get_function('kmain')
            HpPHI_GPU = cp.empty((FPHI, N0, N1), dtype=np.complex128)
            _func(args=(SREF_pqp0q0_GPU, SPixA_FTpq_GPU, SPixA_CFTpq_GPU, HpPHI_GPU), \
                block=TpB_PIX, grid=BpG_PIX)

            # b. PrePHI = SCALE_L * Re[DFT(HpPHI)]
            for k in range(FPHI):
                HpPHI_GPU[k: k+1] = cp.fft.fft2(HpPHI_GPU[k: k+1])
            HpPHI_GPU *= SCALE

            PrePHI_GPU = cp.empty((FPHI, N0, N1), dtype=np.float64)
            PrePHI_GPU[:, :, :] = HpPHI_GPU.real
            PrePHI_GPU[:, :, :] *= SCALE_L
            del HpPHI_GPU

            # c. Fill Linear System with PrePHI
            _module = SFFTModule_dict['FillLS_PHI']
            _func = _module.get_function('kmain')
            _func(args=(PrePHI_GPU, LHMAT_GPU), block=TpB_PHI, grid=BpG_PHI)

            del PrePHI_GPU
            dt6 = time.time() - t6

            t7 = time.time()
            # * -- -- -- -- -- -- -- -- Establish Linear System through THETA & DELTA  -- -- -- -- -- -- -- -- *

            # a1. Hadamard Product for THETA  [HpTHE]
            _module = SFFTModule_dict['HadProd_THE']
            _func = _module.get_function('kmain')
            HpTHE_GPU = cp.empty((FTHE, N0, N1), dtype=np.complex128)
            _func(args=(SPixA_FIij_GPU, PixA_CFJ_GPU, HpTHE_GPU), block=TpB_PIX, grid=BpG_PIX)

            # a2. Hadamard Product for DELTA  [HpDEL]
            _module = SFFTModule_dict['HadProd_DEL']
            _func = _module.get_function('kmain')
            HpDEL_GPU = cp.empty((FDEL, N0, N1), dtype=np.complex128)
            _func(args=(SPixA_FTpq_GPU, PixA_CFJ_GPU, HpDEL_GPU), block=TpB_PIX, grid=BpG_PIX)

            # b1. PreTHE = 1 * Re[DFT(HpTHE)]
            # b2. PreDEL = SCALE_L * Re[DFT(HpDEL)]
            for k in range(FTHE):
                HpTHE_GPU[k: k+1] = cp.fft.fft2(HpTHE_GPU[k: k+1])
            HpTHE_GPU[:, :, :] *= SCALE
            
            for k in range(FDEL):
                HpDEL_GPU[k: k+1] = cp.fft.fft2(HpDEL_GPU[k: k+1])
            HpDEL_GPU[:, :, :] *= SCALE

            PreTHE_GPU = cp.empty((FTHE, N0, N1), dtype=np.float64)
            PreTHE_GPU[:, :, :] = HpTHE_GPU.real
            del HpTHE_GPU
            
            PreDEL_GPU = cp.empty((FDEL, N0, N1), dtype=np.float64)
            PreDEL_GPU[:, :, :] = HpDEL_GPU.real
            PreDEL_GPU[:, :, :] *= SCALE_L
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

            t8 = time.time()
            if not ConstPhotRatio: pass
            if ConstPhotRatio:
                # * -- -- -- -- -- -- -- -- Tweak Linear System  -- -- -- -- -- -- -- -- *
                LHMAT_tweaked_GPU = cp.empty((NEQt, NEQt), dtype=np.float64)
                RHb_tweaked_GPU = cp.empty(NEQt, dtype=np.float64)

                PresIDX = np.setdiff1d(np.arange(NEQ), ij00[1:], assume_unique=True).astype(np.int32)
                assert np.all(PresIDX[:-1] < PresIDX[1:])
                assert PresIDX[ij00[0]] == ij00[0]
                PresIDX_GPU = cp.array(PresIDX)

                BpG_tweak_PA, TpB_tweak_PA = GPUManage(NEQt)   # per axis
                BpG_tweak, TpB_tweak = (BpG_tweak_PA, BpG_tweak_PA), (TpB_tweak_PA, TpB_tweak_PA, 1)

                if KerSpType == 'Polynomial':
                    _module = SFFTModule_dict['TweakLS']
                    _func = _module.get_function('kmain')
                    _func(args=(LHMAT_GPU, RHb_GPU, PresIDX_GPU, LHMAT_tweaked_GPU, \
                        RHb_tweaked_GPU), block=TpB_tweak, grid=BpG_tweak)
                
                if KerSpType == 'B-Spline':
                    _module = SFFTModule_dict['TweakLS']
                    _func = _module.get_function('kmain')
                    _func(args=(LHMAT_GPU, RHb_GPU, PresIDX_GPU, ij00_GPU, \
                        LHMAT_tweaked_GPU, RHb_tweaked_GPU), block=TpB_tweak, grid=BpG_tweak)
            
            # * -- -- -- -- -- -- -- -- Solve Linear System  -- -- -- -- -- -- -- -- *
            
            if not ConstPhotRatio: 
                Solution_GPU = LSSolver(LHMAT_GPU=LHMAT_GPU, RHb_GPU=RHb_GPU)

            if ConstPhotRatio:
                Solution_tweaked_GPU = LSSolver(LHMAT_GPU=LHMAT_tweaked_GPU, RHb_GPU=RHb_tweaked_GPU)
                
                # * -- -- -- -- -- -- -- -- Restore Solution  -- -- -- -- -- -- -- -- *
                BpG_restore, TpB_restore = GPUManage(NEQt)
                BpG_restore, TpB_restore = (BpG_restore, 1), (TpB_restore, 1, 1)

                if KerSpType == 'Polynomial':
                    Solution_GPU = cp.zeros(NEQ, dtype=np.float64)
                
                if KerSpType == 'B-Spline':
                    Solution_GPU = cp.zeros(NEQ, dtype=np.float64)
                    Solution_GPU[ij00[1:]] = Solution_tweaked_GPU[ij00[0]]

                _module = SFFTModule_dict['Restore_Solution']
                _func = _module.get_function('kmain')
                _func(args=(Solution_tweaked_GPU, PresIDX_GPU, Solution_GPU), \
                    block=TpB_restore, grid=BpG_restore)
            
            a_ijab_GPU = Solution_GPU[: Fijab]
            b_pq_GPU = Solution_GPU[Fijab: ]
            Solution = cp.asnumpy(Solution_GPU)
            dt8 = time.time() - t8
            dtb = time.time() - tb

            if VERBOSE_LEVEL in [1, 2]:
                print('\nMeLOn CheckPoint: SFFT-SUBTRACTION Establish & Solve Linear System takes [%.4fs]' %dtb)

            if VERBOSE_LEVEL in [2]:
                print('/////   d   ///// Establish OMG                       (%.4fs)' %dt3)
                print('/////   e   ///// Establish GAM                       (%.4fs)' %dt4)
                print('/////   f   ///// Establish PSI                       (%.4fs)' %dt5)
                print('/////   g   ///// Establish PHI                       (%.4fs)' %dt6)
                print('/////   h   ///// Establish THE & DEL                 (%.4fs)' %dt7)
                print('/////   i   ///// Solve Linear System                 (%.4fs)' %dt8)

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

            if VERBOSE_LEVEL in [1, 2]:
                print('\nMeLOn CheckPoint: SFFT-SUBTRACTION Perform Subtraction takes [%.4fs]' %dtc)
            
            if VERBOSE_LEVEL in [2]:
                print('/////   j   ///// Calculate Kab         (%.4fs)' %dt9)
                print('/////   k   ///// Construct DIFF        (%.4fs)' %dt10)
        
        if VERBOSE_LEVEL in [1, 2]:
            print('\n --||--||--||--||-- EXIT SFFT SUBTRACTION [Cupy] --||--||--||--||-- ')

        return Solution, PixA_DIFF

class ElementalSFFTSubtract_Numpy:
    @staticmethod
    def ESSN(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        import pyfftw
        pyfftw.config.NUM_THREADS = NUM_CPU_THREADS_4SUBTRACT
        pyfftw.interfaces.cache.enable()

        def Create_BSplineBasis(N, IntKnot, BSplineDegree):
            BSplineBasis = []
            PixCoord = (1.0+np.arange(N))/N
            Knot = np.concatenate(([0.5]*(BSplineDegree+1), IntKnot, [N+0.5]*(BSplineDegree+1)))/N
            Nc = len(IntKnot) + BSplineDegree + 1    # number of control points/coeffcients
            for idx in range(Nc):
                Coeff = (np.arange(Nc) == idx).astype(float)
                BaseFunc = BSpline(t=Knot, c=Coeff, k=BSplineDegree, extrapolate=False)
                BSplineBasis.append(BaseFunc(PixCoord))
            BSplineBasis = np.array(BSplineBasis)
            return BSplineBasis
        
        ta = time.time()
        # * Read SFFT parameters
        SFFTParam_dict, SFFTModule_dict = SFFTConfig

        KerHW = SFFTParam_dict['KerHW']
        KerSpType = SFFTParam_dict['KerSpType']
        KerSpDegree = SFFTParam_dict['KerSpDegree']
        KerIntKnotX = SFFTParam_dict['KerIntKnotX']
        KerIntKnotY = SFFTParam_dict['KerIntKnotY']
        BkgSpType = SFFTParam_dict['BkgSpType']
        BkgSpDegree = SFFTParam_dict['BkgSpDegree']
        BkgIntKnotX = SFFTParam_dict['BkgIntKnotX']
        BkgIntKnotY = SFFTParam_dict['BkgIntKnotY']
        ConstPhotRatio = SFFTParam_dict['ConstPhotRatio']

        N0 = SFFTParam_dict['N0']    # a.k.a, NX
        N1 = SFFTParam_dict['N1']    # a.k.a, NY
        w0 = SFFTParam_dict['w0']    # a.k.a, KerHW
        w1 = SFFTParam_dict['w1']    # a.k.a, KerHW
        DK = SFFTParam_dict['DK']    # a.k.a, KerSpDegree
        DB = SFFTParam_dict['DB']    # a.k.a, BkgSpDegree
        
        MaxThreadPerB = SFFTParam_dict['MaxThreadPerB']  # trivial
        SCALE = SFFTParam_dict['SCALE']
        SCALE_L = SFFTParam_dict['SCALE_L']

        L0, L1, Fab = SFFTParam_dict['L0'], SFFTParam_dict['L1'], SFFTParam_dict['Fab']
        Fi, Fj, Fij = SFFTParam_dict['Fi'], SFFTParam_dict['Fj'], SFFTParam_dict['Fij']
        Fp, Fq, Fpq = SFFTParam_dict['Fp'], SFFTParam_dict['Fq'], SFFTParam_dict['Fpq']
        Fijab = SFFTParam_dict['Fijab']

        FOMG, FGAM = SFFTParam_dict['FOMG'], SFFTParam_dict['FGAM']
        FTHE, FPSI = SFFTParam_dict['FTHE'], SFFTParam_dict['FPSI']
        FPHI, FDEL = SFFTParam_dict['FPHI'], SFFTParam_dict['FDEL']
        NEQ, NEQt = SFFTParam_dict['NEQ'], SFFTParam_dict['NEQt']

        assert PixA_I.shape == (N0, N1) and PixA_J.shape == (N0, N1)
            
        if VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- TRIGGER SFFT SUBTRACTION [Numpy] --//--//--//--//-- ')
            if KerSpType == 'Polynomial':
                print('\n ---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(DK, w0))
            if KerSpType == 'B-Spline':
                print('\n ---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                    %(len(KerIntKnotX), len(KerIntKnotY), DK, w0))
            if BkgSpType == 'Polynomial':
                print('\n ---//--- Polynomial Background | BkgSpDegree %d ---//---' %DB)
            if BkgSpType == 'B-Spline':
                print('\n ---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), DB))

        # * Define First-order MultiIndex Reference
        if KerSpType == 'Polynomial':
            REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(np.int32)
        if KerSpType == 'B-Spline':
            REF_ij = np.array([(i, j) for i in range(Fi) for j in range(Fj)]).astype(np.int32)
        if BkgSpType == 'Polynomial':
            REF_pq = np.array([(p, q) for p in range(DB+1) for q in range(DB+1-p)]).astype(np.int32)
        if BkgSpType == 'B-Spline':
            REF_pq = np.array([(p, q) for p in range(Fp) for q in range(Fq)]).astype(np.int32)        
        REF_ab = np.array([(a_pos-w0, b_pos-w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(np.int32)

        # * Define Second-order MultiIndex Reference
        SREF_iji0j0 = np.array([(ij, i0j0) for ij in range(Fij) for i0j0 in range(Fij)]).astype(np.int32)
        SREF_pqp0q0 = np.array([(pq, p0q0) for pq in range(Fpq) for p0q0 in range(Fpq)]).astype(np.int32)
        SREF_ijpq = np.array([(ij, pq) for ij in range(Fij) for pq in range(Fpq)]).astype(np.int32)
        SREF_pqij = np.array([(pq, ij) for pq in range(Fpq) for ij in range(Fij)]).astype(np.int32)
        SREF_ijab = np.array([(ij, ab) for ij in range(Fij) for ab in range(Fab)]).astype(np.int32)

        # * Indices related to Constant Scaling Case
        ij00 = np.arange(w0 * L1 + w1, Fijab, Fab).astype(np.int32)

        t0 = time.time()
        # * Read input images as C-order arrays
        if not PixA_I.flags['C_CONTIGUOUS']:
            PixA_I = np.ascontiguousarray(PixA_I, np.float64)
        else: PixA_I = PixA_I.astype(np.float64)

        if not PixA_J.flags['C_CONTIGUOUS']:
            PixA_J = np.ascontiguousarray(PixA_J, np.float64)
        else: PixA_J = PixA_J.astype(np.float64)
        dt0 = time.time() - t0

        # * Symbol Convention Notes
        #   X (x) / Y (y) ----- pixel row / column index
        #   CX (cx) / CY (cy) ----- ScaledFortranCoor of pixel (x, y) center   
        #   e.g. pixel (x, y) = (3, 5) corresponds (cx, cy) = (4.0/N0, 6.0/N1)
        #   NOTE cx / cy is literally \mathtt{x} / \mathtt{y} in SFFT paper.
        #   NOTE Without special definition, MeLOn convention refers to X (x) / Y (y) as FortranCoor.

        # * Get Spatial Coordinates
        t1 = time.time()
        PixA_X = np.zeros((N0, N1), dtype=np.int32)      # row index, [0, N0)
        PixA_Y = np.zeros((N0, N1), dtype=np.int32)      # column index, [0, N1)
        PixA_CX = np.zeros((N0, N1), dtype=np.float64)   # coordinate.x
        PixA_CY = np.zeros((N0, N1), dtype=np.float64)   # coordinate.y 

        _func = SFFTModule_dict['SpatialCoord']
        _func(PixA_X=PixA_X, PixA_Y=PixA_Y, PixA_CX=PixA_CX, PixA_CY=PixA_CY)

        # * Spatial Variation Iij
        SPixA_Iij = np.zeros((Fij, N0, N1), dtype=np.float64)
        if KerSpType == 'Polynomial':
            _func = SFFTModule_dict['KerSpatial']
            _func(REF_ij=REF_ij, PixA_CX=PixA_CX, PixA_CY=PixA_CY, \
                PixA_I=PixA_I, SPixA_Iij=SPixA_Iij)

        if KerSpType == 'B-Spline':
            KerSplBasisX = Create_BSplineBasis(N=N0, IntKnot=KerIntKnotX, BSplineDegree=DK).astype(np.float64)
            KerSplBasisY = Create_BSplineBasis(N=N1, IntKnot=KerIntKnotY, BSplineDegree=DK).astype(np.float64)

            _func = SFFTModule_dict['KerSpatial']
            _func(REF_ij=REF_ij, KerSplBasisX=KerSplBasisX, KerSplBasisY=KerSplBasisY, \
                PixA_I=PixA_I, SPixA_Iij=SPixA_Iij)

        # * Spatial Variation Tpq
        SPixA_Tpq = np.zeros((Fpq, N0, N1), dtype=np.float64)
        if BkgSpType == 'Polynomial':
            _func = SFFTModule_dict['BkgSpatial']
            _func(REF_pq=REF_pq, PixA_CX=PixA_CX, PixA_CY=PixA_CY, SPixA_Tpq=SPixA_Tpq)
        
        if BkgSpType == 'B-Spline':
            BkgSplBasisX = Create_BSplineBasis(N=N0, IntKnot=BkgIntKnotX, BSplineDegree=DB).astype(np.float64)
            BkgSplBasisY = Create_BSplineBasis(N=N1, IntKnot=BkgIntKnotY, BSplineDegree=DB).astype(np.float64)

            _func = SFFTModule_dict['BkgSpatial']
            _func(REF_pq=REF_pq, BkgSplBasisX=BkgSplBasisX, BkgSplBasisY=BkgSplBasisY, SPixA_Tpq=SPixA_Tpq)

        dt1 = time.time() - t1
        
        t2 = time.time()
        Batchsize = 1 + Fij + Fpq
        _SPixA_FFT = np.empty((Batchsize, N0, N1), dtype=np.complex128)
        _SPixA_FFT[0, :, :] = PixA_J.astype(np.complex128)
        _SPixA_FFT[1: Fij+1, :, :] = SPixA_Iij.astype(np.complex128)
        _SPixA_FFT[Fij+1:, :, :] = SPixA_Tpq.astype(np.complex128)
        _SPixA_FFT = SCALE * pyfftw.interfaces.numpy_fft.fft2(_SPixA_FFT)

        PixA_FJ = _SPixA_FFT[0, :, :]
        SPixA_FIij = _SPixA_FFT[1: Fij+1, :, :]
        SPixA_FTpq = _SPixA_FFT[Fij+1:, :, :]

        PixA_CFJ = np.conj(PixA_FJ)
        SPixA_CFIij = np.conj(SPixA_FIij)
        SPixA_CFTpq = np.conj(SPixA_FTpq)
        dt2 = time.time() - t2
        dta = time.time() - ta

        if VERBOSE_LEVEL in [1, 2]:
            print('\nMeLOn CheckPoint: SFFT-SUBTRACTION Preliminary Steps takes [%.4fs]' %dta)

        if VERBOSE_LEVEL in [2]:
            print('/////   a   ///// Read Input Images  (%.4fs)' %dt0)
            print('/////   b   ///// Spatial Polynomial (%.4fs)' %dt1)
            print('/////   c   ///// DFT-%d             (%.4fs)' %(1 + Fij + Fpq, dt2))

        # * Consider The Major Sources of the Linear System 
        #    Omega_i8j8ij   &    DELTA_i8j8pq    ||   THETA_i8j8
        #    PSI_p8q8ij     &    PHI_p8q8pq      ||   DELTA_p8q8
        
        # - Remarks
        #    a. They have consistent form: Greek(rho, eps) = PreGreek(Mod_N0(rho), Mod_N1(eps))
        #    b. PreGreek = s * Re[DFT(HpGreek)], where HpGreek = GLH * GRH and s is some real-scale
        #    c. Considering the subscripted variables, HpGreek/PreGreek is Complex/Real 3D with shape (F_Greek, N0, N1)

        if SFFTSolution is not None:
            Solution = SFFTSolution.astype(np.float64)
            a_ijab = Solution[: Fijab]
            b_pq = Solution[Fijab: ]
            
        if SFFTSolution is None:
            tb = time.time()    
            
            LHMAT = np.empty((NEQ, NEQ), dtype=np.float64)
            RHb = np.empty(NEQ, dtype=np.float64)
            t3 = time.time()
            
            # * -- -- -- -- -- -- -- -- Establish Linear System through Omega  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for Omega [HpOMG]
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
            # * -- -- -- -- -- -- -- -- Establish Linear System through GAMMA  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for GAMMA [HpGAM]
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
            # * -- -- -- -- -- -- -- -- Establish Linear System through PSI  -- -- -- -- -- -- -- -- *

            # a. Hadamard Product for PSI  [HpPSI]
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
            # * -- -- -- -- -- -- -- -- Establish Linear System through PHI  -- -- -- -- -- -- -- -- *
            
            # a. Hadamard Product for PHI  [HpPHI]
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
            # * -- -- -- -- -- -- -- -- Establish Linear System through THETA & DELTA  -- -- -- -- -- -- -- -- *

            # a1. Hadamard Product for THETA  [HpTHE]
            _func = SFFTModule_dict['HadProd_THE']
            HpTHE = np.empty((FTHE, N0, N1), dtype=np.complex128)
            HpTHE = _func(SPixA_FIij=SPixA_FIij, PixA_CFJ=PixA_CFJ, HpTHE=HpTHE)

            # a2. Hadamard Product for DELTA  [HpDEL]
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
            if not ConstPhotRatio: pass
            if ConstPhotRatio:
                # * -- -- -- -- -- -- -- -- Tweak Linear System  -- -- -- -- -- -- -- -- *
                LHMAT_tweaked = np.empty((NEQt, NEQt), dtype=np.float64)
                RHb_tweaked = np.empty(NEQt, dtype=np.float64)

                PresIDX = np.setdiff1d(np.arange(NEQ), ij00[1:], assume_unique=True).astype(np.int32)
                assert np.all(PresIDX[:-1] < PresIDX[1:])
                assert PresIDX[ij00[0]] == ij00[0]

                if KerSpType == 'Polynomial':
                    _func = SFFTModule_dict['TweakLS']
                    _func(LHMAT=LHMAT, RHb=RHb, PresIDX=PresIDX, \
                        LHMAT_tweaked=LHMAT_tweaked, RHb_tweaked=RHb_tweaked)
                
                if KerSpType == 'B-Spline':
                    _func = SFFTModule_dict['TweakLS']
                    _func(LHMAT=LHMAT, RHb=RHb, PresIDX=PresIDX, ij00=ij00, \
                        LHMAT_tweaked=LHMAT_tweaked, RHb_tweaked=RHb_tweaked)

            if not ConstPhotRatio: 
                Solution = np.linalg.solve(LHMAT, RHb).astype(np.float64)

            if ConstPhotRatio:
                Solution_tweaked = np.linalg.solve(LHMAT_tweaked, RHb_tweaked).astype(np.float64)
                            
                # * -- -- -- -- -- -- -- -- Restore Solution  -- -- -- -- -- -- -- -- *
                if KerSpType == 'Polynomial':
                    Solution = np.zeros(NEQ, dtype=np.float64)
                
                if KerSpType == 'B-Spline':
                    Solution = np.zeros(NEQ, dtype=np.float64)
                    Solution[ij00[1:]] = Solution_tweaked[ij00[0]]

                _func = SFFTModule_dict['Restore_Solution']
                _func(Solution_tweaked=Solution_tweaked, PresIDX=PresIDX, Solution=Solution)

            a_ijab = Solution[: Fijab]
            b_pq = Solution[Fijab: ]

            dt8 = time.time() - t8
            dtb = time.time() - tb

            if VERBOSE_LEVEL in [1, 2]:
                print('\nMeLOn CheckPoint: SFFT-SUBTRACTION Establish & Solve Linear System takes [%.4fs]' %dtb)

            if VERBOSE_LEVEL in [2]:
                print('/////   d   ///// Establish OMG                       (%.4fs)' %dt3)
                print('/////   e   ///// Establish GAM                       (%.4fs)' %dt4)
                print('/////   f   ///// Establish PSI                       (%.4fs)' %dt5)
                print('/////   g   ///// Establish PHI                       (%.4fs)' %dt6)
                print('/////   h   ///// Establish THE & DEL                 (%.4fs)' %dt7)
                print('/////   i   ///// Solve Linear System                 (%.4fs)' %dt8)
        
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

            if VERBOSE_LEVEL in [1, 2]:
                print('\nMeLOn CheckPoint: SFFT-SUBTRACTION Perform Subtraction takes [%.4fs]' %dtc)
            
            if VERBOSE_LEVEL in [2]:
                print('/////   j   ///// Calculate Kab         (%.4fs)' %dt9)
                print('/////   k   ///// Construct DIFF        (%.4fs)' %dt10)
        
        if VERBOSE_LEVEL in [1, 2]:
            print('\n --||--||--||--||-- EXIT SFFT SUBTRACTION [Numpy] --||--||--||--||-- ')

        return Solution, PixA_DIFF

class ElementalSFFTSubtract:
    @staticmethod
    def ESS(PixA_I, PixA_J, SFFTConfig, SFFTSolution=None, Subtract=False, \
        BACKEND_4SUBTRACT='Cupy', NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        if BACKEND_4SUBTRACT == 'Cupy':
            Solution, PixA_DIFF = ElementalSFFTSubtract_Cupy.ESSC(PixA_I=PixA_I, PixA_J=PixA_J, \
                SFFTConfig=SFFTConfig, SFFTSolution=SFFTSolution, Subtract=Subtract, VERBOSE_LEVEL=VERBOSE_LEVEL)

        if BACKEND_4SUBTRACT == 'Numpy':
            Solution, PixA_DIFF = ElementalSFFTSubtract_Numpy.ESSN(PixA_I=PixA_I, PixA_J=PixA_J, \
                SFFTConfig=SFFTConfig, SFFTSolution=SFFTSolution, Subtract=Subtract, \
                NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        return Solution, PixA_DIFF

class GeneralSFFTSubtract:
    @staticmethod
    def GSS(PixA_I, PixA_J, PixA_mI, PixA_mJ, SFFTConfig, ContamMask_I=None, \
        BACKEND_4SUBTRACT='Cupy', NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        """
        # Perform image subtraction on I & J with SFFT parameters solved from mI & mJ
        #
        # Arguments:
        # -PixA_I: Image I that will be convolved [NaN-Free]                                    
        # -PixA_J: Image J that won't be convolved [NaN-Free]                                   
        # -PixA_mI: Masked version of Image I. 'same' means it is identical with I [NaN-Free]  
        # -PixA_mJ: Masked version of Image J. 'same' means it is identical with J [NaN-Free]
        # -SFFTConfig: Configurations of SFFT
        
        # -ContamMask_I: Contamination Mask of Image I (e.g., Saturation and Bad pixels)
        # -BACKEND_4SUBTRACT: The backend with which you perform SFFT subtraction              | ['Cupy', 'Numpy']
        # -NUM_CPU_THREADS_4SUBTRACT: The number of CPU threads for Numpy-SFFT subtraction     | e.g., 8
        # -VERBOSE_LEVEL: The level of verbosity, can be 0/1/2: QUIET/NORMAL/FULL              | [0, 1, 2]
        #
        """
        
        SFFT_BANNER = r"""
                                __    __    __    __
                               /  \  /  \  /  \  /  \
                              /    \/    \/    \/    \
            █████████████████/  /██/  /██/  /██/  /█████████████████████████
                            /  / \   / \   / \   / \  \____
                           /  /   \_/   \_/   \_/   \    o \__,
                          / _/                       \_____/  `
                          |/
        
                      █████████  ███████████ ███████████ ███████████        
                     ███░░░░░███░░███░░░░░░█░░███░░░░░░█░█░░░███░░░█            
                    ░███    ░░░  ░███   █ ░  ░███   █ ░ ░   ░███  ░ 
                    ░░█████████  ░███████    ░███████       ░███    
                     ░░░░░░░░███ ░███░░░█    ░███░░░█       ░███    
                     ███    ░███ ░███  ░     ░███  ░        ░███    
                    ░░█████████  █████       █████          █████   
                     ░░░░░░░░░  ░░░░░       ░░░░░          ░░░░░         
        
                    Saccadic Fast Fourier Transform (SFFT) algorithm
                    sfft (v1.*) supported by @LeiHu
        
                    GitHub: https://github.com/thomasvrussell/sfft
                    Related Paper: https://arxiv.org/abs/2109.09334
                    
            ████████████████████████████████████████████████████████████████
            
            """
        
        if VERBOSE_LEVEL in [2]:
            print(SFFT_BANNER)
        
        # * Check Size of input images
        tmplst = [PixA_I.shape, PixA_J.shape, PixA_mI.shape, PixA_mI.shape]
        if len(set(tmplst)) > 1:
            raise Exception('MeLOn ERROR: Input images should have same size!')
        
        # * Subtraction Solution derived from input masked image-pair
        Solution = ElementalSFFTSubtract.ESS(PixA_I=PixA_mI, PixA_J=PixA_mJ, \
            SFFTConfig=SFFTConfig, SFFTSolution=None, Subtract=False, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)[0]
        
        # * Subtraction of the input image-pair (use above solution)
        PixA_DIFF = ElementalSFFTSubtract.ESS(PixA_I=PixA_I, PixA_J=PixA_J, \
            SFFTConfig=SFFTConfig, SFFTSolution=Solution, Subtract=True, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, VERBOSE_LEVEL=VERBOSE_LEVEL)[1]
        
        # * Identify propagated contamination region through convolving I
        ContamMask_CI = None
        if ContamMask_I is not None:
            DB = SFFTConfig[0]['DB']
            Fpq = int((DB+1)*(DB+2)/2)
            
            tSolution = Solution.copy()
            tSolution[-Fpq:] = 0.0

            _tmpI = ContamMask_I.astype(np.float64)
            _tmpJ = np.zeros(PixA_J.shape).astype(np.float64)
            _tmpD = ElementalSFFTSubtract.ESS(PixA_I=_tmpI, PixA_J=_tmpJ, SFFTConfig=SFFTConfig, \
                SFFTSolution=tSolution, Subtract=True, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
                NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, \
                VERBOSE_LEVEL=VERBOSE_LEVEL)[1]
            
            FTHRESH = -0.001  # emperical value
            ContamMask_CI = _tmpD < FTHRESH
        
        return Solution, PixA_DIFF, ContamMask_CI
