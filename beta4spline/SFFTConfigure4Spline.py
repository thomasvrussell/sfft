import os
import sys
import numpy as np
from scipy.interpolate import BSpline

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class SingleSFFTConfigure_Numpy:
    @staticmethod
    def SSCN(NX, NY, KerHW, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, NUM_CPU_THREADS_4SUBTRACT=8):

        import numba as nb
        nb.set_num_threads = NUM_CPU_THREADS_4SUBTRACT

        N0, N1 = int(NX), int(NY)
        w0, w1 = int(KerHW), int(KerHW)
        DK, DB = int(KerPolyOrder), int(BGPolyOrder)
        MaxThreadPerB = 8     # FIXME USER-DEFINED
        SFFTParam_dict = {}
        SFFTModule_dict = {}
        
        if DK not in [0, 1, 2, 3]:
            # NOTE DK = 0 means a spatial invariant matching-kernel.
            sys.exit('MeLOn ERROR: Input KerPolyOrder should be 0 / 1 / 2 / 3 !')
        if DB not in [0, 1, 2, 3]:
            # NOTE DB = 0 means a flat differential-background, however, we cannot kill the sky term in our script.
            sys.exit('MeLOn ERROR: Input BGPolyOrder should be 0 / 1 / 2 / 3 !')
        if (N0 < MaxThreadPerB) or (N1 < MaxThreadPerB):
            sys.exit('MeLOn ERROR: Input Image has dramatically small size !')

        print('\n  --//--//--//--//-- TRIGGER SFFT COMPILATION --//--//--//--//-- ')
        print('\n  ---//--- KerPolyOrder %d | BGPolyOrder %d | KerHW [%d] ---//--- ' %(DK, DB, w0))

        # * Make a dictionary for SFFT parameters
        L0 = 2*w0+1                       # matching-kernel XSize
        L1 = 2*w1+1                       # matching-kernel YSize
        Fab = L0 * L1                     # dof for index 𝛼𝛽
        #Fij = int((DK+1)*(DK+2)/2)        # dof for matching-kernel polynomial index ij 
        Fij = 36                          # BiQuadratic B-Spline with Ndiv=4, SplD=2
        Fpq = int((DB+1)*(DB+2)/2)        # dof for differential-background polynomial index pq
        SCALE = np.float64(1/(N0*N1))     # Scale of Image-Size
        SCALE_L = np.float64(1/SCALE)     # Reciprocal Scale of Image Size

        NEQ = Fij*Fab+Fpq                         # Linear-System side-length
        Fijab = Fij * Fab                         # Linear-System Major side-length
        NEQ_FSfree = NEQ - (Fij - 1)              # Forbidden-Stripes-Free Linear-System side-length

        FOMG, FGAM, FTHE = Fij**2, Fij*Fpq, Fij   # OMG / GAM / THE has shape (dof, N0, N1)
        FPSI, FPHI, FDEL = Fpq*Fij, Fpq**2, Fpq   # PSI / PHI / DEL has shape (dof, N0, N1)

        SFFTParam_dict['N0'] = N0
        SFFTParam_dict['N1'] = N1
        SFFTParam_dict['w0'] = w0
        SFFTParam_dict['w1'] = w1
        SFFTParam_dict['DK'] = DK
        SFFTParam_dict['DB'] = DB
        SFFTParam_dict['ConstPhotRatio'] = ConstPhotRatio
        SFFTParam_dict['MaxThreadPerB'] = MaxThreadPerB

        SFFTParam_dict['L0'] = L0
        SFFTParam_dict['L1'] = L1
        SFFTParam_dict['Fab'] = Fab
        SFFTParam_dict['Fij'] = Fij
        SFFTParam_dict['Fpq'] = Fpq
        SFFTParam_dict['SCALE'] = SCALE
        SFFTParam_dict['SCALE_L'] = SCALE_L
        SFFTParam_dict['NEQ'] = NEQ
        SFFTParam_dict['Fijab'] = Fijab
        SFFTParam_dict['NEQ_FSfree'] = NEQ_FSfree

        SFFTParam_dict['FOMG'] = FOMG
        SFFTParam_dict['FGAM'] = FGAM
        SFFTParam_dict['FTHE'] = FTHE
        SFFTParam_dict['FPSI'] = FPSI
        SFFTParam_dict['FPHI'] = FPHI
        SFFTParam_dict['FDEL'] = FDEL

        # * Load SFFT Numpy-based modules
        # ************************************ SpatialCoor.py ************************************ #

        # ** produce spatial coordinate X/Y/oX/oY-map
        _strdec = 'Tuple((i4[:,:], i4[:,:], f8[:,:], f8[:,:]))(i4[:,:], i4[:,:], f8[:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def SpatialCoor(PixA_X, PixA_Y, PixA_CX, PixA_CY):

            #assert PixA_X.dtype == np.int32 and PixA_X.shape == (N0, N1)
            #assert PixA_Y.dtype == np.int32 and PixA_Y.shape == (N0, N1)
            #assert PixA_CX.dtype == np.float64 and PixA_CX.shape == (N0, N1)
            #assert PixA_CY.dtype == np.float64 and PixA_CY.shape == (N0, N1)

            for ROW in nb.prange(N0):
                for COL in nb.prange(N1):
                    PixA_X[ROW, COL] = ROW
                    PixA_Y[ROW, COL] = COL
                    PixA_CX[ROW, COL] = (float(ROW) + 1.0) / N0   # NOTE UPDATE
                    PixA_CY[ROW, COL] = (float(COL) + 1.0) / N1   # NOTE UPDATE

            return PixA_X, PixA_Y, PixA_CX, PixA_CY

        SFFTModule_dict['SpatialCoor'] = SpatialCoor
        # ************************************ SpatialPoly.py ************************************ #

        def GenBasis(NPIX, Ndiv, SplD):
            PIXCOOR = (1.0+np.arange(NPIX))/NPIX  # Fortune & Normalized
            iKnotVector = np.linspace(0.5/NPIX, (NPIX+0.5)/NPIX, Ndiv+1)
            KnotVector = np.concatenate(([iKnotVector[0]]*SplD, \
                iKnotVector, [iKnotVector[-1]]*SplD))  # clamped knots
            NCP = len(KnotVector)-SplD-1   # control points (coefficients), m = n+p+1 
            BSplBASIS = []
            for cidx in range(NCP):
                Coeff = (np.arange(NCP)==cidx).astype(float)
                BSplBaseFunc = BSpline(t=KnotVector, c=Coeff, k=SplD, extrapolate=False)
                BSplBASIS.append(BSplBaseFunc(PIXCOOR))
            BSplBASIS = np.array(BSplBASIS)
            return PIXCOOR, BSplBASIS
        
        BiBSplBASIS = []
        PIXCOOR_X, BSplBASIS_X = GenBasis(NPIX=N0, Ndiv=4, SplD=2)
        PIXCOOR_Y, BSplBASIS_Y = GenBasis(NPIX=N1, Ndiv=4, SplD=2)
        for i in range(BSplBASIS_X.shape[0]):
            for j in range(BSplBASIS_Y.shape[0]):
                _tmp = np.tensordot(BSplBASIS_X[i], BSplBASIS_Y[j], axes=0)
                BiBSplBASIS.append(_tmp)
        BiBSplBASIS = np.array(BiBSplBASIS)

        # ** produce Tij, Iij, Tpq
        _strdec = 'Tuple((f8[:,:,:], f8[:,:,:], f8[:,:,:]))' + \
            '(i4[:,:], i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def SpatialPoly(REF_ij, REF_pq, PixA_CX, PixA_CY, PixA_I, SPixA_Tij, SPixA_Iij, SPixA_Tpq):
            
            #assert REF_ij.dtype == np.int32 and REF_ij.shape == (Fij, 2)
            #assert REF_pq.dtype == np.int32 and REF_pq.shape == (Fpq, 2)
            
            #assert PixA_CX.dtype == np.float64 and PixA_CX.shape == (N0, N1)
            #assert PixA_CY.dtype == np.float64 and PixA_CY.shape == (N0, N1)
            #assert PixA_I.dtype == np.float64 and PixA_I.shape == (N0, N1)
            
            #assert SPixA_Tij.dtype == np.float64 and SPixA_Tij.shape == (Fij, N0, N1)
            #assert SPixA_Iij.dtype == np.float64 and SPixA_Iij.shape == (Fij, N0, N1)
            #assert SPixA_Tpq.dtype == np.float64 and SPixA_Tpq.shape == (Fpq, N0, N1)
            
            """
            for ij in nb.prange(Fij):
                i, j = REF_ij[ij]
                PixA_kpoly = np.power(PixA_CX, i) * np.power(PixA_CY, j)
                SPixA_Tij[ij, :, :] = PixA_kpoly   # NOTE UPDATE
                SPixA_Iij[ij, :, :] = PixA_I * PixA_kpoly   # NOTE UPDATE
            
            for pq in nb.prange(Fpq):
                p, q = REF_pq[pq]
                PixA_bpoly = np.power(PixA_CX, p) * np.power(PixA_CY, q)
                SPixA_Tpq[pq, :, :] = PixA_bpoly   # NOTE UPDATE
            
            """

            for ij in nb.prange(Fij):
                PixA_kpoly = BiBSplBASIS[ij]
                SPixA_Tij[ij, :, :] = PixA_kpoly   # NOTE UPDATE
                SPixA_Iij[ij, :, :] = PixA_I * PixA_kpoly   # NOTE UPDATE
            
            #for pq in nb.prange(Fpq):
            #    PixA_bpoly = BiBSplBASIS[pq]
            #    SPixA_Tpq[pq, :, :] = PixA_bpoly   # NOTE UPDATE
            
            for pq in nb.prange(Fpq):
                p, q = REF_pq[pq]
                PixA_bpoly = np.power(PixA_CX, p) * np.power(PixA_CY, q)
                SPixA_Tpq[pq, :, :] = PixA_bpoly   # NOTE UPDATE
            
            return SPixA_Tij, SPixA_Iij, SPixA_Tpq

        SFFTModule_dict['SpatialPoly'] = SpatialPoly
        # ************************************ HadProd_OMG.py & FillLS_OMG.py ************************************ #

        # ** Hadamard Product [𝛀]
        _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_OMG(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, HpOMG):
            
            #assert SREF_iji0j0.dtype == np.int32 and SREF_iji0j0.shape == (FOMG, 2)
            #assert SPixA_FIij.dtype == np.complex128 and SPixA_FIij.shape == (Fij, N0, N1)
            #assert SPixA_CFIij.dtype == np.complex128 and SPixA_CFIij.shape == (Fij, N0, N1)
            #assert HpOMG.dtype == np.complex128 and HpOMG.shape == (FOMG, N0, N1)

            for i8j8ij in nb.prange(FOMG):
                i8j8, ij = SREF_iji0j0[i8j8ij]
                HpOMG[i8j8ij, :, :] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]   # NOTE UPDATE

            return HpOMG

        # ** Fill Linear-System [𝛀]
        _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def FillLS_OMG(SREF_ijab, REF_ab, PreOMG, LHMAT):
            
            #assert SREF_ijab.dtype == np.int32 and SREF_ijab.shape == (Fijab, 2)
            #assert REF_ab.dtype == np.int32 and REF_ab.shape == (Fab, 2)
            #assert PreOMG.dtype == np.float64 and PreOMG.shape == (FOMG, N0, N1)
            #assert LHMAT.dtype == np.float64 and LHMAT.shape == (NEQ, NEQ)
            
            for ROW in nb.prange(Fijab):
                for COL in nb.prange(Fijab):
                    
                    # *** analysze index
                    i8j8, a8b8 = SREF_ijab[ROW]
                    ij, ab = SREF_ijab[COL]

                    a8, b8 = REF_ab[a8b8]
                    a, b = REF_ab[ab]
                    idx = i8j8 * Fij + ij
                    
                    # *** define Mod_N0(rho), Mod_N1(eps)
                    tmp = int(np.fmod(float(a8), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MODa8 = tmp
                    
                    tmp = int(np.fmod(float(b8), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MODb8 = tmp

                    tmp = int(np.fmod(float(-a), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MOD_a = tmp

                    tmp = int(np.fmod(float(-b), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MOD_b = tmp

                    tmp = int(np.fmod(float(a8-a), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MODa8_a = tmp

                    tmp = int(np.fmod(float(b8-b), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MODb8_b = tmp
                
                    # *** fill linear system [A-component]
                    if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                        LHMAT[ROW, COL] = - PreOMG[idx, MODa8, MODb8] \
                                          - PreOMG[idx, MOD_a, MOD_b] \
                                          + PreOMG[idx, MODa8_a, MODb8_b] \
                                          + PreOMG[idx, 0, 0]   # NOTE UPDATE

                    if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                        LHMAT[ROW, COL] = PreOMG[idx, MOD_a, MOD_b] - PreOMG[idx, 0, 0]   # NOTE UPDATE

                    if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                        LHMAT[ROW, COL] = PreOMG[idx, MODa8, MODb8] - PreOMG[idx, 0, 0]   # NOTE UPDATE

                    if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                        LHMAT[ROW, COL] = PreOMG[idx, 0, 0]   # NOTE UPDATE
                    
            return LHMAT

        SFFTModule_dict['HadProd_OMG'] = HadProd_OMG
        SFFTModule_dict['FillLS_OMG'] = FillLS_OMG
        # ************************************ HadProd_GAM.py & FillLS_GAM.py ************************************ #

        # ** Hadamard Product [𝜦]
        _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_GAM(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, HpGAM):
            
            #assert SREF_ijpq.dtype == np.int32 and SREF_ijpq.shape == (FGAM, 2)
            #assert SPixA_FIij.dtype == np.complex128 and SPixA_FIij.shape == (Fij, N0, N1)
            #assert SPixA_CFTpq.dtype == np.complex128 and SPixA_CFTpq.shape == (Fpq, N0, N1)
            #assert HpGAM.dtype == np.complex128 and HpGAM.shape == (FGAM, N0, N1)
            
            for i8j8pq in nb.prange(FGAM):
                i8j8, pq = SREF_ijpq[i8j8pq]
                HpGAM[i8j8pq, :, :] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]   # NOTE UPDATE

            return HpGAM

        # ** Fill Linear-System [𝜦]
        _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def FillLS_GAM(SREF_ijab, REF_ab, PreGAM, LHMAT):
            
            #assert SREF_ijab.dtype == np.int32 and SREF_ijab.shape == (Fijab, 2)
            #assert REF_ab.dtype == np.int32 and REF_ab.shape == (Fab, 2)
            #assert PreGAM.dtype == np.float64 and PreGAM.shape == (FGAM, N0, N1)
            #assert LHMAT.dtype == np.float64 and LHMAT.shape == (NEQ, NEQ)
            
            for ROW in nb.prange(Fijab):
                for COL in nb.prange(Fpq):
                    
                    # *** analysze index
                    i8j8, a8b8 = SREF_ijab[ROW]
                    pq = COL
                    
                    a8, b8 = REF_ab[a8b8]
                    idx = i8j8 * Fpq + pq
                    cCOL = Fijab + COL       # add offset
                    
                    # *** define Mod_N0(rho), Mod_N1(eps)
                    tmp = int(np.fmod(float(a8), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MODa8 = tmp

                    tmp = int(np.fmod(float(b8), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MODb8 = tmp
                    
                    # *** fill linear system [B-component]
                    if (a8 != 0 or b8 != 0):
                        LHMAT[ROW, cCOL] = PreGAM[idx, MODa8, MODb8] - PreGAM[idx, 0, 0]   # NOTE UPDATE
                    
                    if (a8 == 0 and b8 == 0):
                        LHMAT[ROW, cCOL] = PreGAM[idx, 0, 0]   # NOTE UPDATE
            
            return LHMAT

        SFFTModule_dict['HadProd_GAM'] = HadProd_GAM
        SFFTModule_dict['FillLS_GAM'] = FillLS_GAM
        # ************************************ HadProd_PSI.py & FillLS_PSI.py ************************************ #

        # ** Hadamard Product [𝜳]
        _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_PSI(SREF_pqij, SPixA_CFIij, SPixA_FTpq, HpPSI):
            
            #assert SREF_pqij.dtype == np.int32 and SREF_pqij.shape == (FPSI, 2)
            #assert SPixA_CFIij.dtype == np.complex128 and SPixA_CFIij.shape == (Fij, N0, N1)
            #assert SPixA_FTpq.dtype == np.complex128 and SPixA_FTpq.shape == (Fpq, N0, N1)
            #assert HpPSI.dtype == np.complex128 and HpPSI.shape == (FPSI, N0, N1)

            for p8q8ij in nb.prange(FPSI):
                p8q8, ij = SREF_pqij[p8q8ij]
                HpPSI[p8q8ij, :, :] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]   # NOTE UPDATE

            return HpPSI

        # ** Fill Linear-System [𝜳]
        _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def FillLS_PSI(SREF_ijab, REF_ab, PrePSI, LHMAT):
            
            #assert SREF_ijab.dtype == np.int32 and SREF_ijab.shape == (Fijab, 2)
            #assert REF_ab.dtype == np.int32 and REF_ab.shape == (Fab, 2)
            #assert PrePSI.dtype == np.float64 and PrePSI.shape == (FPSI, N0, N1)
            #assert LHMAT.dtype == np.float64 and LHMAT.shape == (NEQ, NEQ)

            for ROW in nb.prange(Fpq):
                for COL in nb.prange(Fijab):
                    
                    # *** analysze index
                    cROW = Fijab + ROW       # add offset
                    p8q8 = ROW
                    
                    ij, ab = SREF_ijab[COL]
                    a, b = REF_ab[ab]
                    idx = p8q8 * Fij + ij
                    
                    # *** define Mod_N0(rho), Mod_N1(eps)
                    tmp = int(np.fmod(float(-a), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MOD_a = tmp

                    tmp = int(np.fmod(float(-b), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MOD_b = tmp
                    
                    # *** fill linear system [B#-component]
                    if (a != 0 or b != 0):
                        LHMAT[cROW, COL] = PrePSI[idx, MOD_a, MOD_b] - PrePSI[idx, 0, 0]   # NOTE UPDATE

                    if (a == 0 and b == 0):
                        LHMAT[cROW, COL] = PrePSI[idx, 0, 0]   # NOTE UPDATE
                    
            return LHMAT

        SFFTModule_dict['HadProd_PSI'] = HadProd_PSI
        SFFTModule_dict['FillLS_PSI'] = FillLS_PSI
        # ************************************ HadProd_PHI.py & FillLS_PHI.py ************************************ #

        # ** Hadamard Product [𝚽]
        _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, HpPHI):
            
            #assert SREF_pqp0q0.dtype == np.int32 and SREF_pqp0q0.shape == (FPHI, 2)
            #assert SPixA_FTpq.dtype == np.complex128 and SPixA_FTpq.shape == (Fpq, N0, N1)
            #assert SPixA_CFTpq.dtype == np.complex128 and SPixA_CFTpq.shape == (Fpq, N0, N1)
            #assert HpPHI.dtype == np.complex128 and HpPHI.shape == (FPHI, N0, N1)
            
            for p8q8pq in nb.prange(FPHI):
                p8q8, pq = SREF_pqp0q0[p8q8pq]
                HpPHI[p8q8pq, :, :] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]   # NOTE UPDATE

            return HpPHI

        # ** Fill Linear-System [𝚽]
        _strdec = 'f8[:,:](f8[:,:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def FillLS_PHI(PrePHI, LHMAT):
            
            #assert PrePHI.dtype == np.float64 and PrePHI.shape == (FPHI, N0, N1)
            #assert LHMAT.dtype == np.float64 and LHMAT.shape == (NEQ, NEQ)
            
            for ROW in nb.prange(Fpq):
                for COL in nb.prange(Fpq):
                    
                    # *** analysze index
                    cROW = Fijab + ROW       # add offset
                    cCOL = Fijab + COL       # add offset
                    
                    p8q8 = ROW
                    pq = COL
                    idx = p8q8 * Fpq + pq
                    
                    # *** fill linear system [C-component]
                    LHMAT[cROW, cCOL] = PrePHI[idx, 0, 0]   # NOTE UPDATE

            return LHMAT

        SFFTModule_dict['HadProd_PHI'] = HadProd_PHI
        SFFTModule_dict['FillLS_PHI'] = FillLS_PHI
        # ************************************ HadProd_THE.py & FillLS_THE.py ************************************ #

        # ** Hadamard Product [𝚯]
        _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_THE(SPixA_FIij, PixA_CFJ, HpTHE):
            
            #assert SPixA_FIij.dtype == np.complex128 and SPixA_FIij.shape == (Fij, N0, N1)
            #assert PixA_CFJ.dtype == np.complex128 and PixA_CFJ.shape == (N0, N1)
            #assert HpTHE.dtype == np.complex128 and HpTHE.shape == (FTHE, N0, N1)
            
            for i8j8 in nb.prange(FTHE):
                HpTHE[i8j8, :, :] = PixA_CFJ * SPixA_FIij[i8j8]   # NOTE UPDATE

            return HpTHE

        # ** Fill Linear-System [𝚯]
        _strdec = 'f8[:](i4[:,:], i4[:,:], f8[:,:,:], f8[:])'
        @nb.njit(_strdec, parallel=True)
        def FillLS_THE(SREF_ijab, REF_ab, PreTHE, RHb):
            
            #assert SREF_ijab.dtype == np.int32 and SREF_ijab.shape == (Fijab, 2)
            #assert REF_ab.dtype == np.int32 and REF_ab.shape == (Fab, 2)
            #assert PreTHE.dtype == np.float64 and PreTHE.shape == (FTHE, N0, N1)
            #assert RHb.dtype == np.float64 and RHb.shape == (NEQ, )
            
            COL = 0   # trivial
            for ROW in nb.prange(Fijab):
                
                # *** analysze index
                i8j8, a8b8 = SREF_ijab[ROW]
                a8, b8 = REF_ab[a8b8]
                idx = i8j8
                
                # *** define Mod_N0(rho), Mod_N1(eps)
                tmp = int(np.fmod(float(a8), float(N0)))
                if tmp < 0.0: tmp += N0
                MODa8 = tmp

                tmp = int(np.fmod(float(b8), float(N1)))
                if tmp < 0.0: tmp += N1
                MODb8 = tmp
                
                # *** fill linear system [D-component]
                if (a8 != 0 or b8 != 0):
                    RHb[ROW] = PreTHE[idx, MODa8, MODb8] - PreTHE[idx, 0, 0]   # NOTE UPDATE

                if (a8 == 0 and b8 == 0):
                    RHb[ROW] = PreTHE[idx, 0, 0]   # NOTE UPDATE
            
            return RHb

        SFFTModule_dict['HadProd_THE'] = HadProd_THE
        SFFTModule_dict['FillLS_THE'] = FillLS_THE
        # ************************************ HadProd_DEL.py & FillLS_DEL.py ************************************ #

        # ** Hadamard Product [𝚫]
        _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_DEL(SPixA_FTpq, PixA_CFJ, HpDEL):
            
            #assert SPixA_FTpq.dtype == np.complex128 and SPixA_FTpq.shape == (Fpq, N0, N1)
            #assert PixA_CFJ.dtype == np.complex128 and PixA_CFJ.shape == (N0, N1)
            #assert HpDEL.dtype == np.complex128 and HpDEL.shape == (FDEL, N0, N1)
            
            for p8q8 in nb.prange(FDEL):
                HpDEL[p8q8, :, :] = PixA_CFJ * SPixA_FTpq[p8q8]   # NOTE UPDATE

            return HpDEL

        # ** Fill Linear-System [𝚫]
        _strdec = 'f8[:](f8[:,:,:], f8[:])'
        @nb.njit(_strdec, parallel=True)
        def FillLS_DEL(PreDEL, RHb):
            
            #assert PreDEL.dtype == np.float64 and PreDEL.shape == (FDEL, N0, N1)
            #assert RHb.dtype == np.float64 and RHb.shape == (NEQ, )
            
            COL = 0   # trivial
            for ROW in nb.prange(Fpq):
                
                # *** analysze index
                cROW = Fijab + ROW       # add offset
                idx = ROW                # i.e. p8q8
                
                # *** fill linear system [E-component]
                RHb[cROW] = PreDEL[idx, 0, 0]   # NOTE UPDATE
            
            return RHb

        SFFTModule_dict['HadProd_DEL'] = HadProd_DEL
        SFFTModule_dict['FillLS_DEL'] = FillLS_DEL
        # ************************************ Remove_LSFStripes.py ************************************ #

        # ** Remove Forbidden Stripes in Linear-System
        _strdec = 'f8[:,:](f8[:,:], i4[:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def Remove_LSFStripes(LHMAT, IDX_nFS, LHMAT_FSfree):
            
            #assert LHMAT.dtype == np.float64 and LHMAT.shape == (NEQ, NEQ)
            #assert IDX_nFS.dtype == np.int32 and IDX_nFS.shape == (NEQ_FSfree, )
            #assert LHMAT_FSfree.dtype == np.float64 and LHMAT_FSfree.shape == (NEQ_FSfree, NEQ_FSfree)
            
            for ROW in nb.prange(NEQ_FSfree):
                for COL in nb.prange(NEQ_FSfree):
                    ROW_nFS = IDX_nFS[ROW]
                    COL_nFS = IDX_nFS[COL]
                    LHMAT_FSfree[ROW, COL] = LHMAT[ROW_nFS, COL_nFS]   # NOTE UPDATE
            
            return LHMAT_FSfree

        SFFTModule_dict['Remove_LSFStripes'] = Remove_LSFStripes
        # ************************************ Extend_Solution.py ************************************ #

        # ** Extend Solution from Forbidden-Stripes-Free Linear-System
        _strdec = 'f8[:](f8[:], i4[:], f8[:])'
        @nb.njit(_strdec, parallel=True)
        def Extend_Solution(Solution_FSfree, IDX_nFS, Solution):
            
            #assert Solution_FSfree.dtype == np.float64 and Solution_FSfree.shape == (NEQ_FSfree, )
            #assert IDX_nFS.dtype == np.int32 and IDX_nFS.shape == (NEQ_FSfree, )
            #assert Solution.dtype == np.float64 and Solution.shape == (NEQ, )
            
            for ROW in nb.prange(NEQ_FSfree):
                ROW_nFS = IDX_nFS[ROW]
                Solution[ROW_nFS] = Solution_FSfree[ROW]   # NOTE UPDATE
            
            return Solution

        SFFTModule_dict['Extend_Solution'] = Extend_Solution
        # ************************************ Construct_FDIFF.py ************************************ #

        _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], c16[:,:])'
        @nb.njit(_strdec, parallel=True)
        def Construct_FDIFF(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, PixA_FDIFF):
            
            #assert SREF_ijab.dtype == np.int32 and SREF_ijab.shape == (Fijab, 2)
            #assert REF_ab.dtype == np.int32 and REF_ab.shape == (Fab, 2)
            
            #assert a_ijab.dtype == np.complex128 and a_ijab.shape == (Fijab, )
            #assert SPixA_FIij.dtype == np.complex128 and SPixA_FIij.shape == (Fij, N0, N1)
            #assert Kab_Wla.dtype == np.complex128 and Kab_Wla.shape == (L0, N0, N1)
            #assert Kab_Wmb.dtype == np.complex128 and Kab_Wmb.shape == (L1, N0, N1)
            #assert b_pq.dtype == np.complex128 and b_pq.shape == (Fpq, )
            #assert SPixA_FTpq.dtype == np.complex128 and SPixA_FTpq.shape == (Fpq, N0, N1)
            
            #assert PixA_FJ.dtype == np.complex128 and PixA_FJ.shape == (N0, N1)
            #assert PixA_FDIFF.dtype == np.complex128 and PixA_FDIFF.shape == (N0, N1)

            ZERO_C = 0.0 + 0.0j
            ONE_C = 1.0 + 0.0j
            SCALE_C = SCALE + 0.0j

            for ROW in nb.prange(N0):
                for COL in nb.prange(N1):
                    
                    PVAL = ZERO_C
                    for ab in range(Fab):
                        a, b = REF_ab[ab]

                        if (a == 0 and b == 0):
                            PVAL_FKab = SCALE_C

                        if (a != 0 or b != 0):
                            PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)

                        for ij in range(Fij):
                            ijab = ij * Fab + ab
                            PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                        
                    for pq in range(Fpq):
                        PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]

                    PixA_FDIFF[ROW, COL] = PixA_FJ[ROW, COL] - PVAL
            
            return PixA_FDIFF

        SFFTModule_dict['Construct_FDIFF'] = Construct_FDIFF

        SFFTConfig = (SFFTParam_dict, SFFTModule_dict)
        print('\n  --//--//--//--//-- EXIT SFFT COMPILATION --//--//--//--//-- ')

        return SFFTConfig

class SingleSFFTConfigure:
    @staticmethod
    def SSC(NX, NY, KerHW, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
        BACKEND_4SUBTRACT='Pycuda', CUDA_DEVICE_4SUBTRACT='0', NUM_CPU_THREADS_4SUBTRACT=8):

        """
        # Arguments:
        # a) NX, NY: Image Size (pixels) along X / Y axis.
        # b) KerHW: Kernel half width for compilation.
        # c) KerPolyOrder: The order of Polynomial Variation for PSF-Matching Kernel.
        # d) BGPolyOrder: The order of Polynomial Variation for Differential Background.
        # e) ConstPhotRatio: Use a constant photometric ratio in image subtraction ?
        # f) BACKEND_4SUBTRACT: Which backend would you like to perform SFFT on ?
        # g) CUDA_DEVICE_4SUBTRACT (BACKEND_4SUBTRACT = Pycuda / Cupy): Which GPU device would you want to perform SFFT on ?
        # h) NUM_CPU_THREADS_4SUBTRACT (BACKEND_4SUBTRACT = Numpy): How many CPU threads would you want to perform SFFT on ?
        
        """

        # * ONLY Numpy backend available now.
        if BACKEND_4SUBTRACT == 'Numpy':
            SFFTConfig = SingleSFFTConfigure_Numpy.SSCN(NX=NX, NY=NY, KerHW=KerHW, \
                KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
                NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
        
        return SFFTConfig

