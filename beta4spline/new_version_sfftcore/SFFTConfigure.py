import numpy as np
# version: Apr 17, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class SingleSFFTConfigure_Cupy:
    @staticmethod
    def SSCC(NX, NY, KerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        ConstPhotRatio=True, MaxThreadPerB=8, VERBOSE_LEVEL=2):

        import cupy as cp

        N0, N1 = int(NX), int(NY)
        w0, w1 = int(KerHW), int(KerHW)
        DK, DB = int(KerSpDegree), int(BkgSpDegree)

        SCALE = np.float64(1/(N0*N1))     # Scale of Image-Size
        SCALE_L = np.float64(1/SCALE)     # Reciprocal Scale of Image-Size

        # NOTE: SFFT allows for Polynomial or B-Spline spatial variation
        assert KerSpType in ['Polynomial', 'B-Spline']
        assert BkgSpType in ['Polynomial', 'B-Spline']

        if KerSpType == 'B-Spline' and KerSpDegree == 0:
            assert len(KerIntKnotX) == 0  # otherwise, discontinuity
            assert len(KerIntKnotY) == 0  # otherwise, discontinuity

        if BkgSpType == 'B-Spline' and BkgSpDegree == 0:
            assert len(BkgIntKnotX) == 0  # otherwise, discontinuity
            assert len(BkgIntKnotY) == 0  # otherwise, discontinuity
        
        # NOTE: DK = 0 means a spatial invariant matching kernel
        # NOTE: DB = 0 means a flat differential background
        assert DK >= 0 and DB >= 0

        # input image should not has dramatically small size
        assert N0 > MaxThreadPerB and N1 > MaxThreadPerB
        
        if VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- TRIGGER SFFT COMPILATION [Cupy] --//--//--//--//-- ')
            if KerSpType == 'Polynomial':
                print('\n ---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(KerSpDegree, KerHW))
            if KerSpType == 'B-Spline':
                print('\n ---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                    %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree, KerHW))
            if BkgSpType == 'Polynomial':
                print('\n ---//--- Polynomial Background | BkgSpDegree %d ---//---' %BkgSpDegree)
            if BkgSpType == 'B-Spline':
                print('\n ---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), BkgSpDegree))

        # * Make a dictionary for SFFT parameters
        L0 = 2*w0+1                                      # matching-kernel XSize
        L1 = 2*w1+1                                      # matching-kernel YSize
        Fab = L0*L1                                      # dof for index ab

        if KerSpType == 'Polynomial':
            Fi, Fj = -1, -1                              # not independent
            Fij = int((DK+1)*(DK+2)/2)                   # dof for matching-kernel polynomial index ij 
        
        if KerSpType == 'B-Spline':
            Fi = len(KerIntKnotX) + KerSpDegree + 1      # dof for matching-kernel B-spline index i (control points/coefficients)
            Fj = len(KerIntKnotY) + KerSpDegree + 1      # dof for matching-kernel B-spline index j (control points/coefficients)
            Fij = Fi*Fj                                  # dof for matching-kernel B-spline index ij
        
        if BkgSpType == 'Polynomial':
            Fp, Fq = -1, -1                              # not independent 
            Fpq = int((DB+1)*(DB+2)/2)                   # dof for diff-background polynomial index pq 
        
        if BkgSpType == 'B-Spline':
            Fp = len(BkgIntKnotX) + BkgSpDegree + 1      # dof for diff-background B-spline index p (control points/coefficients)
            Fq = len(BkgIntKnotY) + BkgSpDegree + 1      # dof for diff-background B-spline index q (control points/coefficients)  
            Fpq = Fp*Fq                                  # dof for diff-background B-spline index pq

        Fijab = Fij*Fab                                  # Linear-System Major side-length
        FOMG, FGAM, FTHE = Fij**2, Fij*Fpq, Fij          # OMG / GAM / THE has shape (dof, N0, N1)
        FPSI, FPHI, FDEL = Fpq*Fij, Fpq**2, Fpq          # PSI / PHI / DEL has shape (dof, N0, N1)
        NEQ = Fij*Fab+Fpq                                # Linear-System side-length
        NEQt = NEQ - Fij + 1                             # tweaked Linear-System side-length for constant ratio

        SFFTParam_dict = {}
        SFFTParam_dict['KerHW'] = KerHW
        SFFTParam_dict['KerSpType'] = KerSpType
        SFFTParam_dict['KerSpDegree'] = KerSpDegree
        SFFTParam_dict['KerIntKnotX'] = KerIntKnotX
        SFFTParam_dict['KerIntKnotY'] = KerIntKnotY
        SFFTParam_dict['BkgSpType'] = BkgSpType
        SFFTParam_dict['BkgSpDegree'] = BkgSpDegree
        SFFTParam_dict['BkgIntKnotX'] = BkgIntKnotX
        SFFTParam_dict['BkgIntKnotY'] = BkgIntKnotY
        SFFTParam_dict['ConstPhotRatio'] = ConstPhotRatio

        SFFTParam_dict['N0'] = N0    # a.k.a, NX
        SFFTParam_dict['N1'] = N1    # a.k.a, NY
        SFFTParam_dict['w0'] = w0    # a.k.a, KerHW
        SFFTParam_dict['w1'] = w1    # a.k.a, KerHW
        SFFTParam_dict['DK'] = DK    # a.k.a, KerSpDegree
        SFFTParam_dict['DB'] = DB    # a.k.a, BkgSpDegree

        SFFTParam_dict['MaxThreadPerB'] = MaxThreadPerB
        SFFTParam_dict['SCALE'] = SCALE
        SFFTParam_dict['SCALE_L'] = SCALE_L

        SFFTParam_dict['L0'] = L0
        SFFTParam_dict['L1'] = L1
        SFFTParam_dict['Fab'] = Fab
        SFFTParam_dict['Fi'] = Fi
        SFFTParam_dict['Fj'] = Fj
        SFFTParam_dict['Fij'] = Fij
        SFFTParam_dict['Fp'] = Fp
        SFFTParam_dict['Fq'] = Fq
        SFFTParam_dict['Fpq'] = Fpq
        SFFTParam_dict['Fijab'] = Fijab
        
        SFFTParam_dict['FOMG'] = FOMG
        SFFTParam_dict['FGAM'] = FGAM
        SFFTParam_dict['FTHE'] = FTHE
        SFFTParam_dict['FPSI'] = FPSI
        SFFTParam_dict['FPHI'] = FPHI
        SFFTParam_dict['FDEL'] = FDEL
        
        SFFTParam_dict['NEQ'] = NEQ
        SFFTParam_dict['NEQt'] = NEQt

        # * Load SFFT CUDA modules
        #   NOTE: Generally, a kernel function is defined without knowledge about Grid-Block-Thread Management.
        #         However, we need to know the size of threads per block if SharedMemory is called.

        # ************************************ SpatialCoord.cu ************************************ #

        SFFTModule_dict = {}
        # ** produce spatial coordinate X/Y/oX/oY-map
        _refdict = {'N0': N0, 'N1': N1}
        _funcstr = r"""
        extern "C" __global__ void kmain(int PixA_X_GPU[%(N0)s][%(N1)s], int PixA_Y_GPU[%(N0)s][%(N1)s], 
            double PixA_CX_GPU[%(N0)s][%(N1)s], double PixA_CY_GPU[%(N0)s][%(N1)s])
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
            int N0 = %(N0)s;
            int N1 = %(N1)s;
            
            if (ROW < N0 && COL < N1) {
                double cx = (ROW + 1.0) / N0;  
                double cy = (COL + 1.0) / N1;
                PixA_X_GPU[ROW][COL] = ROW;
                PixA_Y_GPU[ROW][COL] = COL;
                PixA_CX_GPU[ROW][COL] = cx;
                PixA_CY_GPU[ROW][COL] = cy;
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['SpatialCoord'] = _module

        # ************************************ KerSpatial.cu ************************************ #
        
        # ** produce Iij
        if KerSpType == 'Polynomial':
            
            _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij}
            _funcstr = r"""
            extern "C" __global__ void kmain(int REF_ij_GPU[%(Fij)s][2],
                double PixA_CX_GPU[%(N0)s][%(N1)s], double PixA_CY_GPU[%(N0)s][%(N1)s],
                double PixA_I_GPU[%(N0)s][%(N1)s], double SPixA_Iij_GPU[%(Fij)s][%(N0)s][%(N1)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
                int N0 = %(N0)s;
                int N1 = %(N1)s;
                int Fij = %(Fij)s;
                
                if (ROW < N0 && COL < N1) {
                    for(int ij = 0; ij < Fij; ++ij) {
                        int i = REF_ij_GPU[ij][0];
                        int j = REF_ij_GPU[ij][1];
                        double poly = pow(PixA_CX_GPU[ROW][COL], i) * pow(PixA_CY_GPU[ROW][COL], j);
                        SPixA_Iij_GPU[ij][ROW][COL] = PixA_I_GPU[ROW][COL] * poly;
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['KerSpatial'] = _module
        
        if KerSpType == 'B-Spline':
            
            _refdict = {'N0': N0, 'N1': N1, 'Fi': Fi, 'Fj': Fj, 'Fij': Fij}
            _funcstr = r"""
            extern "C" __global__ void kmain(int REF_ij_GPU[%(Fij)s][2],
                double KerSplBasisX_GPU[%(Fi)s][%(N0)s], double KerSplBasisY_GPU[%(Fj)s][%(N1)s],
                double PixA_I_GPU[%(N0)s][%(N1)s], double SPixA_Iij_GPU[%(Fij)s][%(N0)s][%(N1)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
                int N0 = %(N0)s;
                int N1 = %(N1)s;
                int Fij = %(Fij)s;
                
                if (ROW < N0 && COL < N1) {
                    for(int ij = 0; ij < Fij; ++ij) {
                        int i = REF_ij_GPU[ij][0];
                        int j = REF_ij_GPU[ij][1];
                        double spl = KerSplBasisX_GPU[i][ROW] * KerSplBasisY_GPU[j][COL];
                        SPixA_Iij_GPU[ij][ROW][COL] = PixA_I_GPU[ROW][COL] * spl;
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['KerSpatial'] = _module
        
        # ** produce Tpq
        if BkgSpType == 'Polynomial':

            _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq}
            _funcstr = r"""
            extern "C" __global__ void kmain(int REF_pq_GPU[%(Fpq)s][2],
                double PixA_CX_GPU[%(N0)s][%(N1)s], double PixA_CY_GPU[%(N0)s][%(N1)s],
                double SPixA_Tpq_GPU[%(Fpq)s][%(N0)s][%(N1)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
                int N0 = %(N0)s;
                int N1 = %(N1)s;
                int Fpq = %(Fpq)s;
                
                if (ROW < N0 && COL < N1) {
                    for(int pq = 0; pq < Fpq; ++pq) {
                        int p = REF_pq_GPU[pq][0];
                        int q = REF_pq_GPU[pq][1];
                        double poly_bterm = pow(PixA_CX_GPU[ROW][COL], p) * pow(PixA_CY_GPU[ROW][COL], q);
                        SPixA_Tpq_GPU[pq][ROW][COL] = poly_bterm;
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['BkgSpatial'] = _module
        
        if BkgSpType == 'B-Spline':

            _refdict = {'N0': N0, 'N1': N1, 'Fp': Fp, 'Fq': Fq, 'Fpq': Fpq}
            _funcstr = r"""
            extern "C" __global__ void kmain(int REF_pq_GPU[%(Fpq)s][2],
                double BkgSplBasisX_GPU[%(Fp)s][%(N0)s], double BkgSplBasisY_GPU[%(Fq)s][%(N1)s],
                double SPixA_Tpq_GPU[%(Fpq)s][%(N0)s][%(N1)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
                int N0 = %(N0)s;
                int N1 = %(N1)s;
                int Fpq = %(Fpq)s;
                
                if (ROW < N0 && COL < N1) {
                    for(int pq = 0; pq < Fpq; ++pq) {
                        int p = REF_pq_GPU[pq][0];
                        int q = REF_pq_GPU[pq][1];
                        double spl_kterm = BkgSplBasisX_GPU[p][ROW] * BkgSplBasisY_GPU[q][COL];
                        SPixA_Tpq_GPU[pq][ROW][COL] = spl_kterm;
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['BkgSpatial'] = _module

        # ************************************ HadProd_OMG.cu & FillLS_OMG.cu ************************************ #

        # ** Hadamard Product [Omega]
        _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FOMG': FOMG, 'ThreadPerB': MaxThreadPerB}
        _funcstr = r"""
        #include "cuComplex.h"
        extern "C" __global__ void kmain(int SREF_iji0j0_GPU[%(FOMG)s][2],
            cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
            cuDoubleComplex SPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
            cuDoubleComplex HpOMG_GPU[%(FOMG)s][%(N0)s][%(N1)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;

            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fij = %(Fij)s;
            int FOMG = %(FOMG)s;

            __shared__ cuDoubleComplex ShSPixA_FIij_GPU[%(Fij)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int ij = 0; ij < Fij; ++ij){
                    ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_FIij_GPU[ij][ROW][COL];
                }
            }

            __shared__ cuDoubleComplex ShSPixA_CFIij_GPU[%(Fij)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int ij = 0; ij < Fij; ++ij){
                    ShSPixA_CFIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_CFIij_GPU[ij][ROW][COL];
                }
            }
            __syncthreads();

            if (ROW < N0 && COL < N1) {
                for(int i8j8ij = 0; i8j8ij < FOMG; ++i8j8ij){
                    
                    int i8j8 = SREF_iji0j0_GPU[i8j8ij][0];
                    int ij = SREF_iji0j0_GPU[i8j8ij][1];

                    HpOMG_GPU[i8j8ij][ROW][COL] = cuCmul(ShSPixA_FIij_GPU[i8j8][threadIdx.x][threadIdx.y], 
                        ShSPixA_CFIij_GPU[ij][threadIdx.x][threadIdx.y]);
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
        SFFTModule_dict['HadProd_OMG'] = _module

        # ** Fill Linear-System [Omega]
        _refdict = {'N0': N0, 'N1': N1, 'Fijab': Fijab, 'Fab': Fab, 'Fij': Fij, 'NEQ': NEQ, 'FOMG': FOMG}
        _funcstr = r"""
        extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
            double PreOMG_GPU[%(FOMG)s][%(N0)s][%(N1)s], double LHMAT_GPU[%(NEQ)s][%(NEQ)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fij = %(Fij)s;
            int Fijab = %(Fijab)s;
            
            if (ROW < Fijab && COL < Fijab) {
                
                // INDEX Analysis
                int i8j8 = SREF_ijab_GPU[ROW][0];
                int a8b8 = SREF_ijab_GPU[ROW][1];
                int ij = SREF_ijab_GPU[COL][0];
                int ab = SREF_ijab_GPU[COL][1];

                int a8 = REF_ab_GPU[a8b8][0];
                int b8 = REF_ab_GPU[a8b8][1];
                int a = REF_ab_GPU[ab][0];
                int b = REF_ab_GPU[ab][1];
                int idx = i8j8 * Fij + ij;

                // Define Mod_N0(rho), Mod_N1(eps)
                float tmp = 0.0;
                tmp = fmod(float(a8), float(N0));
                if (tmp < 0.0) {tmp += float(N0);}
                int MODa8 = tmp;

                tmp = fmod(float(b8), float(N1));
                if (tmp < 0.0) {tmp += float(N1);}
                int MODb8 = tmp;

                tmp = fmod(float(-a), float(N0));
                if (tmp < 0.0) {tmp += float(N0);}
                int MOD_a = tmp;

                tmp = fmod(float(-b), float(N1));
                if (tmp < 0.0) {tmp += float(N1);}
                int MOD_b = tmp;

                tmp = fmod(float(a8-a), float(N0));
                if (tmp < 0.0) {tmp += float(N0);}
                int MODa8_a = tmp;

                tmp = fmod(float(b8-b), float(N1));
                if (tmp < 0.0) {tmp += float(N1);}
                int MODb8_b = tmp;

                // Fill Linear System [A-component]
                if ((a8 != 0 || b8 != 0) && (a != 0 || b != 0)) {
                    LHMAT_GPU[ROW][COL] = - PreOMG_GPU[idx][MODa8][MODb8]
                                          - PreOMG_GPU[idx][MOD_a][MOD_b] 
                                          + PreOMG_GPU[idx][MODa8_a][MODb8_b] 
                                          + PreOMG_GPU[idx][0][0];
                }

                if ((a8 == 0 && b8 == 0) && (a != 0 || b != 0)) {
                    LHMAT_GPU[ROW][COL] = PreOMG_GPU[idx][MOD_a][MOD_b] - PreOMG_GPU[idx][0][0];
                }

                if ((a8 != 0 || b8 != 0) && (a == 0 && b == 0)) {
                    LHMAT_GPU[ROW][COL] = PreOMG_GPU[idx][MODa8][MODb8] - PreOMG_GPU[idx][0][0];
                }

                if ((a8 == 0 && b8 == 0) && (a == 0 && b == 0)) {
                    LHMAT_GPU[ROW][COL] = PreOMG_GPU[idx][0][0];
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['FillLS_OMG'] = _module

        # ************************************ HadProd_GAM.cu & FillLS_GAM.cu ************************************ #

        # ** Hadamard Product [Gamma]
        _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FGAM': FGAM, 'ThreadPerB': MaxThreadPerB}
        _funcstr = r"""
        #include "cuComplex.h"
        extern "C" __global__ void kmain(int SREF_ijpq_GPU[%(FGAM)s][2], 
            cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
            cuDoubleComplex SPixA_CFTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
            cuDoubleComplex HpGAM_GPU[%(FGAM)s][%(N0)s][%(N1)s])
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;

            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fij = %(Fij)s;
            int Fpq = %(Fpq)s;
            int FGAM = %(FGAM)s;
            
            __shared__ cuDoubleComplex ShSPixA_FIij_GPU[%(Fij)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int ij = 0; ij < Fij; ++ij){
                    ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_FIij_GPU[ij][ROW][COL];
                }
            }

            __shared__ cuDoubleComplex ShSPixA_CFTpq_GPU[%(Fpq)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int pq = 0; pq < Fpq; ++pq){
                    ShSPixA_CFTpq_GPU[pq][threadIdx.x][threadIdx.y] = SPixA_CFTpq_GPU[pq][ROW][COL];
                }
            }
            __syncthreads();
            
            if (ROW < N0 && COL < N1) {        
                for(int i8j8pq = 0; i8j8pq < FGAM; ++i8j8pq){

                    int i8j8 = SREF_ijpq_GPU[i8j8pq][0];
                    int pq = SREF_ijpq_GPU[i8j8pq][1];

                    HpGAM_GPU[i8j8pq][ROW][COL] = cuCmul(ShSPixA_FIij_GPU[i8j8][threadIdx.x][threadIdx.y], 
                        ShSPixA_CFTpq_GPU[pq][threadIdx.x][threadIdx.y]);
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
        SFFTModule_dict['HadProd_GAM'] = _module

        # ** Fill Linear-System [Gamma]
        _refdict = {'N0': N0, 'N1': N1, 'NEQ': NEQ, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, 'FGAM': FGAM}
        _funcstr = r"""
        extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
            double PreGAM_GPU[%(FGAM)s][%(N0)s][%(N1)s], double LHMAT_GPU[%(NEQ)s][%(NEQ)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;

            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fpq = %(Fpq)s;
            int Fijab = %(Fijab)s;
            
            if (ROW < Fijab && COL < Fpq) {
                
                // INDEX Analysis
                int i8j8 = SREF_ijab_GPU[ROW][0];
                int a8b8 = SREF_ijab_GPU[ROW][1];
                int pq = COL;

                int a8 = REF_ab_GPU[a8b8][0];
                int b8 = REF_ab_GPU[a8b8][1];
                int idx = i8j8 * Fpq + pq;
                int cCOL = Fijab + COL;              // add offset

                // Define Mod_N0(rho), Mod_N1(eps)
                float tmp = 0.0;
                tmp = fmod(float(a8), float(N0));
                if (tmp < 0.0) {tmp += float(N0);}
                int MODa8 = tmp;

                tmp = fmod(float(b8), float(N1));
                if (tmp < 0.0) {tmp += float(N1);}
                int MODb8 = tmp;

                // Fill Linear System [B-component]
                if (a8 != 0 || b8 != 0) {
                    LHMAT_GPU[ROW][cCOL] = PreGAM_GPU[idx][MODa8][MODb8] - PreGAM_GPU[idx][0][0];
                }

                if (a8 == 0 && b8 == 0) {
                    LHMAT_GPU[ROW][cCOL] = PreGAM_GPU[idx][0][0];
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['FillLS_GAM'] = _module

        # ************************************ HadProd_PSI.cu & FillLS_PSI.cu ************************************ #

        # ** Hadamard Product [PSI]
        _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FPSI': FPSI, 'ThreadPerB': MaxThreadPerB}
        _funcstr = r"""
        #include "cuComplex.h"
        extern "C" __global__ void kmain(int SREF_pqij_GPU[%(FPSI)s][2],
            cuDoubleComplex SPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
            cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
            cuDoubleComplex HpPSI_GPU[%(FPSI)s][%(N0)s][%(N1)s]) 
        {    
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;

            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fpq = %(Fpq)s;
            int Fij = %(Fij)s;
            int FPSI = %(FPSI)s;
            
            __shared__ cuDoubleComplex ShSPixA_FTpq_GPU[%(Fpq)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int pq = 0; pq < Fpq; ++pq){
                    ShSPixA_FTpq_GPU[pq][threadIdx.x][threadIdx.y] = SPixA_FTpq_GPU[pq][ROW][COL];
                }
            }

            __shared__ cuDoubleComplex ShSPixA_CFIij_GPU[%(Fij)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int ij = 0; ij < Fij; ++ij){
                    ShSPixA_CFIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_CFIij_GPU[ij][ROW][COL];
                }
            }
            __syncthreads();

            if (ROW < N0 && COL < N1) {
                for(int p8q8ij = 0; p8q8ij < FPSI; ++p8q8ij){
                    
                    int p8q8 = SREF_pqij_GPU[p8q8ij][0];
                    int ij = SREF_pqij_GPU[p8q8ij][1];

                    HpPSI_GPU[p8q8ij][ROW][COL] = cuCmul(ShSPixA_FTpq_GPU[p8q8][threadIdx.x][threadIdx.y], 
                        ShSPixA_CFIij_GPU[ij][threadIdx.x][threadIdx.y]);
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
        SFFTModule_dict['HadProd_PSI'] = _module

        # ** Fill Linear-System [PSI]
        _refdict = {'N0': N0, 'N1': N1, 'NEQ': NEQ, 'Fab': Fab, 'Fij': Fij, 'Fpq': Fpq, 'Fijab': Fijab, 'FPSI': FPSI}
        _funcstr = r"""
        extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
            double PrePSI_GPU[%(FPSI)s][%(N0)s][%(N1)s], double LHMAT_GPU[%(NEQ)s][%(NEQ)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fij = %(Fij)s;
            int Fpq = %(Fpq)s;
            int Fijab = %(Fijab)s;

            if (ROW < Fpq && COL < Fijab) {

                // INDEX Analysis
                int cROW = Fijab + ROW;              // add offset
                int p8q8 = ROW;
                int ij = SREF_ijab_GPU[COL][0];
                int ab = SREF_ijab_GPU[COL][1];
                int a = REF_ab_GPU[ab][0];
                int b = REF_ab_GPU[ab][1];
                int idx = p8q8 * Fij + ij;

                // Define Mod_N0(rho), Mod_N1(eps)
                float tmp = 0.0;
                tmp = fmod(float(-a), float(N0));
                if (tmp < 0.0) {tmp += float(N0);}
                int MOD_a = tmp;

                tmp = fmod(float(-b), float(N1));
                if (tmp < 0.0) {tmp += float(N1);}
                int MOD_b = tmp;
                
                // Fill Linear System [B#-component]
                if (a != 0 || b != 0) {
                    LHMAT_GPU[cROW][COL] = PrePSI_GPU[idx][MOD_a][MOD_b] - PrePSI_GPU[idx][0][0];
                }

                if (a == 0 && b == 0) {
                    LHMAT_GPU[cROW][COL] = PrePSI_GPU[idx][0][0];
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['FillLS_PSI'] = _module

        # ************************************ HadProd_PHI.cu & FillLS_PHI.cu ************************************ #

        # ** Hadamard Product [PHI]
        _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'FPHI': FPHI, 'ThreadPerB': MaxThreadPerB}
        _funcstr = r"""
        #include "cuComplex.h"
        extern "C" __global__ void kmain(int SREF_pqp0q0_GPU[%(FPHI)s][2], 
            cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
            cuDoubleComplex SPixA_CFTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s],  
            cuDoubleComplex HpPHI_GPU[%(FPHI)s][%(N0)s][%(N1)s]) 
        {    
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;

            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fpq = %(Fpq)s;
            int FPHI = %(FPHI)s;

            __shared__ cuDoubleComplex ShSPixA_FTpq_GPU[%(Fpq)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int pq = 0; pq < Fpq; ++pq){
                    ShSPixA_FTpq_GPU[pq][threadIdx.x][threadIdx.y] = SPixA_FTpq_GPU[pq][ROW][COL];
                }
            }

            __shared__ cuDoubleComplex ShSPixA_CFTpq_GPU[%(Fpq)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int pq = 0; pq < Fpq; ++pq){
                    ShSPixA_CFTpq_GPU[pq][threadIdx.x][threadIdx.y] = SPixA_CFTpq_GPU[pq][ROW][COL];
                }
            }
            __syncthreads();
            
            if (ROW < N0 && COL < N1) {
                for(int p8q8pq = 0; p8q8pq < FPHI; ++p8q8pq){
                    
                    int p8q8 = SREF_pqp0q0_GPU[p8q8pq][0];
                    int pq = SREF_pqp0q0_GPU[p8q8pq][1];

                    HpPHI_GPU[p8q8pq][ROW][COL] = cuCmul(ShSPixA_FTpq_GPU[p8q8][threadIdx.x][threadIdx.y], 
                        ShSPixA_CFTpq_GPU[pq][threadIdx.x][threadIdx.y]);
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
        SFFTModule_dict['HadProd_PHI'] = _module

        # ** Fill Linear-System [PHI]
        _refdict = {'N0': N0, 'N1': N1, 'NEQ': NEQ, 'Fpq': Fpq, 'Fijab': Fijab, 'FPHI': FPHI}
        _funcstr = r"""
        extern "C" __global__ void kmain(double PrePHI_GPU[%(FPHI)s][%(N0)s][%(N1)s], 
            double LHMAT_GPU[%(NEQ)s][%(NEQ)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
            int Fpq = %(Fpq)s;
            int Fijab = %(Fijab)s;

            if (ROW < Fpq && COL < Fpq) {
                
                // INDEX Analysis
                int cROW = Fijab + ROW;              // add offset
                int cCOL = Fijab + COL;              // add offset
                
                int p8q8 = ROW;
                int pq = COL;
                int idx = p8q8 * Fpq + pq;

                // Fill Linear System [C-component]
                LHMAT_GPU[cROW][cCOL] = PrePHI_GPU[idx][0][0];
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['FillLS_PHI'] = _module

        # ************************************ HadProd_THE.cu & FillLS_THE.cu ************************************ #

        # ** Hadamard Product [THETA]
        _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FTHE': FTHE}
        _funcstr = r"""
        #include "cuComplex.h"
        extern "C" __global__ void kmain(cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
            cuDoubleComplex PixA_CFJ_GPU[%(N0)s][%(N1)s], cuDoubleComplex HpTHE_GPU[%(FTHE)s][%(N0)s][%(N1)s])
        {    
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;

            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int FTHE = %(FTHE)s;

            if (ROW < N0 && COL < N1) {
                for(int i8j8 = 0; i8j8 < FTHE; ++i8j8){
                    HpTHE_GPU[i8j8][ROW][COL] = cuCmul(PixA_CFJ_GPU[ROW][COL], SPixA_FIij_GPU[i8j8][ROW][COL]);
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
        SFFTModule_dict['HadProd_THE'] = _module

        # ** Fill Linear-System [THETA]  
        _refdict = {'N0': N0, 'N1': N1, 'NEQ': NEQ, 'Fijab': Fijab, 'Fab': Fab, 'FTHE': FTHE}
        _funcstr = r"""
        extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
            double PreTHE_GPU[%(FTHE)s][%(N0)s][%(N1)s], double RHb_GPU[%(NEQ)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fijab = %(Fijab)s;

            if (ROW < Fijab && COL == 0) {

                // INDEX Analysis
                int i8j8 = SREF_ijab_GPU[ROW][0];
                int a8b8 = SREF_ijab_GPU[ROW][1];
                int a8 = REF_ab_GPU[a8b8][0];
                int b8 = REF_ab_GPU[a8b8][1];
                int idx = i8j8;

                // Define Mod_N0(rho), Mod_N1(eps)
                float tmp = 0.0;
                tmp = fmod(float(a8), float(N0));
                if (tmp < 0.0) {tmp += float(N0);}
                int MODa8 = tmp;

                tmp = fmod(float(b8), float(N1));
                if (tmp < 0.0) {tmp += float(N1);}
                int MODb8 = tmp;
                
                // Fill Linear System [D-component]
                if (a8 != 0 || b8 != 0) {
                    RHb_GPU[ROW] = PreTHE_GPU[idx][MODa8][MODb8] - PreTHE_GPU[idx][0][0];
                }

                if (a8 == 0 && b8 == 0) {
                    RHb_GPU[ROW] = PreTHE_GPU[idx][0][0];
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['FillLS_THE'] = _module

        # ************************************ HadProd_DEL.cu & FillLS_DEL.cu ************************************ #

        # ** Hadamard Product [DELTA]
        _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'FDEL': FDEL}
        _funcstr = r"""
        #include "cuComplex.h"
        extern "C" __global__ void kmain(cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
            cuDoubleComplex PixA_CFJ_GPU[%(N0)s][%(N1)s], cuDoubleComplex HpDEL_GPU[%(FDEL)s][%(N0)s][%(N1)s])
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;

            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int FDEL = %(FDEL)s;

            if (ROW < N0 && COL < N1) {
                for(int p8q8 = 0; p8q8 < FDEL; ++p8q8){
                    HpDEL_GPU[p8q8][ROW][COL] = 
                        cuCmul(PixA_CFJ_GPU[ROW][COL], SPixA_FTpq_GPU[p8q8][ROW][COL]);
                }
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
        SFFTModule_dict['HadProd_DEL'] = _module

        # ** Fill Linear-System [DELTA]
        _refdict = {'N0': N0, 'N1': N1, 'NEQ': NEQ, 'Fpq': Fpq, 'Fijab': Fijab, 'FDEL': FDEL}
        _funcstr = r"""
        extern "C" __global__ void kmain(double PreDEL_GPU[%(FDEL)s][%(N0)s][%(N1)s], 
            double RHb_GPU[%(NEQ)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
            int Fpq = %(Fpq)s;
            int Fijab = %(Fijab)s;

            if (ROW < Fpq && COL == 0) {
                // INDEX Analysis
                int cROW = Fijab + ROW;              // add offset
                int idx = ROW;                       // i.e. p8q8

                // Fill Linear System [E-component]
                RHb_GPU[cROW] = PreDEL_GPU[idx][0][0];
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['FillLS_DEL'] = _module

        # ************************************ TweakLS.cu ************************************ #

        # ** Tweak Linear-System for Constant Scaling Case
        if KerSpType == 'Polynomial':
            
            _refdict = {'NEQ': NEQ, 'NEQt': NEQt}
            _funcstr = r"""
            extern "C" __global__ void kmain(double LHMAT_GPU[%(NEQ)s][%(NEQ)s], double RHb_GPU[%(NEQ)s], 
                int PresIDX_GPU[%(NEQt)s], double LHMAT_tweaked_GPU[%(NEQt)s][%(NEQt)s], 
                double RHb_tweaked_GPU[%(NEQt)s]) 
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;  // row index of tweakedLS
                int COL = blockIdx.y*blockDim.y+threadIdx.y;  // column index of tweakedLS
                int NEQt = %(NEQt)s;

                if (ROW < NEQt && COL < NEQt) {
                    int PresROW = PresIDX_GPU[ROW];  // row index of LS
                    int PresCOL = PresIDX_GPU[COL];  // column index of LS
                    LHMAT_tweaked_GPU[ROW][COL] = LHMAT_GPU[PresROW][PresCOL];
                }

                if (ROW < NEQt && COL == 0) {
                    int PresROW = PresIDX_GPU[ROW];  // row index of LS
                    RHb_tweaked_GPU[ROW] = RHb_GPU[PresROW];
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['TweakLS'] = _module

        if KerSpType == 'B-Spline':
            
            _refdict = {'Fij': Fij, 'NEQ': NEQ, 'NEQt': NEQt}
            _funcstr = r"""
            extern "C" __global__ void kmain(double LHMAT_GPU[%(NEQ)s][%(NEQ)s], double RHb_GPU[%(NEQ)s], 
                int PresIDX_GPU[%(NEQt)s], int ij00_GPU[%(Fij)s], double LHMAT_tweaked_GPU[%(NEQt)s][%(NEQt)s], 
                double RHb_tweaked_GPU[%(NEQt)s]) 
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;  // row index of tweakedLS
                int COL = blockIdx.y*blockDim.y+threadIdx.y;  // column index of tweakedLS

                int NEQt = %(NEQt)s;
                int Fij = %(Fij)s;
                int keyIdx = ij00_GPU[0];  // key index (ijab = 0000) of LS (tweakedLS)

                if (ROW == keyIdx && COL != keyIdx && COL < NEQt) {
                    double cum1 = 0.0;
                    for(int ij = 0; ij < Fij; ++ij){
                        int ridx = ij00_GPU[ij];         // row index of LS
                        int PresCOL = PresIDX_GPU[COL];  // column index of LS   
                        cum1 += LHMAT_GPU[ridx][PresCOL];
                    }
                    LHMAT_tweaked_GPU[ROW][COL] = cum1;
                }
                
                if (ROW != keyIdx && ROW < NEQt && COL == keyIdx) {
                    double cum2 = 0.0;
                    for(int ij = 0; ij < Fij; ++ij){
                        int PresROW = PresIDX_GPU[ROW];  // row index of LS
                        int cidx = ij00_GPU[ij];         // column index of LS
                        cum2 += LHMAT_GPU[PresROW][cidx];
                    }
                    LHMAT_tweaked_GPU[ROW][COL] = cum2;
                }

                if (ROW == keyIdx && COL == keyIdx) {
                    double cum3 = 0.0;
                    for(int ij = 0; ij < Fij; ++ij){
                        for(int i8j8 = 0; i8j8 < Fij; ++i8j8){
                            int ridx = ij00_GPU[ij];    // row index of LS
                            int cidx = ij00_GPU[i8j8];  // column index of LS
                            cum3 += LHMAT_GPU[ridx][cidx];
                        }
                    }
                    LHMAT_tweaked_GPU[ROW][COL] = cum3;
                }

                if (ROW != keyIdx && ROW < NEQt && COL != keyIdx && COL < NEQt) {
                    int PresROW = PresIDX_GPU[ROW];  // row index of LS
                    int PresCOL = PresIDX_GPU[COL];  // column index of LS
                    LHMAT_tweaked_GPU[ROW][COL] = LHMAT_GPU[PresROW][PresCOL];
                }

                if (ROW == keyIdx && COL == 0) {
                    double cum4 = 0.0;
                    for(int ij = 0; ij < Fij; ++ij){
                        int ridx = ij00_GPU[ij];  // row index of LS
                        cum4 += RHb_GPU[ridx];
                    }
                    RHb_tweaked_GPU[ROW] = cum4;
                }

                if (ROW != keyIdx && ROW < NEQt && COL == 0) {
                    int PresROW = PresIDX_GPU[ROW];  // row index of LS 
                    RHb_tweaked_GPU[ROW] = RHb_GPU[PresROW];
                }
                
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['TweakLS'] = _module
        
        # ************************************ Restore_Solution.cu ************************************ #

        # ** Restore Solution for Constant Scaling Case
        _refdict = {'NEQ': NEQ, 'NEQt': NEQt}
        _funcstr = r"""
        extern "C" __global__ void kmain(double Solution_tweaked_GPU[%(NEQt)s], 
            int PresIDX_GPU[%(NEQt)s], double Solution_GPU[%(NEQ)s]) 
        {    
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;  // row index of tweakedLS
            int COL = blockIdx.y*blockDim.y+threadIdx.y;  // trivial
            int NEQt = %(NEQt)s;
            
            if (ROW < NEQt && COL == 0) {
                int PresROW = PresIDX_GPU[ROW];  // row index of LS
                Solution_GPU[PresROW] = Solution_tweaked_GPU[ROW];
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['Restore_Solution'] = _module

        # ************************************ Construct_FDIFF.cu ************************************ #

        # ** Construct DIFF Image in Fourier Space 
        _refdict = {'N0': N0, 'N1': N1, 'L0': L0, 'L1': L1, 'w0': w0, 'w1': w1, \
                        'Fijab': Fijab, 'Fab': Fab, 'Fij': Fij, 'Fpq': Fpq, \
                        'SCALE': SCALE, 'ThreadPerB': MaxThreadPerB}
        _funcstr = r"""
        #include "cuComplex.h"
        extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2],  
            cuDoubleComplex a_ijab_GPU[%(Fijab)s], cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
            cuDoubleComplex Kab_Wla_GPU[%(L0)s][%(N0)s][%(N1)s], cuDoubleComplex Kab_Wmb_GPU[%(L1)s][%(N0)s][%(N1)s], 
            cuDoubleComplex b_pq_GPU[%(Fpq)s], cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
            cuDoubleComplex PixA_FJ_GPU[%(N0)s][%(N1)s], cuDoubleComplex PixA_FDIFF_GPU[%(N0)s][%(N1)s])
        {       
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            
            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int w0 = %(w0)s;
            int w1 = %(w1)s;
            int Fij = %(Fij)s;
            int Fab = %(Fab)s;
            int Fpq = %(Fpq)s;

            double One_re = 1.0;
            double Zero_re = 0.0;
            double Zero_im = 0.0;
            double SCALE_re = %(SCALE)s;
            cuDoubleComplex ZERO = make_cuDoubleComplex(Zero_re, Zero_im);
            cuDoubleComplex ONE = make_cuDoubleComplex(One_re, Zero_im);
            cuDoubleComplex SCA = make_cuDoubleComplex(SCALE_re, Zero_im);
            cuDoubleComplex PVAL = ZERO;
            cuDoubleComplex PVAL_FKab = ZERO;

            __shared__ cuDoubleComplex ShSPixA_FIij_GPU[%(Fij)s][%(ThreadPerB)s][%(ThreadPerB)s];
            if (ROW < N0 && COL < N1) {
                for(int ij = 0; ij < Fij; ++ij){
                    ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_FIij_GPU[ij][ROW][COL];
                }
            }
            __syncthreads();

            if (ROW < N0 && COL < N1) {

                PVAL = ZERO;
                PVAL_FKab = ZERO;
                for(int ab = 0; ab < Fab; ++ab){
                    int a = REF_ab_GPU[ab][0];
                    int b = REF_ab_GPU[ab][1];
                    
                    if (a == 0 && b == 0) {
                        PVAL_FKab = SCA;
                    }
                    
                    if (a != 0 || b != 0) {
                        PVAL_FKab = cuCmul(SCA, cuCsub(cuCmul(Kab_Wla_GPU[w0 + a][ROW][COL], Kab_Wmb_GPU[w1 + b][ROW][COL]), ONE));
                    }

                    for(int ij = 0; ij < Fij; ++ij){
                        int ijab = ij * Fab + ab;
                        PVAL = cuCadd(PVAL, cuCmul(cuCmul(a_ijab_GPU[ijab], ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y]), PVAL_FKab));
                    }
                }

                for(int pq = 0; pq < Fpq; ++pq){
                    PVAL = cuCadd(PVAL, cuCmul(b_pq_GPU[pq], SPixA_FTpq_GPU[pq][ROW][COL]));
                }
                
                PixA_FDIFF_GPU[ROW][COL] = cuCsub(PixA_FJ_GPU[ROW][COL], PVAL);
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
        SFFTModule_dict['Construct_FDIFF'] = _module
        
        SFFTConfig = (SFFTParam_dict, SFFTModule_dict)
        if VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- EXIT SFFT COMPILATION --//--//--//--//-- ')

        return SFFTConfig

class SingleSFFTConfigure_Numpy:
    @staticmethod
    def SSCN(NX, NY, KerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        ConstPhotRatio=True, NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        import numba as nb
        nb.set_num_threads = NUM_CPU_THREADS_4SUBTRACT

        N0, N1 = int(NX), int(NY)
        w0, w1 = int(KerHW), int(KerHW)
        DK, DB = int(KerSpDegree), int(BkgSpDegree)

        MaxThreadPerB = None              # trivial
        SCALE = np.float64(1/(N0*N1))     # Scale of Image-Size
        SCALE_L = np.float64(1/SCALE)     # Reciprocal Scale of Image-Size

        # NOTE: SFFT allows for Polynomial or B-Spline spatial variation
        assert KerSpType in ['Polynomial', 'B-Spline']
        assert BkgSpType in ['Polynomial', 'B-Spline']

        if KerSpType == 'B-Spline' and KerSpDegree == 0:
            assert len(KerIntKnotX) == 0  # otherwise, discontinuity
            assert len(KerIntKnotY) == 0  # otherwise, discontinuity
        
        if BkgSpType == 'B-Spline' and BkgSpDegree == 0:
            assert len(BkgIntKnotX) == 0  # otherwise, discontinuity
            assert len(BkgIntKnotY) == 0  # otherwise, discontinuity
        
        # NOTE: DK = 0 means a spatial invariant matching kernel
        # NOTE: DB = 0 means a flat differential background
        assert DK >= 0 and DB >= 0
        
        if VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- TRIGGER SFFT COMPILATION [Numpy] --//--//--//--//-- ')
            if KerSpType == 'Polynomial':
                print('\n ---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(KerSpDegree, KerHW))
            if KerSpType == 'B-Spline':
                print('\n ---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                    %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree, KerHW))
            if BkgSpType == 'Polynomial':
                print('\n ---//--- Polynomial Background | BkgSpDegree %d ---//---' %BkgSpDegree)
            if BkgSpType == 'B-Spline':
                print('\n ---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), BkgSpDegree))

        # * Make a dictionary for SFFT parameters
        L0 = 2*w0+1                                      # matching-kernel XSize
        L1 = 2*w1+1                                      # matching-kernel YSize
        Fab = L0*L1                                      # dof for index ab

        if KerSpType == 'Polynomial':
            Fi, Fj = -1, -1                              # not independent
            Fij = int((DK+1)*(DK+2)/2)                   # dof for matching-kernel polynomial index ij 
        
        if KerSpType == 'B-Spline':
            Fi = len(KerIntKnotX) + KerSpDegree + 1      # dof for matching-kernel B-spline index i (control points/coefficients)
            Fj = len(KerIntKnotY) + KerSpDegree + 1      # dof for matching-kernel B-spline index j (control points/coefficients)
            Fij = Fi*Fj                                  # dof for matching-kernel B-spline index ij
        
        if BkgSpType == 'Polynomial':
            Fp, Fq = -1, -1                              # not independent
            Fpq = int((DB+1)*(DB+2)/2)                   # dof for diff-background polynomial index pq 
        
        if BkgSpType == 'B-Spline':
            Fp = len(BkgIntKnotX) + BkgSpDegree + 1      # dof for diff-background B-spline index p (control points/coefficients)
            Fq = len(BkgIntKnotY) + BkgSpDegree + 1      # dof for diff-background B-spline index q (control points/coefficients)  
            Fpq = Fp*Fq                                  # dof for diff-background B-spline index pq

        Fijab = Fij*Fab                                  # Linear-System Major side-length
        FOMG, FGAM, FTHE = Fij**2, Fij*Fpq, Fij          # OMG / GAM / THE has shape (dof, N0, N1)
        FPSI, FPHI, FDEL = Fpq*Fij, Fpq**2, Fpq          # PSI / PHI / DEL has shape (dof, N0, N1)
        NEQ = Fij*Fab+Fpq                                # Linear-System side-length
        NEQt = NEQ - Fij + 1                             # tweaked Linear-System side-length for constant ratio

        SFFTParam_dict = {}
        SFFTParam_dict['KerHW'] = KerHW
        SFFTParam_dict['KerSpType'] = KerSpType
        SFFTParam_dict['KerSpDegree'] = KerSpDegree
        SFFTParam_dict['KerIntKnotX'] = KerIntKnotX
        SFFTParam_dict['KerIntKnotY'] = KerIntKnotY
        SFFTParam_dict['BkgSpType'] = BkgSpType
        SFFTParam_dict['BkgSpDegree'] = BkgSpDegree
        SFFTParam_dict['BkgIntKnotX'] = BkgIntKnotX
        SFFTParam_dict['BkgIntKnotY'] = BkgIntKnotY
        SFFTParam_dict['ConstPhotRatio'] = ConstPhotRatio

        SFFTParam_dict['N0'] = N0    # a.k.a, NX
        SFFTParam_dict['N1'] = N1    # a.k.a, NY
        SFFTParam_dict['w0'] = w0    # a.k.a, KerHW
        SFFTParam_dict['w1'] = w1    # a.k.a, KerHW
        SFFTParam_dict['DK'] = DK    # a.k.a, KerSpDegree
        SFFTParam_dict['DB'] = DB    # a.k.a, BkgSpDegree

        SFFTParam_dict['MaxThreadPerB'] = MaxThreadPerB
        SFFTParam_dict['SCALE'] = SCALE
        SFFTParam_dict['SCALE_L'] = SCALE_L

        SFFTParam_dict['L0'] = L0
        SFFTParam_dict['L1'] = L1
        SFFTParam_dict['Fab'] = Fab
        SFFTParam_dict['Fi'] = Fi
        SFFTParam_dict['Fj'] = Fj
        SFFTParam_dict['Fij'] = Fij
        SFFTParam_dict['Fp'] = Fp
        SFFTParam_dict['Fq'] = Fq
        SFFTParam_dict['Fpq'] = Fpq
        SFFTParam_dict['Fijab'] = Fijab
        
        SFFTParam_dict['FOMG'] = FOMG
        SFFTParam_dict['FGAM'] = FGAM
        SFFTParam_dict['FTHE'] = FTHE
        SFFTParam_dict['FPSI'] = FPSI
        SFFTParam_dict['FPHI'] = FPHI
        SFFTParam_dict['FDEL'] = FDEL
        
        SFFTParam_dict['NEQ'] = NEQ
        SFFTParam_dict['NEQt'] = NEQt

        # * Load SFFT Numpy-based modules
        # ************************************ SpatialCoord.py ************************************ #

        SFFTModule_dict = {}
        # ** produce spatial coordinate X/Y/oX/oY-map
        _strdec = 'Tuple((i4[:,:], i4[:,:], f8[:,:], f8[:,:]))(i4[:,:], i4[:,:], f8[:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def SpatialCoord(PixA_X, PixA_Y, PixA_CX, PixA_CY):
            
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

        SFFTModule_dict['SpatialCoord'] = SpatialCoord

        # ************************************ KerSpatial.py ************************************ #

        # ** produce Iij
        if KerSpType == 'Polynomial':

            _strdec = 'f8[:,:,:](i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def KerSpatial(REF_ij, PixA_CX, PixA_CY, PixA_I, SPixA_Iij):

                #assert REF_ij.dtype == np.int32 and REF_ij.shape == (Fij, 2)
                #assert PixA_CX.dtype == np.float64 and PixA_CX.shape == (N0, N1)
                #assert PixA_CY.dtype == np.float64 and PixA_CY.shape == (N0, N1)
                #assert PixA_I.dtype == np.float64 and PixA_I.shape == (N0, N1)
                #assert SPixA_Iij.dtype == np.float64 and SPixA_Iij.shape == (Fij, N0, N1)

                for ij in nb.prange(Fij):
                    i, j = REF_ij[ij]
                    PixA_poly = np.power(PixA_CX, i) * np.power(PixA_CY, j)
                    SPixA_Iij[ij, :, :] = PixA_I * PixA_poly   # NOTE UPDATE
                
                return SPixA_Iij

            SFFTModule_dict['KerSpatial'] = KerSpatial
        
        if KerSpType == 'B-Spline':

            _strdec = 'f8[:,:,:](i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def KerSpatial(REF_ij, KerSplBasisX, KerSplBasisY, PixA_I, SPixA_Iij):

                #assert REF_ij.dtype == np.int32 and REF_ij.shape == (Fij, 2)
                #assert KerSplBasisX.dtype == np.float64 and KerSplBasisX.shape == (Fi, N0)
                #assert KerSplBasisY.dtype == np.float64 and KerSplBasisY.shape == (Fj, N1)
                #assert PixA_I.dtype == np.float64 and PixA_I.shape == (N0, N1)
                #assert SPixA_Iij.dtype == np.float64 and SPixA_Iij.shape == (Fij, N0, N1)

                for ij in nb.prange(Fij):
                    i, j = REF_ij[ij]
                    for ROW in nb.prange(N0):
                        splx = KerSplBasisX[i, ROW]
                        for COL in nb.prange(N1):
                            sply = KerSplBasisY[j, COL]
                            SPixA_Iij[ij, ROW, COL] = PixA_I[ROW, COL] * splx * sply   # NOTE UPDATE

                return SPixA_Iij

            SFFTModule_dict['KerSpatial'] = KerSpatial
        
        # ** produce Tpq
        if BkgSpType == 'Polynomial':

            _strdec = 'f8[:,:,:](i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def BkgSpatial(REF_pq, PixA_CX, PixA_CY, SPixA_Tpq):

                #assert REF_pq.dtype == np.int32 and REF_pq.shape == (Fpq, 2)
                #assert PixA_CX.dtype == np.float64 and PixA_CX.shape == (N0, N1)
                #assert PixA_CY.dtype == np.float64 and PixA_CY.shape == (N0, N1)
                #assert SPixA_Tpq.dtype == np.float64 and SPixA_Tpq.shape == (Fpq, N0, N1)

                for pq in nb.prange(Fpq):
                    p, q = REF_pq[pq]
                    PixA_poly = np.power(PixA_CX, p) * np.power(PixA_CY, q)
                    SPixA_Tpq[pq, :, :] = PixA_poly   # NOTE UPDATE
                
                return SPixA_Tpq

            SFFTModule_dict['BkgSpatial'] = BkgSpatial
        
        if BkgSpType == 'B-Spline':

            _strdec = 'f8[:,:,:](i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def KerSpatial(REF_pq, BkgSplBasisX, BkgSplBasisY, SPixA_Tpq):

                #assert REF_pq.dtype == np.int32 and REF_pq.shape == (Fpq, 2)
                #assert BkgSplBasisX.dtype == np.float64 and BkgSplBasisX.shape == (Fp, N0)
                #assert BkgSplBasisY.dtype == np.float64 and BkgSplBasisY.shape == (Fq, N1)
                #assert SPixA_Tpq.dtype == np.float64 and SPixA_Tpq.shape == (Fpq, N0, N1)

                for pq in nb.prange(Fpq):
                    p, q = REF_pq[pq]
                    for ROW in nb.prange(N0):
                        splx = BkgSplBasisX[p, ROW]
                        for COL in nb.prange(N1):
                            sply = BkgSplBasisY[q, COL]
                            SPixA_Tpq[pq, ROW, COL] = splx * sply   # NOTE UPDATE

                return SPixA_Tpq

            SFFTModule_dict['BkgSpatial'] = BkgSpatial
        
        # ************************************ HadProd_OMG.py & FillLS_OMG.py ************************************ #

        # ** Hadamard Product [Omega]
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

        SFFTModule_dict['HadProd_OMG'] = HadProd_OMG

        # ** Fill Linear-System [Omega]
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

        SFFTModule_dict['FillLS_OMG'] = FillLS_OMG

        # ************************************ HadProd_GAM.py & FillLS_GAM.py ************************************ #

        # ** Hadamard Product [Gamma]
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

        SFFTModule_dict['HadProd_GAM'] = HadProd_GAM

        # ** Fill Linear-System [Gamma]
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

        SFFTModule_dict['FillLS_GAM'] = FillLS_GAM

        # ************************************ HadProd_PSI.py & FillLS_PSI.py ************************************ #

        # ** Hadamard Product [PSI]
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

        SFFTModule_dict['HadProd_PSI'] = HadProd_PSI

        # ** Fill Linear-System [PSI]
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
        
        SFFTModule_dict['FillLS_PSI'] = FillLS_PSI

        # ************************************ HadProd_PHI.py & FillLS_PHI.py ************************************ #

        # ** Hadamard Product [PHI]
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

        SFFTModule_dict['HadProd_PHI'] = HadProd_PHI

        # ** Fill Linear-System [PHI]
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

        SFFTModule_dict['FillLS_PHI'] = FillLS_PHI

        # ************************************ HadProd_THE.py & FillLS_THE.py ************************************ #

        # ** Hadamard Product [THETA]
        _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_THE(SPixA_FIij, PixA_CFJ, HpTHE):
            
            #assert SPixA_FIij.dtype == np.complex128 and SPixA_FIij.shape == (Fij, N0, N1)
            #assert PixA_CFJ.dtype == np.complex128 and PixA_CFJ.shape == (N0, N1)
            #assert HpTHE.dtype == np.complex128 and HpTHE.shape == (FTHE, N0, N1)
            
            for i8j8 in nb.prange(FTHE):
                HpTHE[i8j8, :, :] = PixA_CFJ * SPixA_FIij[i8j8]   # NOTE UPDATE

            return HpTHE
        
        SFFTModule_dict['HadProd_THE'] = HadProd_THE

        # ** Fill Linear-System [THETA]
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

        SFFTModule_dict['FillLS_THE'] = FillLS_THE

        # ************************************ HadProd_DEL.py & FillLS_DEL.py ************************************ #

        # ** Hadamard Product [DELTA]
        _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
        @nb.njit(_strdec, parallel=True)
        def HadProd_DEL(SPixA_FTpq, PixA_CFJ, HpDEL):
            
            #assert SPixA_FTpq.dtype == np.complex128 and SPixA_FTpq.shape == (Fpq, N0, N1)
            #assert PixA_CFJ.dtype == np.complex128 and PixA_CFJ.shape == (N0, N1)
            #assert HpDEL.dtype == np.complex128 and HpDEL.shape == (FDEL, N0, N1)
            
            for p8q8 in nb.prange(FDEL):
                HpDEL[p8q8, :, :] = PixA_CFJ * SPixA_FTpq[p8q8]   # NOTE UPDATE

            return HpDEL

        SFFTModule_dict['HadProd_DEL'] = HadProd_DEL

        # ** Fill Linear-System [DELTA]
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
        
        SFFTModule_dict['FillLS_DEL'] = FillLS_DEL

        # ************************************ TweakLS.py ************************************ #

        # ** Tweak Linear-System for Constant Scaling Case
        if KerSpType == 'Polynomial':
            
            _strdec = 'Tuple((f8[:,:], f8[:]))(f8[:,:], f8[:], i4[:], f8[:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def TweakLS(LHMAT, RHb, PresIDX, LHMAT_tweaked, RHb_tweaked):
                
                #assert LHMAT.dtype == np.float64 and LHMAT.shape == (NEQ, NEQ)
                #assert RHb.dtype == np.float64 and RHb.shape == (NEQ, )
                #assert PresIDX.dtype == np.int32 and PresIDX.shape == (NEQt, )
                #assert LHMAT_tweaked.dtype == np.float64 and LHMAT_tweaked.shape == (NEQt, NEQt)
                #assert RHb_tweaked.dtype == np.float64 and RHb_tweaked.shape == (NEQt, )
                
                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    for COL in nb.prange(NEQt):
                        PresCOL = PresIDX[COL]
                        LHMAT_tweaked[ROW, COL] = LHMAT[PresROW, PresCOL]
                    RHb_tweaked[ROW] = RHb[PresROW]
                
                return LHMAT_tweaked, RHb_tweaked

            SFFTModule_dict['TweakLS'] = TweakLS

        if KerSpType == 'B-Spline':

            _strdec = 'Tuple((f8[:,:], f8[:]))(f8[:,:], f8[:], i4[:], i4[:], f8[:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def TweakLS(LHMAT, RHb, PresIDX, ij00, LHMAT_tweaked, RHb_tweaked):

                #assert LHMAT.dtype == np.float64 and LHMAT.shape == (NEQ, NEQ)
                #assert RHb.dtype == np.float64 and RHb.shape == (NEQ, )
                #assert PresIDX.dtype == np.int32 and PresIDX.shape == (NEQt, )
                #assert ij00.dtype == np.int32 and ij00.shape == (Fij, )
                #assert LHMAT_tweaked.dtype == np.float64 and LHMAT_tweaked.shape == (NEQt, NEQt)
                #assert RHb_tweaked.dtype == np.float64 and RHb_tweaked.shape == (NEQt, )
                
                keyIdx = ij00[0]
                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    
                    for COL in nb.prange(NEQt):
                        PresCOL = PresIDX[COL]
                        
                        if (ROW == keyIdx and COL != keyIdx):
                            val = 0.0
                            for idx in ij00: val += LHMAT[idx, PresCOL]
                            LHMAT_tweaked[ROW, COL] = val
                        
                        if (ROW != keyIdx and COL == keyIdx):
                            val = 0.0
                            for idx in ij00: val += LHMAT[PresROW, idx]
                            LHMAT_tweaked[ROW, COL] = val
                            
                        if (ROW == keyIdx and COL == keyIdx):
                            val = 0.0
                            for idx in ij00:
                                for idx8 in ij00:
                                    val += LHMAT[idx, idx8]
                            LHMAT_tweaked[ROW, COL] = val
                        
                        if (ROW != keyIdx and COL != keyIdx):
                            LHMAT_tweaked[ROW, COL] = LHMAT[PresROW, PresCOL]

                    if ROW == keyIdx:
                        val = 0.0
                        for idx in ij00: val += RHb[idx]
                        RHb_tweaked[ROW] = val
                        
                    if ROW != keyIdx:
                        RHb_tweaked[ROW] = RHb[PresROW]

                return LHMAT_tweaked, RHb_tweaked

            SFFTModule_dict['TweakLS'] = TweakLS

        # ************************************ Restore_Solution.py ************************************ #

        # ** Restore Solution for Constant Scaling Case
        _strdec = 'f8[:](f8[:], i4[:], f8[:])'
        @nb.njit(_strdec, parallel=True)
        def Restore_Solution(Solution_tweaked, PresIDX, Solution):

            #assert Solution_tweaked.dtype == np.float64 and Solution_tweaked.shape == (NEQt, )
            #assert PresIDX.dtype == np.int32 and PresIDX.shape == (NEQt, )
            #assert Solution.dtype == np.float64 and Solution.shape == (NEQ, )

            for ROW in nb.prange(NEQt):
                PresROW = PresIDX[ROW]
                Solution[PresROW] = Solution_tweaked[ROW]

            return Solution

        SFFTModule_dict['Restore_Solution'] = Restore_Solution

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
        if VERBOSE_LEVEL in [1, 2]: 
            print('\n --//--//--//--//-- EXIT SFFT COMPILATION --//--//--//--//-- ')

        return SFFTConfig

class SingleSFFTConfigure:
    @staticmethod
    def SSC(NX, NY, KerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], ConstPhotRatio=True, \
        BACKEND_4SUBTRACT='Cupy', MaxThreadPerB=8, NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        """
        # Compile Functions for SFFT
        #
        # Arguments:
        # -NX: Image size along X (pix)                                                     | e.g., 1024
        # -NY: Image size along Y (pix)                                                     | e.g., 1024
        # -KerHW: Kernel half width for compilation                                         | e.g., 8
        # -KerSpType: Spatial Varaition Type of Matching Kernel                             | ['Polynomial', 'B-Spline']
        # -KerSpDegree: Polynomial/B-Spline Degree of Kernel Spatial Varaition              | [0, 1, 2, 3]
        # -KerIntKnotX: Internal Knots of Kernel B-Spline Spatial Varaition along X         | e.g., [256., 512., 768.]
        # -KerIntKnotY: Internal Knots of Kernel B-Spline Spatial Varaition along Y         | e.g., [256., 512., 768.]
        # -BkgSpType: Spatial Varaition Type of Differential Background                     | ['Polynomial', 'B-Spline']
        # -BkgSpDegree: Polynomial/Spline Degree of Background Spatial Varaition            | [0, 1, 2, 3]
        # -BkgIntKnotX: Internal Knots of Background B-Spline Spatial Varaition along X     | e.g., [256., 512., 768.]
        # -BkgIntKnotY: Internal Knots of Background B-Spline Spatial Varaition along Y     | e.g., [256., 512., 768.]
        # -ConstPhotRatio: Use a constant photometric ratio in image subtraction?           | [True, False]
        # -BACKEND_4SUBTRACT : The backend with which you perform SFFT subtraction          | ['Cupy', 'Numpy']
        # -MaxThreadPerB: Maximum Threads per Block for CUDA configuration                  | e.g., 8
        # -NUM_CPU_THREADS_4SUBTRACT: The number of CPU threads for Numpy-SFFT subtraction  | e.g., 8
        # -VERBOSE_LEVEL: The level of verbosity, can be 0/1/2: QUIET/NORMAL/FULL           | [0, 1, 2]
        #
        """
        
        if BACKEND_4SUBTRACT == 'Cupy':
            SFFTConfig = SingleSFFTConfigure_Cupy.SSCC(NX=NX, NY=NY, KerHW=KerHW, \
                KerSpType=KerSpType, KerSpDegree=KerSpDegree, KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, \
                BkgSpType=BkgSpType, BkgSpDegree=BkgSpDegree, BkgIntKnotX=BkgIntKnotX, BkgIntKnotY=BkgIntKnotY, \
                ConstPhotRatio=ConstPhotRatio, MaxThreadPerB=MaxThreadPerB, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        if BACKEND_4SUBTRACT == 'Numpy':
            SFFTConfig = SingleSFFTConfigure_Numpy.SSCN(NX=NX, NY=NY, KerHW=KerHW, \
                KerSpType=KerSpType, KerSpDegree=KerSpDegree, KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, \
                BkgSpType=BkgSpType, BkgSpDegree=BkgSpDegree, BkgIntKnotX=BkgIntKnotX, BkgIntKnotY=BkgIntKnotY, \
                ConstPhotRatio=ConstPhotRatio, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, \
                VERBOSE_LEVEL=VERBOSE_LEVEL)

        return SFFTConfig
