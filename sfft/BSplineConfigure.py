import numpy as np
# version: Oct 10, 2025

# WARNING: THIS MODULE IS STILL BEING DEVELOPED AND LOT OF IMPROVEMENT NOT IMPLETEMENTED YET!
__author__ = "Lei Hu <leihu@andrew.cmu.edu>, Zachary Stone <stone28@illinois.edu>"
__version__ = "v2.0dev"

class SingleSFFTConfigure_Cupy:
    @staticmethod
    def SSCC(NX, NY, KerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        REGULARIZE_KERNEL=False, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=None, WEIGHT_REGULARIZE=None, \
        LAMBDA_REGULARIZE=1e-6, MAX_THREADS_PER_BLOCK=8, MINIMIZE_GPU_MEMORY_USAGE=False, VERBOSE_LEVEL=2):

        import cupy as cp

        N0, N1 = int(NX), int(NY)
        w0, w1 = int(KerHW), int(KerHW)

        SCALE = np.float64(1/(N0*N1))     # Scale of Image-Size
        SCALE_L = np.float64(1/SCALE)     # Reciprocal Scale of Image-Size

        # kernel spatial variation
        DK = int(KerSpDegree)
        assert DK >= 0
        assert KerSpType in ['Polynomial', 'B-Spline']
        if KerSpType == 'B-Spline' and DK == 0:
            assert len(KerIntKnotX) == 0  # otherwise, discontinuity
            assert len(KerIntKnotY) == 0  # otherwise, discontinuity
        
        # scaling spatial variation
        if SEPARATE_SCALING:
            DS = int(ScaSpDegree)
            assert DS >= 0
            assert ScaSpType in ['Polynomial', 'B-Spline']
            if ScaSpType == 'B-Spline' and DS == 0:
                assert len(ScaIntKnotX) == 0  # otherwise, discontinuity
                assert len(ScaIntKnotY) == 0  # otherwise, discontinuity

        # Remarks on SCALING_MODE
        # SEPARATE_SCALING & ScaSpDegree >>>      SCALING_MODE
        #        N         &     any     >>>       'ENTANGLED'
        #        Y         &      0      >>>   'SEPARATE-CONSTANT'
        #        Y         &     > 0     >>>   'SEPARATE-VARYING'

        SCALING_MODE = None
        if not SEPARATE_SCALING:
            SCALING_MODE = 'ENTANGLED'
        elif ScaSpDegree == 0:
            SCALING_MODE = 'SEPARATE-CONSTANT'
        else: SCALING_MODE = 'SEPARATE-VARYING'
        assert SCALING_MODE is not None

        if SCALING_MODE == 'SEPARATE-CONSTANT':
            assert KerSpDegree != 0   # otherwise, reduced to ENTANGLED

        if SCALING_MODE == 'SEPARATE-VARYING':
            # force to activate MINIMIZE_GPU_MEMORY_USAGE
            assert MINIMIZE_GPU_MEMORY_USAGE

            if KerSpType == 'Polynomial' and ScaSpType == 'Polynomial':
                assert ScaSpDegree != KerSpDegree   # otherwise, reduced to ENTANGLED
            
            if KerSpType == 'B-Spline' and ScaSpType == 'B-Spline':
                if np.all(KerIntKnotX == ScaIntKnotX) and np.all(KerIntKnotY == ScaIntKnotY):
                    assert ScaSpDegree != KerSpDegree   # otherwise, reduced to ENTANGLED
        
        # background spatial variation
        DB = int(BkgSpDegree)
        assert DB >= 0
        assert BkgSpType in ['Polynomial', 'B-Spline']
        if BkgSpType == 'B-Spline' and BkgSpDegree == 0:
            assert len(BkgIntKnotX) == 0  # otherwise, discontinuity
            assert len(BkgIntKnotY) == 0  # otherwise, discontinuity

        # NOTE input image should not has dramatically small size
        assert N0 > MAX_THREADS_PER_BLOCK and N1 > MAX_THREADS_PER_BLOCK

        if REGULARIZE_KERNEL:
            assert XY_REGULARIZE is not None
            assert len(XY_REGULARIZE.shape) == 2 and XY_REGULARIZE.shape[1] == 2

        if VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- TRIGGER SFFT COMPILATION [Cupy] --//--//--//--//-- ')

            if KerSpType == 'Polynomial':
                print('\n ---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(KerSpDegree, KerHW))
                if not SEPARATE_SCALING:
                    print('\n ---//--- [ENTANGLED] Polynomial Scaling | KerSpDegree %d ---//---' %KerSpDegree)

            if KerSpType == 'B-Spline':
                print('\n ---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                      %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree, KerHW))
                if not SEPARATE_SCALING: 
                    print('\n ---//--- [ENTANGLED] B-Spline Scaling | Internal Knots %d,%d | KerSpDegree %d ---//---' \
                          %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree))
            
            if SEPARATE_SCALING:
                if ScaSpType == 'Polynomial':
                    print('\n ---//--- [SEPARATE] Polynomial Scaling | ScaSpDegree %d ---//---' %ScaSpDegree)
                
                if ScaSpType == 'B-Spline':
                    print('\n ---//--- [SEPARATE] B-Spline Scaling | Internal Knots %d,%d | ScaSpDegree %d ---//---' \
                          %(len(ScaIntKnotX), len(ScaIntKnotY), ScaSpDegree))
            
            if BkgSpType == 'Polynomial':
                print('\n ---//--- Polynomial Background | BkgSpDegree %d ---//---' %BkgSpDegree)
            
            if BkgSpType == 'B-Spline':
                print('\n ---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), BkgSpDegree))

        SFFTParam_dict = {}
        SFFTParam_dict['KerHW'] = KerHW
        SFFTParam_dict['KerSpType'] = KerSpType
        SFFTParam_dict['KerSpDegree'] = KerSpDegree
        SFFTParam_dict['KerIntKnotX'] = KerIntKnotX
        SFFTParam_dict['KerIntKnotY'] = KerIntKnotY

        SFFTParam_dict['SEPARATE_SCALING'] = SEPARATE_SCALING
        if SEPARATE_SCALING:
            SFFTParam_dict['ScaSpType'] = ScaSpType
            SFFTParam_dict['ScaSpDegree'] = ScaSpDegree
            SFFTParam_dict['ScaIntKnotX'] = ScaIntKnotX
            SFFTParam_dict['ScaIntKnotY'] = ScaIntKnotY

        SFFTParam_dict['BkgSpType'] = BkgSpType
        SFFTParam_dict['BkgSpDegree'] = BkgSpDegree
        SFFTParam_dict['BkgIntKnotX'] = BkgIntKnotX
        SFFTParam_dict['BkgIntKnotY'] = BkgIntKnotY

        SFFTParam_dict['REGULARIZE_KERNEL'] = REGULARIZE_KERNEL
        SFFTParam_dict['IGNORE_LAPLACIAN_KERCENT'] = IGNORE_LAPLACIAN_KERCENT
        SFFTParam_dict['XY_REGULARIZE'] = XY_REGULARIZE
        SFFTParam_dict['WEIGHT_REGULARIZE'] = WEIGHT_REGULARIZE
        SFFTParam_dict['LAMBDA_REGULARIZE'] = LAMBDA_REGULARIZE

        SFFTParam_dict['MAX_THREADS_PER_BLOCK'] = MAX_THREADS_PER_BLOCK
        SFFTParam_dict['MINIMIZE_GPU_MEMORY_USAGE'] = MINIMIZE_GPU_MEMORY_USAGE
        
        # * Make a dictionary for SFFT parameters
        L0 = 2*w0+1                                       # matching-kernel XSize
        L1 = 2*w1+1                                       # matching-kernel YSize
        Fab = L0*L1                                       # dof for index ab

        if KerSpType == 'Polynomial':
            Fi, Fj = -1, -1                               # not independent, placeholder
            Fij = ((DK+1)*(DK+2))//2                      # dof for matching-kernel polynomial index ij 
        
        if KerSpType == 'B-Spline':
            Fi = len(KerIntKnotX) + KerSpDegree + 1       # dof for matching-kernel B-spline index i (control points/coefficients)
            Fj = len(KerIntKnotY) + KerSpDegree + 1       # dof for matching-kernel B-spline index j (control points/coefficients)
            Fij = Fi*Fj                                   # dof for matching-kernel B-spline index ij
        
        if BkgSpType == 'Polynomial':
            Fp, Fq = -1, -1                               # not independent, placeholder
            Fpq = ((DB+1)*(DB+2))//2                      # dof for diff-background polynomial index pq 
        
        if BkgSpType == 'B-Spline':
            Fp = len(BkgIntKnotX) + BkgSpDegree + 1       # dof for diff-background B-spline index p (control points/coefficients)
            Fq = len(BkgIntKnotY) + BkgSpDegree + 1       # dof for diff-background B-spline index q (control points/coefficients)  
            Fpq = Fp*Fq                                   # dof for diff-background B-spline index pq

        if SCALING_MODE == 'SEPARATE-VARYING':
            if ScaSpType == 'Polynomial':
                ScaFi, ScaFj = -1, -1                     # not independent, placeholder
                ScaFij = ((DS+1)*(DS+2))//2               # effective dof for scaling polynomial index ij
            
            if ScaSpType == 'B-Spline':
                ScaFi = len(ScaIntKnotX) + ScaSpDegree + 1    # dof for scaling B-spline index i (control points/coefficients)
                ScaFj = len(ScaIntKnotY) + ScaSpDegree + 1    # dof for scaling B-spline index j (control points/coefficients)
                ScaFij = ScaFi*ScaFj                          # effective dof for scaling B-spline index ij
            
            # Remarks on the scaling effective dof
            # I. current version not support scaling effective dof no higher than kernel variation.
            #    for simplicity, we use trivail zero basis as placeholder so that 
            #    the apparent dof of scaling and kernel are consistent.
            # II. ScaFij = Fij is allowed, e.g.m, B-Spline, same degree and 
            #     same number of internal knots but at different positions.

            assert ScaFij <= Fij
        
        Fijab = Fij*Fab                                   # Linear-System Major side-length
        FOMG, FGAM, FTHE = Fij**2, Fij*Fpq, Fij           # OMG / GAM / THE has shape (dof, N0, N1)
        FPSI, FPHI, FDEL = Fpq*Fij, Fpq**2, Fpq           # PSI / PHI / DEL has shape (dof, N0, N1)
        NEQ = Fij*Fab+Fpq                                 # Linear-System side-length
        
        NEQt = NEQ
        if SCALING_MODE == 'SEPARATE-CONSTANT':
            NEQt = NEQ-Fij+1                    # tweaked Linear-System side-length for constant scaling

        if SCALING_MODE == 'SEPARATE-VARYING':
            NEQt = NEQ-(Fij-ScaFij)             # tweaked Linear-System side-length for polynomial-varying scaling

        SFFTParam_dict['N0'] = N0               # a.k.a, NX
        SFFTParam_dict['N1'] = N1               # a.k.a, NY
        SFFTParam_dict['w0'] = w0               # a.k.a, KerHW
        SFFTParam_dict['w1'] = w1               # a.k.a, KerHW
        SFFTParam_dict['DK'] = DK               # a.k.a, KerSpDegree
        SFFTParam_dict['DB'] = DB               # a.k.a, BkgSpDegree
        if SEPARATE_SCALING: 
            SFFTParam_dict['DS'] = DS           # a.k.a, ScaSpDegree

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

        if SCALING_MODE == 'SEPARATE-VARYING':
            SFFTParam_dict['ScaFi'] = ScaFi
            SFFTParam_dict['ScaFj'] = ScaFj
            SFFTParam_dict['ScaFij'] = ScaFij        
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

        SFFTModule_dict = {}
        # ************************************ Spatial Variation ************************************ #

        # <*****> produce spatial coordinate X/Y/oX/oY-map <*****> #
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

        # <*****> produce Iij <*****> #
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
        
        if SCALING_MODE == 'SEPARATE-VARYING':

            if ScaSpType == 'Polynomial':

                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij}
                _funcstr = r"""
                extern "C" __global__ void kmain(int ScaREF_ij_GPU[%(Fij)s][2],
                    double PixA_CX_GPU[%(N0)s][%(N1)s], double PixA_CY_GPU[%(N0)s][%(N1)s],
                    double PixA_I_GPU[%(N0)s][%(N1)s], double ScaSPixA_Iij_GPU[%(Fij)s][%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;
                
                    int N0 = %(N0)s;
                    int N1 = %(N1)s;
                    int Fij = %(Fij)s;
                    
                    if (ROW < N0 && COL < N1) {
                        for(int ij = 0; ij < Fij; ++ij) {
                            int i = ScaREF_ij_GPU[ij][0];
                            int j = ScaREF_ij_GPU[ij][1];

                            if (i >= 0 && j >= 0) {
                                double poly = pow(PixA_CX_GPU[ROW][COL], i) * pow(PixA_CY_GPU[ROW][COL], j);
                                ScaSPixA_Iij_GPU[ij][ROW][COL] = PixA_I_GPU[ROW][COL] * poly;
                            }

                            if (i == -1 || j == -1) {
                                ScaSPixA_Iij_GPU[ij][ROW][COL] = 0.0;
                            } 
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['ScaSpatial'] = _module

            if ScaSpType == 'B-Spline':

                _refdict = {'N0': N0, 'N1': N1, 'Fi': Fi, 'Fj': Fj, 'Fij': Fij}
                _funcstr = r"""
                extern "C" __global__ void kmain(int ScaREF_ij_GPU[%(Fij)s][2],
                    double ScaSplBasisX_GPU[%(Fi)s][%(N0)s], double ScaSplBasisY_GPU[%(Fj)s][%(N1)s],
                    double PixA_I_GPU[%(N0)s][%(N1)s], double ScaSPixA_Iij_GPU[%(Fij)s][%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;
                
                    int N0 = %(N0)s;
                    int N1 = %(N1)s;
                    int Fij = %(Fij)s;
                    
                    if (ROW < N0 && COL < N1) {
                        for(int ij = 0; ij < Fij; ++ij) {
                            int i = ScaREF_ij_GPU[ij][0];
                            int j = ScaREF_ij_GPU[ij][1];
                            double spl = ScaSplBasisX_GPU[i][ROW] * ScaSplBasisY_GPU[j][COL];
                            ScaSPixA_Iij_GPU[ij][ROW][COL] = PixA_I_GPU[ROW][COL] * spl;
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['ScaSpatial'] = _module
        
        # <*****> produce Tpq <*****> #
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

        # ************************************ Constuct Linear System ************************************ #

        # <*****> OMEGA & GAMMA & PSI & PHI & THETA & DELTA <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            if not MINIMIZE_GPU_MEMORY_USAGE:
                
                # ** Hadamard Product [OMEGA]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FOMG': FOMG, 'TpB': MAX_THREADS_PER_BLOCK}
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

                    __shared__ cuDoubleComplex ShSPixA_FIij_GPU[%(Fij)s][%(TpB)s][%(TpB)s];
                    if (ROW < N0 && COL < N1) {
                        for(int ij = 0; ij < Fij; ++ij){
                            ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_FIij_GPU[ij][ROW][COL];
                        }
                    }

                    __shared__ cuDoubleComplex ShSPixA_CFIij_GPU[%(Fij)s][%(TpB)s][%(TpB)s];
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

                # ** Fill Linear-System [OMEGA]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fab': Fab, 'Fijab': Fijab, 'FOMG': FOMG, 'NEQ': NEQ}
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

                # ** Hadamard Product [GAMMA]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FGAM': FGAM, 'TpB': MAX_THREADS_PER_BLOCK}
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
                    
                    __shared__ cuDoubleComplex ShSPixA_FIij_GPU[%(Fij)s][%(TpB)s][%(TpB)s];
                    if (ROW < N0 && COL < N1) {
                        for(int ij = 0; ij < Fij; ++ij){
                            ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_FIij_GPU[ij][ROW][COL];
                        }
                    }

                    __shared__ cuDoubleComplex ShSPixA_CFTpq_GPU[%(Fpq)s][%(TpB)s][%(TpB)s];
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

                # ** Fill Linear-System [GAMMA]
                _refdict = {'N0': N0, 'N1': N1, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, 'FGAM': FGAM, 'NEQ': NEQ}
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

                # ** Hadamard Product [PSI]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FPSI': FPSI, 'TpB': MAX_THREADS_PER_BLOCK}
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
                    
                    __shared__ cuDoubleComplex ShSPixA_FTpq_GPU[%(Fpq)s][%(TpB)s][%(TpB)s];
                    if (ROW < N0 && COL < N1) {
                        for(int pq = 0; pq < Fpq; ++pq){
                            ShSPixA_FTpq_GPU[pq][threadIdx.x][threadIdx.y] = SPixA_FTpq_GPU[pq][ROW][COL];
                        }
                    }

                    __shared__ cuDoubleComplex ShSPixA_CFIij_GPU[%(Fij)s][%(TpB)s][%(TpB)s];
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
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, 'FPSI': FPSI, 'NEQ': NEQ}
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

                # ** Hadamard Product [PHI]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'FPHI': FPHI, 'TpB': MAX_THREADS_PER_BLOCK}
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

                    __shared__ cuDoubleComplex ShSPixA_FTpq_GPU[%(Fpq)s][%(TpB)s][%(TpB)s];
                    if (ROW < N0 && COL < N1) {
                        for(int pq = 0; pq < Fpq; ++pq){
                            ShSPixA_FTpq_GPU[pq][threadIdx.x][threadIdx.y] = SPixA_FTpq_GPU[pq][ROW][COL];
                        }
                    }

                    __shared__ cuDoubleComplex ShSPixA_CFTpq_GPU[%(Fpq)s][%(TpB)s][%(TpB)s];
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
                _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'Fijab': Fijab, 'FPHI': FPHI, 'NEQ': NEQ}
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

            if MINIMIZE_GPU_MEMORY_USAGE:

                # ** Hadamard Product [OMEGA]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FOMG': FOMG}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_iji0j0_GPU[%(FOMG)s][2],
                    cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex SPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpOMG_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {
                        
                        int i8j8 = SREF_iji0j0_GPU[cIdx][0];
                        int ij = SREF_iji0j0_GPU[cIdx][1];

                        cHpOMG_GPU[ROW][COL] = cuCmul(SPixA_FIij_GPU[i8j8][ROW][COL], 
                            SPixA_CFIij_GPU[ij][ROW][COL]);
                    
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_OMG'] = _module

                # ** Fill Linear-System [OMEGA]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fab': Fab, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
                    int cIdx, double cPreOMG_GPU[%(N0)s][%(N1)s], double LHMAT_GPU[%(NEQ)s][%(NEQ)s]) 
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

                        if (idx == cIdx) {
                        
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
                                LHMAT_GPU[ROW][COL] = - cPreOMG_GPU[MODa8][MODb8]
                                                    - cPreOMG_GPU[MOD_a][MOD_b] 
                                                    + cPreOMG_GPU[MODa8_a][MODb8_b] 
                                                    + cPreOMG_GPU[0][0];
                            }

                            if ((a8 == 0 && b8 == 0) && (a != 0 || b != 0)) {
                                LHMAT_GPU[ROW][COL] = cPreOMG_GPU[MOD_a][MOD_b] - cPreOMG_GPU[0][0];
                            }

                            if ((a8 != 0 || b8 != 0) && (a == 0 && b == 0)) {
                                LHMAT_GPU[ROW][COL] = cPreOMG_GPU[MODa8][MODb8] - cPreOMG_GPU[0][0];
                            }

                            if ((a8 == 0 && b8 == 0) && (a == 0 && b == 0)) {
                                LHMAT_GPU[ROW][COL] = cPreOMG_GPU[0][0];
                            }
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_OMG'] = _module

                # ** Hadamard Product [GAMMA]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FGAM': FGAM}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_ijpq_GPU[%(FGAM)s][2], 
                    cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex SPixA_CFTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpGAM_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;
                    
                    if (ROW < N0 && COL < N1) {
                    
                        int i8j8 = SREF_ijpq_GPU[cIdx][0];
                        int pq = SREF_ijpq_GPU[cIdx][1];

                        cHpGAM_GPU[ROW][COL] = cuCmul(SPixA_FIij_GPU[i8j8][ROW][COL], 
                            SPixA_CFTpq_GPU[pq][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_GAM'] = _module

                # ** Fill Linear-System [GAMMA]
                _refdict = {'N0': N0, 'N1': N1, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
                    int cIdx, double cPreGAM_GPU[%(N0)s][%(N1)s], double LHMAT_GPU[%(NEQ)s][%(NEQ)s])
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

                        if (idx == cIdx) {

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
                                LHMAT_GPU[ROW][cCOL] = cPreGAM_GPU[MODa8][MODb8] - cPreGAM_GPU[0][0];
                            }

                            if (a8 == 0 && b8 == 0) {
                                LHMAT_GPU[ROW][cCOL] = cPreGAM_GPU[0][0];
                            }
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_GAM'] = _module

                # ** Hadamard Product [PSI]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FPSI': FPSI}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_pqij_GPU[%(FPSI)s][2],
                    cuDoubleComplex SPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpPSI_GPU[%(N0)s][%(N1)s])
                {    
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {
    
                        int p8q8 = SREF_pqij_GPU[cIdx][0];
                        int ij = SREF_pqij_GPU[cIdx][1];

                        cHpPSI_GPU[ROW][COL] = cuCmul(SPixA_FTpq_GPU[p8q8][ROW][COL], 
                            SPixA_CFIij_GPU[ij][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_PSI'] = _module

                # ** Fill Linear-System [PSI]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
                    int cIdx, double cPrePSI_GPU[%(N0)s][%(N1)s], double LHMAT_GPU[%(NEQ)s][%(NEQ)s])
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

                        if (idx == cIdx) {

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
                                LHMAT_GPU[cROW][COL] = cPrePSI_GPU[MOD_a][MOD_b] - cPrePSI_GPU[0][0];
                            }

                            if (a == 0 && b == 0) {
                                LHMAT_GPU[cROW][COL] = cPrePSI_GPU[0][0];
                            }
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_PSI'] = _module

                # ** Hadamard Product [PHI]
                _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'FPHI': FPHI}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_pqp0q0_GPU[%(FPHI)s][2], 
                    cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    cuDoubleComplex SPixA_CFTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s],  
                    int cIdx, cuDoubleComplex cHpPHI_GPU[%(N0)s][%(N1)s])
                {    
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;
                    int Fpq = %(Fpq)s;
                    int FPHI = %(FPHI)s;

                    if (ROW < N0 && COL < N1) {
                    
                        int p8q8 = SREF_pqp0q0_GPU[cIdx][0];
                        int pq = SREF_pqp0q0_GPU[cIdx][1];

                        cHpPHI_GPU[ROW][COL] = cuCmul(SPixA_FTpq_GPU[p8q8][ROW][COL], 
                            SPixA_CFTpq_GPU[pq][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_PHI'] = _module

                # ** Fill Linear-System [PHI]
                _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int cIdx, double cPrePHI_GPU[%(N0)s][%(N1)s],
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

                        if (idx == cIdx) {
                        
                            // Fill Linear System [C-component]
                            LHMAT_GPU[cROW][cCOL] = cPrePHI_GPU[0][0];
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_PHI'] = _module

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

        if SCALING_MODE == 'SEPARATE-VARYING':
            assert MINIMIZE_GPU_MEMORY_USAGE # Force
            
            if MINIMIZE_GPU_MEMORY_USAGE:
            
                # ** Hadamard Product [OMEGA_11]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FOMG': FOMG}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_iji0j0_GPU[%(FOMG)s][2],
                    cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex SPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpOMG11_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {
                        
                        int i8j8 = SREF_iji0j0_GPU[cIdx][0];
                        int ij = SREF_iji0j0_GPU[cIdx][1];

                        cHpOMG11_GPU[ROW][COL] = cuCmul(SPixA_FIij_GPU[i8j8][ROW][COL], 
                            SPixA_CFIij_GPU[ij][ROW][COL]);
                    
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_OMG11'] = _module

                # ** Hadamard Product [OMEGA_01]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FOMG': FOMG}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_iji0j0_GPU[%(FOMG)s][2],
                    cuDoubleComplex ScaSPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex SPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpOMG01_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {
                        
                        int i8j8 = SREF_iji0j0_GPU[cIdx][0];
                        int ij = SREF_iji0j0_GPU[cIdx][1];

                        cHpOMG01_GPU[ROW][COL] = cuCmul(ScaSPixA_FIij_GPU[i8j8][ROW][COL], 
                            SPixA_CFIij_GPU[ij][ROW][COL]);
                    
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_OMG01'] = _module

                # ** Hadamard Product [OMEGA_10]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FOMG': FOMG}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_iji0j0_GPU[%(FOMG)s][2],
                    cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex ScaSPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpOMG10_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {
                        
                        int i8j8 = SREF_iji0j0_GPU[cIdx][0];
                        int ij = SREF_iji0j0_GPU[cIdx][1];

                        cHpOMG10_GPU[ROW][COL] = cuCmul(SPixA_FIij_GPU[i8j8][ROW][COL], 
                            ScaSPixA_CFIij_GPU[ij][ROW][COL]);
                    
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_OMG10'] = _module

                # ** Hadamard Product [OMEGA_00]  # TODO: redundant, only one element used.
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FOMG': FOMG}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_iji0j0_GPU[%(FOMG)s][2],
                    cuDoubleComplex ScaSPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex ScaSPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpOMG00_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {
                        
                        int i8j8 = SREF_iji0j0_GPU[cIdx][0];
                        int ij = SREF_iji0j0_GPU[cIdx][1];

                        cHpOMG00_GPU[ROW][COL] = cuCmul(ScaSPixA_FIij_GPU[i8j8][ROW][COL], 
                            ScaSPixA_CFIij_GPU[ij][ROW][COL]);
                    
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_OMG00'] = _module

                # ** Fill Linear-System [OMEGA]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fab': Fab, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
                    int cIdx, double cPreOMG11_GPU[%(N0)s][%(N1)s], double cPreOMG01_GPU[%(N0)s][%(N1)s], 
                    double cPreOMG10_GPU[%(N0)s][%(N1)s], double cPreOMG00_GPU[%(N0)s][%(N1)s], 
                    double LHMAT_GPU[%(NEQ)s][%(NEQ)s])
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

                        if (idx == cIdx) {
                        
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
                                LHMAT_GPU[ROW][COL] = - cPreOMG11_GPU[MODa8][MODb8]
                                                      - cPreOMG11_GPU[MOD_a][MOD_b] 
                                                      + cPreOMG11_GPU[MODa8_a][MODb8_b] 
                                                      + cPreOMG11_GPU[0][0];
                            }

                            if ((a8 == 0 && b8 == 0) && (a != 0 || b != 0)) {
                                LHMAT_GPU[ROW][COL] = cPreOMG01_GPU[MOD_a][MOD_b] 
                                                      - cPreOMG01_GPU[0][0];
                            }

                            if ((a8 != 0 || b8 != 0) && (a == 0 && b == 0)) {
                                LHMAT_GPU[ROW][COL] = cPreOMG10_GPU[MODa8][MODb8] 
                                                      - cPreOMG10_GPU[0][0];
                            }

                            if ((a8 == 0 && b8 == 0) && (a == 0 && b == 0)) {
                                LHMAT_GPU[ROW][COL] = cPreOMG00_GPU[0][0];
                            }
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_OMG'] = _module

                # ** Hadamard Product [GAMMA_1]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FGAM': FGAM}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_ijpq_GPU[%(FGAM)s][2], 
                    cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex SPixA_CFTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpGAM1_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;
                    
                    if (ROW < N0 && COL < N1) {
                    
                        int i8j8 = SREF_ijpq_GPU[cIdx][0];
                        int pq = SREF_ijpq_GPU[cIdx][1];

                        cHpGAM1_GPU[ROW][COL] = cuCmul(SPixA_FIij_GPU[i8j8][ROW][COL], 
                            SPixA_CFTpq_GPU[pq][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_GAM1'] = _module

                # ** Hadamard Product [GAMMA_0]  # TODO: redundant, only one element used.
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FGAM': FGAM}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_ijpq_GPU[%(FGAM)s][2], 
                    cuDoubleComplex ScaSPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                    cuDoubleComplex SPixA_CFTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpGAM0_GPU[%(N0)s][%(N1)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;
                    
                    if (ROW < N0 && COL < N1) {
                    
                        int i8j8 = SREF_ijpq_GPU[cIdx][0];
                        int pq = SREF_ijpq_GPU[cIdx][1];

                        cHpGAM0_GPU[ROW][COL] = cuCmul(ScaSPixA_FIij_GPU[i8j8][ROW][COL], 
                            SPixA_CFTpq_GPU[pq][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_GAM0'] = _module

                # ** Fill Linear-System [GAMMA]
                _refdict = {'N0': N0, 'N1': N1, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
                    int cIdx, double cPreGAM1_GPU[%(N0)s][%(N1)s], double cPreGAM0_GPU[%(N0)s][%(N1)s], 
                    double LHMAT_GPU[%(NEQ)s][%(NEQ)s])
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
                        int cCOL = Fijab + COL;       // add offset

                        if (idx == cIdx) {

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
                                LHMAT_GPU[ROW][cCOL] = cPreGAM1_GPU[MODa8][MODb8] 
                                                       - cPreGAM1_GPU[0][0];
                            }

                            if (a8 == 0 && b8 == 0) {
                                LHMAT_GPU[ROW][cCOL] = cPreGAM0_GPU[0][0];
                            }
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_GAM'] = _module

                # ** Hadamard Product [PSI_1]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FPSI': FPSI}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_pqij_GPU[%(FPSI)s][2],
                    cuDoubleComplex SPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpPSI1_GPU[%(N0)s][%(N1)s])
                {    
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {

                        int p8q8 = SREF_pqij_GPU[cIdx][0];
                        int ij = SREF_pqij_GPU[cIdx][1];

                        cHpPSI1_GPU[ROW][COL] = cuCmul(SPixA_FTpq_GPU[p8q8][ROW][COL], 
                            SPixA_CFIij_GPU[ij][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_PSI1'] = _module

                # ** Hadamard Product [PSI_0]  # TODO: redundant, only one element used.
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq, 'FPSI': FPSI}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_pqij_GPU[%(FPSI)s][2],
                    cuDoubleComplex ScaSPixA_CFIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                    cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    int cIdx, cuDoubleComplex cHpPSI0_GPU[%(N0)s][%(N1)s])
                {    
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;

                    if (ROW < N0 && COL < N1) {

                        int p8q8 = SREF_pqij_GPU[cIdx][0];
                        int ij = SREF_pqij_GPU[cIdx][1];

                        cHpPSI0_GPU[ROW][COL] = cuCmul(SPixA_FTpq_GPU[p8q8][ROW][COL], 
                            ScaSPixA_CFIij_GPU[ij][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_PSI0'] = _module

                # ** Fill Linear-System [PSI]
                _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
                    int cIdx, double cPrePSI1_GPU[%(N0)s][%(N1)s], double cPrePSI0_GPU[%(N0)s][%(N1)s], 
                    double LHMAT_GPU[%(NEQ)s][%(NEQ)s])
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

                        if (idx == cIdx) {

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
                                LHMAT_GPU[cROW][COL] = cPrePSI1_GPU[MOD_a][MOD_b] 
                                                       - cPrePSI1_GPU[0][0];
                            }

                            if (a == 0 && b == 0) {
                                LHMAT_GPU[cROW][COL] = cPrePSI0_GPU[0][0];
                            }
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_PSI'] = _module

                # ** Hadamard Product [PHI]
                _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'FPHI': FPHI}
                _funcstr = r"""
                #include "cuComplex.h"
                extern "C" __global__ void kmain(int SREF_pqp0q0_GPU[%(FPHI)s][2], 
                    cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                    cuDoubleComplex SPixA_CFTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s],  
                    int cIdx, cuDoubleComplex cHpPHI_GPU[%(N0)s][%(N1)s])
                {    
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int N0 = %(N0)s;
                    int N1 = %(N1)s;
                    int Fpq = %(Fpq)s;
                    int FPHI = %(FPHI)s;

                    if (ROW < N0 && COL < N1) {
                    
                        int p8q8 = SREF_pqp0q0_GPU[cIdx][0];
                        int pq = SREF_pqp0q0_GPU[cIdx][1];

                        cHpPHI_GPU[ROW][COL] = cuCmul(SPixA_FTpq_GPU[p8q8][ROW][COL], 
                            SPixA_CFTpq_GPU[pq][ROW][COL]);
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
                SFFTModule_dict['HadProd_PHI'] = _module

                # ** Fill Linear-System [PHI]
                _refdict = {'N0': N0, 'N1': N1, 'Fpq': Fpq, 'Fijab': Fijab, 'NEQ': NEQ}
                _funcstr = r"""
                extern "C" __global__ void kmain(int cIdx, double cPrePHI_GPU[%(N0)s][%(N1)s],
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

                        if (idx == cIdx) {
                        
                            // Fill Linear System [C-component]
                            LHMAT_GPU[cROW][cCOL] = cPrePHI_GPU[0][0];
                        }
                    }
                }
                """
                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['FillLS_PHI'] = _module

            # ** Hadamard Product [THETA_1]
            _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FTHE': FTHE}
            _funcstr = r"""
            #include "cuComplex.h"
            extern "C" __global__ void kmain(cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                cuDoubleComplex PixA_CFJ_GPU[%(N0)s][%(N1)s], cuDoubleComplex HpTHE1_GPU[%(FTHE)s][%(N0)s][%(N1)s])
            {    
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int N0 = %(N0)s;
                int N1 = %(N1)s;
                int FTHE = %(FTHE)s;

                if (ROW < N0 && COL < N1) {
                    for(int i8j8 = 0; i8j8 < FTHE; ++i8j8){
                        HpTHE1_GPU[i8j8][ROW][COL] = cuCmul(PixA_CFJ_GPU[ROW][COL], 
                            SPixA_FIij_GPU[i8j8][ROW][COL]);
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
            SFFTModule_dict['HadProd_THE1'] = _module

            # ** Hadamard Product [THETA_0]  # TODO: redundant, only one element used.
            _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'FTHE': FTHE}
            _funcstr = r"""
            #include "cuComplex.h"
            extern "C" __global__ void kmain(cuDoubleComplex ScaSPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s],
                cuDoubleComplex PixA_CFJ_GPU[%(N0)s][%(N1)s], cuDoubleComplex HpTHE0_GPU[%(FTHE)s][%(N0)s][%(N1)s])
            {    
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int N0 = %(N0)s;
                int N1 = %(N1)s;
                int FTHE = %(FTHE)s;

                if (ROW < N0 && COL < N1) {
                    for(int i8j8 = 0; i8j8 < FTHE; ++i8j8){
                        HpTHE0_GPU[i8j8][ROW][COL] = cuCmul(PixA_CFJ_GPU[ROW][COL], 
                            ScaSPixA_FIij_GPU[i8j8][ROW][COL]);
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
            SFFTModule_dict['HadProd_THE0'] = _module

            # ** Fill Linear-System [THETA]  
            _refdict = {'N0': N0, 'N1': N1, 'NEQ': NEQ, 'Fijab': Fijab, 'Fab': Fab, 'FTHE': FTHE}
            _funcstr = r"""
            extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2], 
                double PreTHE1_GPU[%(FTHE)s][%(N0)s][%(N1)s], double PreTHE0_GPU[%(FTHE)s][%(N0)s][%(N1)s], 
                double RHb_GPU[%(NEQ)s]) 
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
                        RHb_GPU[ROW] = PreTHE1_GPU[idx][MODa8][MODb8] 
                                       - PreTHE1_GPU[idx][0][0];
                    }

                    if (a8 == 0 && b8 == 0) {
                        RHb_GPU[ROW] = PreTHE0_GPU[idx][0][0];
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['FillLS_THE'] = _module

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

        # <*****> regularize matrix <*****> #
        if REGULARIZE_KERNEL:

            _refdict = {'Fab': Fab}
            _funcstr = r"""
            extern "C" __global__ void kmain(int LAPMAT_GPU[%(Fab)s][%(Fab)s],
                int RRF_GPU[%(Fab)s], int CCF_GPU[%(Fab)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int Fab = %(Fab)s;

                if (ROW < Fab && COL < Fab) {
                    if (ROW != COL) {

                        int r1 = RRF_GPU[ROW];
                        int c1 = CCF_GPU[ROW];
                        int r2 = RRF_GPU[COL];
                        int c2 = CCF_GPU[COL];

                        if (r2 == r1-1 && c2 == c1) {
                            LAPMAT_GPU[ROW][COL] = -1;
                        }
                        if (r2 == r1+1 && c2 == c1) {
                            LAPMAT_GPU[ROW][COL] = -1;
                        }
                        if (r2 == r1 && c2 == c1-1) {
                            LAPMAT_GPU[ROW][COL] = -1;
                        }
                        if (r2 == r1 && c2 == c1+1) {
                            LAPMAT_GPU[ROW][COL] = -1;
                        }   
                    }
                }
            }
            """

            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['fill_lapmat_nondiagonal'] = _module

            c0 = w0*L1+w1
            _refdict = {'Fab': Fab, 'c0': c0}
            _funcstr = r"""
            extern "C" __global__ void kmain(int iREGMAT_GPU[%(Fab)s][%(Fab)s],
                int LTLMAT_GPU[%(Fab)s][%(Fab)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int Fab = %(Fab)s;
                int c0 = %(c0)s;

                if (ROW < Fab && COL < Fab) {
                    
                    if (ROW != c0 && COL != c0) {
                        iREGMAT_GPU[ROW][COL] = LTLMAT_GPU[ROW][COL] + LTLMAT_GPU[COL][ROW]
                                                - LTLMAT_GPU[c0][ROW] - LTLMAT_GPU[c0][COL]
                                                - LTLMAT_GPU[ROW][c0] - LTLMAT_GPU[COL][c0]
                                                + 2 * LTLMAT_GPU[c0][c0];
                    }
                    
                    if (ROW != c0 && COL == c0) {
                        iREGMAT_GPU[ROW][COL] = LTLMAT_GPU[ROW][c0] + LTLMAT_GPU[c0][ROW] 
                                                - 2 * LTLMAT_GPU[c0][c0];
                    }
                    
                    if (ROW == c0 && COL != c0) {
                        iREGMAT_GPU[ROW][COL] = LTLMAT_GPU[COL][c0] + LTLMAT_GPU[c0][COL]
                                                - 2 * LTLMAT_GPU[c0][c0];
                    }
                    
                    if (ROW == c0 && COL == c0) {
                        iREGMAT_GPU[ROW][COL] = 2 * LTLMAT_GPU[c0][c0];
                    }
                }
            }
            """

            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            SFFTModule_dict['fill_iregmat'] = _module

            if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

                SCALE2 = SCALE**2
                _refdict = {'Fij': Fij, 'Fab': Fab, 'Fijab': Fijab, 'NEQ': NEQ, 'SCALE2': SCALE2, 'RemSymbol': '%'}
                _funcstr = r"""
                extern "C" __global__ void kmain(int iREGMAT_GPU[%(Fab)s][%(Fab)s],
                    double SSTMAT_GPU[%(Fij)s][%(Fij)s], 
                    double REGMAT_GPU[%(NEQ)s][%(NEQ)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int Fijab = %(Fijab)s;
                    int Fab = %(Fab)s;
                    double SCALE2 = %(SCALE2)s;
                    
                    if (ROW < Fijab && COL < Fijab) {
                        int k = ROW / Fab;
                        int c = ROW %(RemSymbol)s Fab;
                        int k8 = COL / Fab;
                        int c8 = COL %(RemSymbol)s Fab;
                        
                        REGMAT_GPU[ROW][COL] = SCALE2 * SSTMAT_GPU[k][k8] * iREGMAT_GPU[c][c8];
                    }
                }
                """

                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['fill_regmat'] = _module
            
            if SCALING_MODE == 'SEPARATE-VARYING':
                
                c0 = w0*L1+w1
                SCALE2 = SCALE**2
                _refdict = {'Fij': Fij, 'Fab': Fab, 'c0': c0, 'Fijab': Fijab, 'NEQ': NEQ, 'SCALE2': SCALE2, 'RemSymbol': '%'}
                _funcstr = r"""
                extern "C" __global__ void kmain(int iREGMAT_GPU[%(Fab)s][%(Fab)s],
                    double SSTMAT_GPU[%(Fij)s][%(Fij)s], 
                    double CSSTMAT_GPU[%(Fij)s][%(Fij)s],
                    double DSSTMAT_GPU[%(Fij)s][%(Fij)s],
                    double REGMAT_GPU[%(NEQ)s][%(NEQ)s])
                {
                    int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                    int COL = blockIdx.y*blockDim.y+threadIdx.y;

                    int Fab = %(Fab)s;
                    int c0 = %(c0)s;
                    int Fijab = %(Fijab)s;
                    double SCALE2 = %(SCALE2)s;
                    
                    if (ROW < Fijab && COL < Fijab) {
                        int k = ROW / Fab;
                        int c = ROW %(RemSymbol)s Fab;
                        int k8 = COL / Fab;
                        int c8 = COL %(RemSymbol)s Fab;
                        
                        if (c != c0 && c8 != c0) {
                            REGMAT_GPU[ROW][COL] = SCALE2 * SSTMAT_GPU[k][k8] * iREGMAT_GPU[c][c8];
                        }

                        if (c != c0 && c8 == c0) {
                            REGMAT_GPU[ROW][COL] = SCALE2 * CSSTMAT_GPU[k][k8] * iREGMAT_GPU[c][c8];
                        }
                        
                        if (c == c0 && c8 != c0) {
                            REGMAT_GPU[ROW][COL] = SCALE2 * CSSTMAT_GPU[k8][k] * iREGMAT_GPU[c][c8];
                        }
                        
                        if (c == c0 && c8 == c0) {
                            REGMAT_GPU[ROW][COL] = SCALE2 * DSSTMAT_GPU[k][k8] * iREGMAT_GPU[c][c8];
                        }
                    }
                }
                """

                _code = _funcstr % _refdict
                _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
                SFFTModule_dict['fill_regmat'] = _module

        # <*****> Tweak Linear-System & Restore Solution <*****> #
        if SCALING_MODE == 'SEPARATE-CONSTANT':

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

        if SCALING_MODE == 'SEPARATE-VARYING' and NEQt < NEQ:
            
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
        
        # ************************************ Construct Difference ************************************ #

        # <*****> Construct difference in Fourier space <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _refdict = {'N0': N0, 'N1': N1, 'L0': L0, 'L1': L1, 'w0': w0, 'w1': w1, \
                        'Fij': Fij, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, \
                        'SCALE': SCALE, 'TpB': MAX_THREADS_PER_BLOCK}
            _funcstr = r"""
            #include "cuComplex.h"
            extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2],  
                cuDoubleComplex a_ijab_GPU[%(Fijab)s], 
                cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                cuDoubleComplex Kab_Wla_GPU[%(L0)s][%(N0)s][%(N1)s], 
                cuDoubleComplex Kab_Wmb_GPU[%(L1)s][%(N0)s][%(N1)s], 
                cuDoubleComplex b_pq_GPU[%(Fpq)s], 
                cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                cuDoubleComplex PixA_FJ_GPU[%(N0)s][%(N1)s], 
                cuDoubleComplex PixA_FDIFF_GPU[%(N0)s][%(N1)s])
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

                __shared__ cuDoubleComplex ShSPixA_FIij_GPU[%(Fij)s][%(TpB)s][%(TpB)s];
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
                            PVAL_FKab = cuCmul(SCA, cuCsub(cuCmul(Kab_Wla_GPU[w0 + a][ROW][COL], 
                                Kab_Wmb_GPU[w1 + b][ROW][COL]), ONE));
                        }

                        for(int ij = 0; ij < Fij; ++ij){
                            int ijab = ij * Fab + ab;
                            PVAL = cuCadd(PVAL, cuCmul(cuCmul(a_ijab_GPU[ijab], 
                                ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y]), PVAL_FKab));
                        }
                    }

                    for(int pq = 0; pq < Fpq; ++pq){
                        PVAL = cuCadd(PVAL, cuCmul(b_pq_GPU[pq], 
                            SPixA_FTpq_GPU[pq][ROW][COL]));
                    }
                    
                    PixA_FDIFF_GPU[ROW][COL] = cuCsub(PixA_FJ_GPU[ROW][COL], PVAL);
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=True)
            SFFTModule_dict['Construct_FDIFF'] = _module

        if SCALING_MODE == 'SEPARATE-VARYING':

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _refdict = {'N0': N0, 'N1': N1, 'L0': L0, 'L1': L1, 'w0': w0, 'w1': w1, \
                        'Fij': Fij, 'Fab': Fab, 'Fpq': Fpq, 'Fijab': Fijab, \
                        'SCALE': SCALE, 'TpB': MAX_THREADS_PER_BLOCK}
            _funcstr = r"""
            #include "cuComplex.h"
            extern "C" __global__ void kmain(int SREF_ijab_GPU[%(Fijab)s][2], int REF_ab_GPU[%(Fab)s][2],  
                cuDoubleComplex a_ijab_GPU[%(Fijab)s], 
                cuDoubleComplex SPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                cuDoubleComplex ScaSPixA_FIij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
                cuDoubleComplex Kab_Wla_GPU[%(L0)s][%(N0)s][%(N1)s], 
                cuDoubleComplex Kab_Wmb_GPU[%(L1)s][%(N0)s][%(N1)s], 
                cuDoubleComplex b_pq_GPU[%(Fpq)s], 
                cuDoubleComplex SPixA_FTpq_GPU[%(Fpq)s][%(N0)s][%(N1)s], 
                cuDoubleComplex PixA_FJ_GPU[%(N0)s][%(N1)s], 
                cuDoubleComplex PixA_FDIFF_GPU[%(N0)s][%(N1)s])
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

                __shared__ cuDoubleComplex ShSPixA_FIij_GPU[%(Fij)s][%(TpB)s][%(TpB)s];
                if (ROW < N0 && COL < N1) {
                    for(int ij = 0; ij < Fij; ++ij){
                        ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y] = SPixA_FIij_GPU[ij][ROW][COL];
                    }
                }

                __shared__ cuDoubleComplex ShScaSPixA_FIij_GPU[%(Fij)s][%(TpB)s][%(TpB)s];
                if (ROW < N0 && COL < N1) {
                    for(int ij = 0; ij < Fij; ++ij){
                        ShScaSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y] = ScaSPixA_FIij_GPU[ij][ROW][COL];
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

                            for(int ij = 0; ij < Fij; ++ij){
                                int ijab = ij * Fab + ab;
                                PVAL = cuCadd(PVAL, cuCmul(cuCmul(a_ijab_GPU[ijab], 
                                    ShScaSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y]), PVAL_FKab));
                            }
                        }
                        
                        if (a != 0 || b != 0) {
                            PVAL_FKab = cuCmul(SCA, cuCsub(cuCmul(Kab_Wla_GPU[w0 + a][ROW][COL], 
                                Kab_Wmb_GPU[w1 + b][ROW][COL]), ONE));

                            for(int ij = 0; ij < Fij; ++ij){
                                int ijab = ij * Fab + ab;
                                PVAL = cuCadd(PVAL, cuCmul(cuCmul(a_ijab_GPU[ijab], 
                                    ShSPixA_FIij_GPU[ij][threadIdx.x][threadIdx.y]), PVAL_FKab));
                            }
                        }
                    }

                    for(int pq = 0; pq < Fpq; ++pq){
                        PVAL = cuCadd(PVAL, cuCmul(b_pq_GPU[pq], 
                            SPixA_FTpq_GPU[pq][ROW][COL]));
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
        SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        REGULARIZE_KERNEL=False, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=None, WEIGHT_REGULARIZE=None, \
        LAMBDA_REGULARIZE=1e-6, NUM_CPU_THREADS_4SUBTRACT=8, MAX_THREADS_PER_BLOCK=8, MINIMIZE_GPU_MEMORY_USAGE=False, VERBOSE_LEVEL=2):

        import numba as nb
        nb.set_num_threads = NUM_CPU_THREADS_4SUBTRACT

        N0, N1 = int(NX), int(NY)
        w0, w1 = int(KerHW), int(KerHW)

        SCALE = np.float64(1/(N0*N1))     # Scale of Image-Size
        SCALE_L = np.float64(1/SCALE)     # Reciprocal Scale of Image-Size

        # kernel spatial variation
        DK = int(KerSpDegree)
        assert DK >= 0
        assert KerSpType in ['Polynomial', 'B-Spline']
        if KerSpType == 'B-Spline' and DK == 0:
            assert len(KerIntKnotX) == 0  # otherwise, discontinuity
            assert len(KerIntKnotY) == 0  # otherwise, discontinuity
        
        # scaling spatial variation
        if SEPARATE_SCALING:
            DS = int(ScaSpDegree)
            assert DS >= 0
            assert ScaSpType in ['Polynomial', 'B-Spline']
            if ScaSpType == 'B-Spline' and DS == 0:
                assert len(ScaIntKnotX) == 0  # otherwise, discontinuity
                assert len(ScaIntKnotY) == 0  # otherwise, discontinuity

        # Remarks on SCALING_MODE
        # SEPARATE_SCALING & ScaSpDegree >>>      SCALING_MODE
        #        N         &     any     >>>       'ENTANGLED'
        #        Y         &      0      >>>   'SEPARATE-CONSTANT'
        #        Y         &     > 0     >>>   'SEPARATE-VARYING'

        SCALING_MODE = None
        if not SEPARATE_SCALING:
            SCALING_MODE = 'ENTANGLED'
        elif ScaSpDegree == 0:
            SCALING_MODE = 'SEPARATE-CONSTANT'
        else: SCALING_MODE = 'SEPARATE-VARYING'
        assert SCALING_MODE is not None

        if SCALING_MODE == 'SEPARATE-CONSTANT':
            assert KerSpDegree != 0   # otherwise, reduced to ENTANGLED

        if SCALING_MODE == 'SEPARATE-VARYING':
            # force to activate MINIMIZE_GPU_MEMORY_USAGE
            #assert MINIMIZE_GPU_MEMORY_USAGE

            if KerSpType == 'Polynomial' and ScaSpType == 'Polynomial':
                assert ScaSpDegree != KerSpDegree   # otherwise, reduced to ENTANGLED
            
            if KerSpType == 'B-Spline' and ScaSpType == 'B-Spline':
                if np.all(KerIntKnotX == ScaIntKnotX) and np.all(KerIntKnotY == ScaIntKnotY):
                    assert ScaSpDegree != KerSpDegree   # otherwise, reduced to ENTANGLED
        
        # background spatial variation
        DB = int(BkgSpDegree)
        assert DB >= 0
        assert BkgSpType in ['Polynomial', 'B-Spline']
        if BkgSpType == 'B-Spline' and BkgSpDegree == 0:
            assert len(BkgIntKnotX) == 0  # otherwise, discontinuity
            assert len(BkgIntKnotY) == 0  # otherwise, discontinuity

        # NOTE input image should not has dramatically small size
        assert N0 > MAX_THREADS_PER_BLOCK and N1 > MAX_THREADS_PER_BLOCK

        if REGULARIZE_KERNEL:
            assert XY_REGULARIZE is not None
            assert len(XY_REGULARIZE.shape) == 2 and XY_REGULARIZE.shape[1] == 2

        if VERBOSE_LEVEL in [1, 2]:
            print('--//--//--//--//-- TRIGGER SFFT COMPILATION [Numpy] --//--//--//--//--')

            if KerSpType == 'Polynomial':
                print('---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(KerSpDegree, KerHW))
                if not SEPARATE_SCALING:
                    print('---//--- [ENTANGLED] Polynomial Scaling | KerSpDegree %d ---//---' %KerSpDegree)

            if KerSpType == 'B-Spline':
                print('---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                      %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree, KerHW))
                if not SEPARATE_SCALING: 
                    print('---//--- [ENTANGLED] B-Spline Scaling | Internal Knots %d,%d | KerSpDegree %d ---//---' \
                          %(len(KerIntKnotX), len(KerIntKnotY), KerSpDegree))
            
            if SEPARATE_SCALING:
                if ScaSpType == 'Polynomial':
                    print('---//--- [SEPARATE] Polynomial Scaling | ScaSpDegree %d ---//---' %ScaSpDegree)
                
                if ScaSpType == 'B-Spline':
                    print('---//--- [SEPARATE] B-Spline Scaling | Internal Knots %d,%d | ScaSpDegree %d ---//---' \
                          %(len(ScaIntKnotX), len(ScaIntKnotY), ScaSpDegree))
            
            if BkgSpType == 'Polynomial':
                print('---//--- Polynomial Background | BkgSpDegree %d ---//---' %BkgSpDegree)
            
            if BkgSpType == 'B-Spline':
                print('---//--- B-Spline Background | Internal Knots %d,%d | BkgSpDegree %d ---//---' \
                    %(len(BkgIntKnotX), len(BkgIntKnotY), BkgSpDegree))

        SFFTParam_dict = {}
        SFFTParam_dict['KerHW'] = KerHW
        SFFTParam_dict['KerSpType'] = KerSpType
        SFFTParam_dict['KerSpDegree'] = KerSpDegree
        SFFTParam_dict['KerIntKnotX'] = KerIntKnotX
        SFFTParam_dict['KerIntKnotY'] = KerIntKnotY

        SFFTParam_dict['SEPARATE_SCALING'] = SEPARATE_SCALING
        if SEPARATE_SCALING:
            SFFTParam_dict['ScaSpType'] = ScaSpType
            SFFTParam_dict['ScaSpDegree'] = ScaSpDegree
            SFFTParam_dict['ScaIntKnotX'] = ScaIntKnotX
            SFFTParam_dict['ScaIntKnotY'] = ScaIntKnotY

        SFFTParam_dict['BkgSpType'] = BkgSpType
        SFFTParam_dict['BkgSpDegree'] = BkgSpDegree
        SFFTParam_dict['BkgIntKnotX'] = BkgIntKnotX
        SFFTParam_dict['BkgIntKnotY'] = BkgIntKnotY

        SFFTParam_dict['REGULARIZE_KERNEL'] = REGULARIZE_KERNEL
        SFFTParam_dict['IGNORE_LAPLACIAN_KERCENT'] = IGNORE_LAPLACIAN_KERCENT
        SFFTParam_dict['XY_REGULARIZE'] = XY_REGULARIZE
        SFFTParam_dict['WEIGHT_REGULARIZE'] = WEIGHT_REGULARIZE
        SFFTParam_dict['LAMBDA_REGULARIZE'] = LAMBDA_REGULARIZE

        SFFTParam_dict['NUM_CPU_THREADS_4SUBTRACT'] = NUM_CPU_THREADS_4SUBTRACT
        SFFTParam_dict['MAX_THREADS_PER_BLOCK'] = MAX_THREADS_PER_BLOCK
        SFFTParam_dict['MINIMIZE_GPU_MEMORY_USAGE'] = MINIMIZE_GPU_MEMORY_USAGE
        
        # * Make a dictionary for SFFT parameters
        L0 = 2*w0+1                                       # matching-kernel XSize
        L1 = 2*w1+1                                       # matching-kernel YSize
        Fab = L0*L1                                       # dof for index ab

        if KerSpType == 'Polynomial':
            Fi, Fj = -1, -1                               # not independent, placeholder
            Fij = ((DK+1)*(DK+2))//2                      # dof for matching-kernel polynomial index ij 
        
        if KerSpType == 'B-Spline':
            Fi = len(KerIntKnotX) + KerSpDegree + 1       # dof for matching-kernel B-spline index i (control points/coefficients)
            Fj = len(KerIntKnotY) + KerSpDegree + 1       # dof for matching-kernel B-spline index j (control points/coefficients)
            Fij = Fi*Fj                                   # dof for matching-kernel B-spline index ij
        
        if BkgSpType == 'Polynomial':
            Fp, Fq = -1, -1                               # not independent, placeholder
            Fpq = ((DB+1)*(DB+2))//2                      # dof for diff-background polynomial index pq 
        
        if BkgSpType == 'B-Spline':
            Fp = len(BkgIntKnotX) + BkgSpDegree + 1       # dof for diff-background B-spline index p (control points/coefficients)
            Fq = len(BkgIntKnotY) + BkgSpDegree + 1       # dof for diff-background B-spline index q (control points/coefficients)  
            Fpq = Fp*Fq                                   # dof for diff-background B-spline index pq

        if SCALING_MODE == 'SEPARATE-VARYING':
            if ScaSpType == 'Polynomial':
                ScaFi, ScaFj = -1, -1                     # not independent, placeholder
                ScaFij = ((DS+1)*(DS+2))//2               # effective dof for scaling polynomial index ij
            
            if ScaSpType == 'B-Spline':
                ScaFi = len(ScaIntKnotX) + ScaSpDegree + 1    # dof for scaling B-spline index i (control points/coefficients)
                ScaFj = len(ScaIntKnotY) + ScaSpDegree + 1    # dof for scaling B-spline index j (control points/coefficients)
                ScaFij = ScaFi*ScaFj                          # effective dof for scaling B-spline index ij
            
            # Remarks on the scaling effective dof
            # I. current version not support scaling effective dof no higher than kernel variation.
            #    for simplicity, we use trivail zero basis as placeholder so that 
            #    the apparent dof of scaling and kernel are consistent.
            # II. ScaFij = Fij is allowed, e.g.m, B-Spline, same degree and 
            #     same number of internal knots but at different positions.

            assert ScaFij <= Fij
        
        Fijab = Fij*Fab                                   # Linear-System Major side-length
        FOMG, FGAM, FTHE = Fij**2, Fij*Fpq, Fij           # OMG / GAM / THE has shape (dof, N0, N1)
        FPSI, FPHI, FDEL = Fpq*Fij, Fpq**2, Fpq           # PSI / PHI / DEL has shape (dof, N0, N1)
        NEQ = Fij*Fab+Fpq                                 # Linear-System side-length
        
        NEQt = NEQ
        if SCALING_MODE == 'SEPARATE-CONSTANT':
            NEQt = NEQ-Fij+1                    # tweaked Linear-System side-length for constant scaling

        if SCALING_MODE == 'SEPARATE-VARYING':
            NEQt = NEQ-(Fij-ScaFij)             # tweaked Linear-System side-length for polynomial-varying scaling

        SFFTParam_dict['N0'] = N0               # a.k.a, NX
        SFFTParam_dict['N1'] = N1               # a.k.a, NY
        SFFTParam_dict['w0'] = w0               # a.k.a, KerHW
        SFFTParam_dict['w1'] = w1               # a.k.a, KerHW
        SFFTParam_dict['DK'] = DK               # a.k.a, KerSpDegree
        SFFTParam_dict['DB'] = DB               # a.k.a, BkgSpDegree
        if SEPARATE_SCALING: 
            SFFTParam_dict['DS'] = DS           # a.k.a, ScaSpDegree

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

        if SCALING_MODE == 'SEPARATE-VARYING':
            SFFTParam_dict['ScaFi'] = ScaFi
            SFFTParam_dict['ScaFj'] = ScaFj
            SFFTParam_dict['ScaFij'] = ScaFij        
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

        SFFTModule_dict = {}
        # ************************************ Spatial Variation ************************************ #

        # <*****> produce spatial coordinate X/Y/oX/oY-map <*****> #
        _strdec = 'Tuple((i4[:,:], i4[:,:], f8[:,:], f8[:,:]))(i4[:,:], i4[:,:], f8[:,:], f8[:,:])'
        @nb.njit(_strdec, parallel=True)
        def SpatialVariation(PixA_X, PixA_Y, PixA_CX, PixA_CY):
            
            for ROW in nb.prange(N0):
                for COL in nb.prange(N1):
                    PixA_X[ROW][COL] = ROW
                    PixA_Y[ROW][COL] = COL
                    PixA_CX[ROW][COL] = (float(ROW) + 1.0) / N0
                    PixA_CY[ROW][COL] = (float(COL) + 1.0) / N1

            return PixA_X, PixA_Y, PixA_CX, PixA_CY


        SFFTModule_dict['SpatialCoord'] = SpatialVariation

        # <*****> produce Iij <*****> #
        if KerSpType == 'Polynomial':
            
            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def SpatialPoly(REF_ij, PixA_CX, PixA_CY, PixA_I, SPixA_Iij):
                
                for ij in nb.prange(Fij):
                    i, j = REF_ij[ij]
                    PixA_kpoly = np.power(PixA_CX, i) * np.power(PixA_CY, j)
                    SPixA_Iij[ij,:,:] = PixA_I * PixA_kpoly
                    
                return SPixA_Iij
            
            SFFTModule_dict['KerSpatial'] = SpatialPoly
        
        if KerSpType == 'B-Spline':
            
            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def SpatialBSpline(REF_ij, KerSplBasisX, KerSplBasisY, PixA_I, SPixA_Iij):
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        for ij in range(Fij):
                            i, j = REF_ij[ij]
                            spl = KerSplBasisX[i,ROW] * KerSplBasisY[j,COL]
                            SPixA_Iij[ij,ROW,COL] = PixA_I[ROW,COL] * spl
                
                return SPixA_Iij
            
            SFFTModule_dict['KerSpatial'] = SpatialBSpline
        
        
        
        if SCALING_MODE == 'SEPARATE-VARYING':

            if ScaSpType == 'Polynomial':

                _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def ScaSpPoly(ScaREF_ij, PixA_CX, PixA_CY, PixA_I, ScaSPixA_Iij):
                    
                    for ij in nb.prange(Fij):
                        i, j = ScaREF_ij[ij]
                        poly = np.power(PixA_CX, i) * np.power(PixA_CY, j)
                        ScaSPixA_Iij[ij,:,:] = PixA_I * poly
                    
                    return ScaSPixA_Iij

                SFFTModule_dict['ScaSpatial'] = ScaSpPoly


            if ScaSpType == 'B-Spline':

                _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def ScaSpBSpline(ScaREF_ij, ScaSplBasisX, ScaSplBasisY, PixA_I, ScaSPixA_Iij):
                    
                    for ROW in nb.prange(N0):
                        for COL in nb.prange(N1):
                            for ij in nb.prange(Fij):
                                i, j = ScaREF_ij[ij]
                                spl = ScaSplBasisX[i,ROW] * ScaSplBasisY[j,COL]
                                ScaSPixA_Iij[ij,ROW,COL] = PixA_I[ROW,COL] * spl
                    
                    return ScaSPixA_Iij
    
                SFFTModule_dict['ScaSpatial'] = ScaSpBSpline
        
    
    
        # <*****> produce Tpq <*****> #
        if BkgSpType == 'Polynomial':

            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def BkgSpaPoly(REF_pq, PixA_CX, PixA_CY, SPixA_Tpq):
                
                for pq in nb.prange(Fpq):
                    p, q = REF_pq[pq]
                    poly_bterm = np.power(PixA_CX, p) * np.power(PixA_CY, q)
                    SPixA_Tpq[pq,:,:] = poly_bterm
                
                return SPixA_Tpq

            SFFTModule_dict['BkgSpatial'] = BkgSpaPoly
        
        if BkgSpType == 'B-Spline':

            _strdec = 'f8[:,:,:]' + '(i4[:,:], f8[:,:], f8[:,:], f8[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def BkgSpaBSpline(REF_pq, BkgSplBasisX, BkgSplBasisY, SPixA_Tpq):
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        for pq in range(Fpq):
                            p, q = REF_pq[pq]
                            spl_kterm = BkgSplBasisX[p][ROW] * BkgSplBasisY[q][COL]
                            SPixA_Tpq[pq,ROW,COL] = spl_kterm
                
                return SPixA_Tpq

            SFFTModule_dict['BkgSpatial'] = BkgSpaBSpline


        # ************************************ Constuct Linear System ************************************ #

        # <*****> OMEGA & GAMMA & PSI & PHI & THETA & DELTA <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            if not MINIMIZE_GPU_MEMORY_USAGE:
                
                # ** Hadamard Product [OMEGA]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, HpOMG):
                    
                    for i8j8ij in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[i8j8ij]
                        HpOMG[i8j8ij,:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                        
                    return HpOMG
                
                
                SFFTModule_dict['HadProd_OMG'] = HadProd_OMG

                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, PreOMG, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int( np.fmod(float(a8), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MODa8 = tmp

                            tmp = int( np.fmod(float(b8), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MODb8 = tmp
                            
                            tmp = int( np.fmod(float(-a), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MOD_a = tmp
                            
                            tmp = int( np.fmod(float(-b), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MOD_b = tmp
                            
                            tmp = int( np.fmod(float(a8-a), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MODa8_a = tmp
                            
                            tmp = int( np.fmod(float(b8-b), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MODb8_b = tmp
                            
                            # ** fill linear system [A-component]
                            if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = - PreOMG[idx, MODa8, MODb8] - PreOMG[idx, MOD_a, MOD_b] + PreOMG[idx, MODa8_a, MODb8_b] + PreOMG[idx, 0, 0]    
                                                  
                            if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = PreOMG[idx, MOD_a, MOD_b] - PreOMG[idx, 0, 0]
                                
                            if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG[idx, MODa8, MODb8] - PreOMG[idx, 0, 0]
                                
                            if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG[idx, 0, 0]
                                
                    return LHMAT              
                            
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG


                # ** Hadamard Product [GAMMA]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, HpGAM):
                    
                    for i8j8pq in nb.prange(FGAM):
                        i8j8, pq = SREF_ijpq[i8j8pq]
                        HpGAM[i8j8pq,:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                        
                    return HpGAM
                
                SFFTModule_dict['HadProd_GAM'] = HadProd_GAM


                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, PreGAM, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL         # add offset
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int( np.fmod(float(a8), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MODa8 = tmp
                            
                            tmp = int( np.fmod(float(b8), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MODb8 = tmp
                            
                            # ** fill linear system [B-component]
                            if (a8 != 0 or b8 != 0):
                                LHMAT[ROW, cCOL] = PreGAM[idx, MODa8, MODb8] - PreGAM[idx, 0, 0]
                                
                            if (a8 == 0 and b8 == 0):
                                LHMAT[ROW, cCOL] = PreGAM[idx, 0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM

                # ** Hadamard Product [PSI]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_PSI(SREF_pqij, SPixA_CFIij, SPixA_FTpq, HpPSI):
                    
                    for p8q8ij in nb.prange(FPSI):
                        p8q8, ij = SREF_pqij[p8q8ij]
                        HpPSI[p8q8ij,:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                        
                    return HpPSI
                
                SFFTModule_dict['HadProd_PSI'] = HadProd_PSI

                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, PrePSI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            cROW = Fijab + ROW      # add offset
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int( np.fmod(float(-a), float(N0)) )
                            if tmp < 0.: tmp += N0
                            MOD_a = tmp
                            
                            tmp = int( np.fmod(float(-b), float(N1)) )
                            if tmp < 0.: tmp += N1
                            MOD_b = tmp
                            
                            # ** fill linear system [B#-component]
                            if (a != 0 or b != 0):
                                LHMAT[cROW, COL] = PrePSI[idx, MOD_a, MOD_b] - PrePSI[idx, 0, 0]
                                
                            if (a == 0 and b == 0):
                                LHMAT[cROW, COL] = PrePSI[idx, 0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI


                # ** Hadamard Product [PHI]
                # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, HpPHI):
                    
                    for p8q8pq in nb.prange(FPHI):
                        p8q8, pq = SREF_pqp0q0[p8q8pq]
                        HpPHI[p8q8pq,:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                        
                    return HpPHI
                
                SFFTModule_dict['HadProd_PHI'] = HadProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(PrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            cROW = Fijab + ROW      # add offset
                            cCOL = Fijab + COL      # add offset
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            # ** fill linear system [C-component]
                            LHMAT[cROW, cCOL] = PrePHI[idx, 0, 0]
                            
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI


            if MINIMIZE_GPU_MEMORY_USAGE:
                
                # ** Hadamard Product [𝛀]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, cIdx, cHpOMG):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG[:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                        
                    return cHpOMG

                SFFTModule_dict['HadProd_OMG'] = HadProd_OMG



                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, cIdx, cPreOMG, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # *** analysze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                            if (idx == cIdx):
                                
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
                                    LHMAT[ROW, COL] = - cPreOMG[MODa8, MODb8] \
                                                      - cPreOMG[MOD_a, MOD_b] \
                                                      + cPreOMG[MODa8_a, MODb8_b] \
                                                      + cPreOMG[0, 0]   # NOTE UPDATE

                                if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                    LHMAT[ROW, COL] = cPreOMG[MOD_a, MOD_b] - cPreOMG[0, 0]   # NOTE UPDATE

                                if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG[MODa8, MODb8] - cPreOMG[0, 0]   # NOTE UPDATE

                                if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG[0, 0]   # NOTE UPDATE
                                    
                    return LHMAT
                
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG

                # ** Hadamard Product [GAMMA]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, cIdx, cHpGAM):
                    
                    i8j8, pq = SREF_ijpq[cIdx]
                    cHpGAM[:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                        
                    return cHpGAM
                
                SFFTModule_dict['HadProd_GAM'] = HadProd_GAM

                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, cIdx, cPreGAM, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # *** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(a8), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8 = tmp
                                
                                tmp = int(np.fmod(float(b8), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8 = tmp
                                
                                # ** fill linear system [B-component]
                                if (a8 != 0 or b8 != 0):
                                    LHMAT[ROW, cCOL] = cPreGAM[MODa8, MODb8] - cPreGAM[0, 0]
                                    
                                if (a8 == 0 and b8 == 0):
                                    LHMAT[ROW, cCOL] = cPreGAM[0, 0]
                                    
                    return LHMAT
                                
                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM


                # ** Hadamard Product [PSI]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI(SREF_pqij, SPixA_CFIij, SPixA_FTpq, cIdx, cHpPSI):
                    
                    p8q8, ij = SREF_pqij[cIdx]
                    cHpPSI[:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                    
                    return cHpPSI
                
                SFFTModule_dict['HadProd_PSI'] = HamProd_PSI

                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, cIdx, cPrePSI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # *** analyze index
                            cROW = Fijab + ROW
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MOD_a = tmp
                                
                                tmp = int(np.fmod(float(-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MOD_b = tmp
                                
                                # ** fill linear system [B#-component]
                                if (a != 0 or b != 0):
                                    LHMAT[cROW, COL] = cPrePSI[MOD_a, MOD_b] - cPrePSI[0, 0]
                                    
                                if (a == 0 and b == 0):
                                    LHMAT[cROW, COL] = cPrePSI[0, 0]
                                    
                    return LHMAT
            
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI

                # ** Hadamard Product [PHI]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, cIdx, cHpPHI):
                    
                    p8q8, pq = SREF_pqp0q0[cIdx]
                    cHpPHI[:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                    
                    return cHpPHI

                SFFTModule_dict['HadProd_PHI'] = HamProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(cIdx, cPrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            cROW = Fijab + ROW     # add offset
                            cCOL = Fijab + COL     # add offset
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            if idx == cIdx:
                                LHMAT[cROW, cCOL] = cPrePHI[0, 0]
                            
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI





            # ** Hadamard Product [THETA]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HadProd_THE(SPixA_FIij, PixA_CFJ, HpTHE):
                
                for i8j8 in nb.prange(FTHE):
                    HpTHE[i8j8,:,:] = PixA_CFJ * SPixA_FIij[i8j8]
                    
                return HpTHE
                
            SFFTModule_dict['HadProd_THE'] = HadProd_THE

            # ** Fill Linear-System [THETA]  
            _strdec = 'f8[:](i4[:,:], i4[:,:], f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_THE(SREF_ijab, REF_ab, PreTHE, RHb):
                
                for ROW in nb.prange(Fijab):
                    
                    # ** analyze index
                    i8j8, a8b8 = SREF_ijab[ROW]
                    a8, b8 = REF_ab[a8b8]
                    idx = i8j8
                    
                    # ** define Mod_N0(rho), Mod_N1(eps)
                    tmp = int(np.fmod(float(a8), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MODa8 = tmp
                    
                    tmp = int(np.fmod(float(b8), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MODb8 = tmp
                    
                    # ** fill linear system [D-component]
                    if (a8 != 0 or b8 != 0):
                        RHb[ROW] = PreTHE[idx, MODa8, MODb8] - PreTHE[idx, 0, 0]
                        
                    if (a8 == 0 and b8 == 0):
                        RHb[ROW] = PreTHE[idx, 0, 0]
                        
                return RHb
        
            SFFTModule_dict['FillLS_THE'] = FillLS_THE

            # ** Hadamard Product [DELTA]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HadProd_DEL(SPixA_FTpq, PixA_CFJ, HpDEL):
                
                for p8q8 in nb.prange(FDEL):
                    HpDEL[p8q8, :, :] = PixA_CFJ * SPixA_FTpq[p8q8]
                    
                return HpDEL
                
            SFFTModule_dict['HadProd_DEL'] = HadProd_DEL

            # ** Fill Linear-System [DELTA]
            _strdec = 'f8[:](f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_DEL(PreDEL, RHb):
                
                for ROW in nb.prange(Fpq):
                    cROW = Fijab + ROW
                    idx = ROW
                    
                    # ** Fill Linear System [E-component]
                    RHb[cROW] = PreDEL[idx, 0, 0]
                    
                return RHb
            
            SFFTModule_dict['FillLS_DEL'] = FillLS_DEL





        if SCALING_MODE == 'SEPARATE-VARYING':
            ###assert MINIMIZE_GPU_MEMORY_USAGE # Force  
            
            if not MINIMIZE_GPU_MEMORY_USAGE:
                
                # ** Hadamard Product [OMEGA_11]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG11(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, HpOMG11):
                    
                    for cIdx in nb.prange(FOMG):                    
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG11[cIdx,:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return HpOMG11
                
                SFFTModule_dict['HadProd_OMG11'] = HadProd_OMG11

                # ** Hadamard Product [OMEGA_01]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG01(SREF_iji0j0, ScaSPixA_FIij, SPixA_CFIij, HpOMG01):
                    
                    for cIdx in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG01[cIdx,:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return HpOMG01
                
                SFFTModule_dict['HadProd_OMG01'] = HadProd_OMG01

                # ** Hadamard Product [OMEGA_10]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG10(SREF_iji0j0, SPixA_FIij, ScaSPixA_CFIij, HpOMG10):
                    
                    for cIdx in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG10[cIdx,:,:] = SPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return HpOMG10
                
                SFFTModule_dict['HadProd_OMG10'] = HadProd_OMG10

                # ** Hadamard Product [OMEGA_00]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG00(SREF_iji0j0, ScaSPixA_FIij, ScaSPixA_CFIij, HpOMG00):
                    
                    for cIdx in nb.prange(FOMG):
                        i8j8, ij = SREF_iji0j0[cIdx]
                        HpOMG00[cIdx, :,:] = ScaSPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return HpOMG00
                
                SFFTModule_dict['HadProd_OMG00'] = HadProd_OMG00
                
                
                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, PreOMG11, PreOMG01, PreOMG10, PreOMG00, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                                
                            # ** define Mod_N0(rho), Mod_N1(eps)
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
                            
                            # ** fill linear system [A-component]
                            if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = - PreOMG11[idx, MODa8, MODb8] \
                                                    - PreOMG11[idx, MOD_a, MOD_b] \
                                                    + PreOMG11[idx, MODa8_a, MODb8_b] \
                                                    + PreOMG11[idx, 0, 0]
                                                    
                            if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                LHMAT[ROW, COL] = PreOMG01[idx, MOD_a, MOD_b] - PreOMG01[idx, 0, 0]
                                
                            if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG10[idx, MODa8, MODb8] - PreOMG10[idx, 0, 0]
                                
                            if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                LHMAT[ROW, COL] = PreOMG00[idx, 0, 0]
                
                    return LHMAT
    
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG
    
    
    
                    # ** Hadamard Product [GAMMA_1]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM1(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, HpGAM1):
                    
                    for cIdx in nb.prange(FGAM):
                        i8j8, pq = SREF_ijpq[cIdx]
                        HpGAM1[cIdx,:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return HpGAM1
                
                SFFTModule_dict['HadProd_GAM1'] = HadProd_GAM1

                # ** Hadamard Product [GAMMA_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM0(SREF_ijpq, ScaSPixA_FIij, SPixA_CFTpq, HpGAM0):
                    
                    for cIdx in nb.prange(FGAM):
                        i8j8, pq = SREF_ijpq[cIdx]
                        HpGAM0[cIdx,:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return HpGAM0
                
                SFFTModule_dict['HadProd_GAM0'] = HadProd_GAM0

                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, PreGAM1, PreGAM0, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL
                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int(np.fmod(float(a8), float(N0)))
                            if tmp < 0.0: tmp += N0
                            MODa8 = tmp
                            
                            tmp = int(np.fmod(float(b8), float(N1)))
                            if tmp < 0.0: tmp += N1
                            MODb8 = tmp
                            
                            # ** fill linear system [B-component]
                            if (a8 != 0 or b8 != 0):
                                LHMAT[ROW, cCOL] = PreGAM1[idx, MODa8, MODb8] - PreGAM1[idx, 0, 0]
                                
                            if (a8 == 0 and b8 == 0):
                                LHMAT[ROW, cCOL] = PreGAM0[idx, 0, 0]
                                    
                    return LHMAT

                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM
    
    
    
                    # ** Hadamard Product [PSI_1]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI1(SREF_pqij, SPixA_CFIij, SPixA_FTpq, HpPSI1):
                    
                    for cIdx in nb.prange(FPSI):
                        p8q8, ij = SREF_pqij[cIdx]
                        HpPSI1[cIdx,:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                    
                    return HpPSI1

                SFFTModule_dict['HadProd_PSI1'] = HamProd_PSI1

                # ** Hadamard Product [PSI_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI0(SREF_pqij, ScaSPixA_CFIij, SPixA_FTpq, HpPSI0):
                    
                    for cIdx in nb.prange(FPSI):
                        p8q8, ij = SREF_pqij[cIdx]
                        HpPSI0[cIdx,:,:] = SPixA_FTpq[p8q8] * ScaSPixA_CFIij[ij]
                    
                    return HpPSI0
                
                SFFTModule_dict['HadProd_PSI0'] = HamProd_PSI0


                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, PrePSI1, PrePSI0, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij

                            
                            # ** define Mod_N0(rho), Mod_N1(eps)
                            tmp = int(np.fmod(float(-a), float(N0)))
                            if tmp < 0.0: tmp += N0
                            MOD_a = tmp
                            
                            tmp = int(np.fmod(float(-b), float(N1)))
                            if tmp < 0.0: tmp += N1
                            MOD_b = tmp
                            
                            # ** fill linear system [B-component]
                            if (a != 0 or b != 0):
                                LHMAT[cROW, COL] = PrePSI1[idx, MOD_a, MOD_b] - PrePSI1[idx, 0, 0]
                                
                            if (a == 0 and b == 0):
                                LHMAT[cROW, COL] = PrePSI0[idx, 0, 0]

                    return LHMAT                
                
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI




                # ** Hadamard Product [PHI]
                _strdec = 'c16[:,:,:](i4[:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, HpPHI):
                    
                    for cIdx in nb.prange(FPHI):
                        p8q8, pq = SREF_pqp0q0[cIdx]
                        HpPHI[cIdx,:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                    
                    return HpPHI
                
                SFFTModule_dict['HadProd_PHI'] = HamProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](f8[:,:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(PrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            cCOL = Fijab + COL
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            # ** fill linear system [C-component]
                            LHMAT[cROW, cCOL] = PrePHI[idx, 0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI
                
            
            if MINIMIZE_GPU_MEMORY_USAGE:
            
                # ** Hadamard Product [OMEGA_11]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG11(SREF_iji0j0, SPixA_FIij, SPixA_CFIij, cIdx, cHpOMG11):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG11[:,:] = SPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return cHpOMG11
                
                SFFTModule_dict['HadProd_OMG11'] = HadProd_OMG11

                # ** Hadamard Product [OMEGA_01]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG01(SREF_iji0j0, ScaSPixA_FIij, SPixA_CFIij, cIdx, cHpOMG01):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG01[:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFIij[ij]
                    
                    return cHpOMG01
                
                SFFTModule_dict['HadProd_OMG01'] = HadProd_OMG01

                # ** Hadamard Product [OMEGA_10]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG10(SREF_iji0j0, SPixA_FIij, ScaSPixA_CFIij, cIdx, cHpOMG10):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG10[:,:] = SPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return cHpOMG10
                
                SFFTModule_dict['HadProd_OMG10'] = HadProd_OMG10

                # ** Hadamard Product [OMEGA_00]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_OMG00(SREF_iji0j0, ScaSPixA_FIij, ScaSPixA_CFIij, cIdx, cHpOMG00):
                    
                    i8j8, ij = SREF_iji0j0[cIdx]
                    cHpOMG00[:,:] = ScaSPixA_FIij[i8j8] * ScaSPixA_CFIij[ij]
                    
                    return cHpOMG00
                
                SFFTModule_dict['HadProd_OMG00'] = HadProd_OMG00


                # ** Fill Linear-System [OMEGA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_OMG(SREF_ijab, REF_ab, cIdx, cPreOMG11, cPreOMG01, cPreOMG10, cPreOMG00, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            ij, ab = SREF_ijab[COL]
                            
                            a8, b8 = REF_ab[a8b8]
                            a, b = REF_ab[ab]
                            idx = i8j8 * Fij + ij
                            
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
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
                                
                                # ** fill linear system [A-component]
                                if ((a8 != 0 or b8 != 0) and (a != 0 or b != 0)):
                                    LHMAT[ROW, COL] = - cPreOMG11[MODa8, MODb8] \
                                                      - cPreOMG11[MOD_a, MOD_b] \
                                                      + cPreOMG11[MODa8_a, MODb8_b] \
                                                      + cPreOMG11[0, 0]
                                                      
                                if ((a8 == 0 and b8 == 0) and (a != 0 or b != 0)):
                                    LHMAT[ROW, COL] = cPreOMG01[MOD_a, MOD_b] - cPreOMG01[0, 0]
                                    
                                if ((a8 != 0 or b8 != 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG10[MODa8, MODb8] - cPreOMG10[0, 0]
                                    
                                if ((a8 == 0 and b8 == 0) and (a == 0 and b == 0)):
                                    LHMAT[ROW, COL] = cPreOMG00[0, 0]
                
                    return LHMAT
    
                SFFTModule_dict['FillLS_OMG'] = FillLS_OMG


                # ** Hadamard Product [GAMMA_1]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM1(SREF_ijpq, SPixA_FIij, SPixA_CFTpq, cIdx, cHpGAM1):
                    
                    i8j8, pq = SREF_ijpq[cIdx]
                    cHpGAM1[:,:] = SPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return cHpGAM1
                
                SFFTModule_dict['HadProd_GAM1'] = HadProd_GAM1

                # ** Hadamard Product [GAMMA_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HadProd_GAM0(SREF_ijpq, ScaSPixA_FIij, SPixA_CFTpq, cIdx, cHpGAM0):
                    
                    i8j8, pq = SREF_ijpq[cIdx]
                    cHpGAM0[:,:] = ScaSPixA_FIij[i8j8] * SPixA_CFTpq[pq]
                    
                    return cHpGAM0
                
                SFFTModule_dict['HadProd_GAM0'] = HadProd_GAM0

                # ** Fill Linear-System [GAMMA]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_GAM(SREF_ijab, REF_ab, cIdx, cPreGAM1, cPreGAM0, LHMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            i8j8, a8b8 = SREF_ijab[ROW]
                            pq = COL
                            
                            a8, b8 = REF_ab[a8b8]
                            idx = i8j8 * Fpq + pq
                            cCOL = Fijab + COL
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(a8), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MODa8 = tmp
                                
                                tmp = int(np.fmod(float(b8), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MODb8 = tmp
                                
                                # ** fill linear system [B-component]
                                if (a8 != 0 or b8 != 0):
                                    LHMAT[ROW, cCOL] = cPreGAM1[MODa8, MODb8] - cPreGAM1[0, 0]
                                    
                                if (a8 == 0 and b8 == 0):
                                    LHMAT[ROW, cCOL] = cPreGAM0[0, 0]
                                    
                    return LHMAT

                SFFTModule_dict['FillLS_GAM'] = FillLS_GAM

                # ** Hadamard Product [PSI_1]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI1(SREF_pqij, SPixA_CFIij, SPixA_FTpq, cIdx, cHpPSI1):
                    
                    p8q8, ij = SREF_pqij[cIdx]
                    cHpPSI1[:,:] = SPixA_FTpq[p8q8] * SPixA_CFIij[ij]
                    
                    return cHpPSI1

                SFFTModule_dict['HadProd_PSI1'] = HamProd_PSI1

                # ** Hadamard Product [PSI_0]  # TODO: redundant, only one element used.
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PSI0(SREF_pqij, ScaSPixA_CFIij, SPixA_FTpq, cIdx, cHpPSI0):
                    
                    p8q8, ij = SREF_pqij[cIdx]
                    cHpPSI0[:,:] = SPixA_FTpq[p8q8] * ScaSPixA_CFIij[ij]
                    
                    return cHpPSI0
                
                SFFTModule_dict['HadProd_PSI0'] = HamProd_PSI0


                # ** Fill Linear-System [PSI]
                _strdec = 'f8[:,:](i4[:,:], i4[:,:], i4, f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PSI(SREF_ijab, REF_ab, cIdx, cPrePSI1, cPrePSI0, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fijab):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            p8q8 = ROW
                            
                            ij, ab = SREF_ijab[COL]
                            a, b = REF_ab[ab]
                            idx = p8q8 * Fij + ij
                            
                            if (idx == cIdx):
                                
                                # ** define Mod_N0(rho), Mod_N1(eps)
                                tmp = int(np.fmod(float(-a), float(N0)))
                                if tmp < 0.0: tmp += N0
                                MOD_a = tmp
                                
                                tmp = int(np.fmod(float(-b), float(N1)))
                                if tmp < 0.0: tmp += N1
                                MOD_b = tmp
                                
                                # ** fill linear system [B-component]
                                if (a != 0 or b != 0):
                                    LHMAT[cROW, COL] = cPrePSI1[MOD_a, MOD_b] - cPrePSI1[0, 0]
                                    
                                if (a == 0 and b == 0):
                                    LHMAT[cROW, COL] = cPrePSI0[0, 0]

                    return LHMAT                
                
                SFFTModule_dict['FillLS_PSI'] = FillLS_PSI

                # ** Hadamard Product [PHI]
                _strdec = 'c16[:,:](i4[:,:], c16[:,:,:], c16[:,:,:], i4, c16[:,:])'
                @nb.njit(_strdec, parallel=True)
                def HamProd_PHI(SREF_pqp0q0, SPixA_FTpq, SPixA_CFTpq, cIdx, cHpPHI):
                    
                    p8q8, pq = SREF_pqp0q0[cIdx]
                    cHpPHI[:,:] = SPixA_FTpq[p8q8] * SPixA_CFTpq[pq]
                    
                    return cHpPHI
                
                SFFTModule_dict['HadProd_PHI'] = HamProd_PHI

                # ** Fill Linear-System [PHI]
                _strdec = 'f8[:,:](i4, f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def FillLS_PHI(cIdx, cPrePHI, LHMAT):
                    
                    for ROW in nb.prange(Fpq):
                        for COL in nb.prange(Fpq):
                            
                            # ** analyze index
                            cROW = Fijab + ROW
                            cCOL = Fijab + COL
                            
                            p8q8 = ROW
                            pq = COL
                            idx = p8q8 * Fpq + pq
                            
                            # ** fill linear system [C-component]
                            if idx == cIdx:
                                LHMAT[cROW, cCOL] = cPrePHI[0, 0]
                                
                    return LHMAT
                
                SFFTModule_dict['FillLS_PHI'] = FillLS_PHI


            # ** Hadamard Product [THETA_1]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HamProd_THE1(SPixA_FIij, PixA_CFJ, HpTHE1):
                
                for i8j8 in nb.prange(FTHE):
                    HpTHE1[i8j8,:,:] = PixA_CFJ * SPixA_FIij[i8j8]
                    
                return HpTHE1
            
            SFFTModule_dict['HadProd_THE1'] = HamProd_THE1


            # ** Hadamard Product [THETA_0]  # TODO: redundant, only one element used.
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HamProd_THE0(ScaSPixA_FIij, PixA_CFJ, HpTHE0):
                
                for i8j8 in nb.prange(FTHE):
                    HpTHE0[i8j8,:,:] = PixA_CFJ * ScaSPixA_FIij[i8j8]
                    
                return HpTHE0
            
            SFFTModule_dict['HadProd_THE0'] = HamProd_THE0


            # ** Fill Linear-System [THETA]  
            _strdec = 'f8[:](i4[:,:], i4[:,:], f8[:,:,:], f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_THE(SREF_ijab, REF_ab, PreTHE1, PreTHE0, RHb):
                
                for ROW in nb.prange(Fijab):
                    
                    # ** analyze index
                    i8j8, a8b8 = SREF_ijab[ROW]
                    a8, b8 = REF_ab[a8b8]
                    idx = i8j8
                    
                    # ** define Mod_N0(rho), Mod_N1(eps)
                    tmp = int(np.fmod(float(a8), float(N0)))
                    if tmp < 0.0: tmp += N0
                    MODa8 = tmp
                    
                    tmp = int(np.fmod(float(b8), float(N1)))
                    if tmp < 0.0: tmp += N1
                    MODb8 = tmp
                    
                    # ** fill linear system [D-component]
                    if (a8 != 0 or b8 != 0):
                        RHb[ROW] = PreTHE1[idx, MODa8, MODb8] - PreTHE1[idx, 0, 0]
                        
                    if (a8 == 0 and b8 == 0):
                        RHb[ROW] = PreTHE0[idx, 0, 0]
                        
                return RHb
            
            SFFTModule_dict['FillLS_THE'] = FillLS_THE



            # ** Hadamard Product [DELTA]
            _strdec = 'c16[:,:,:](c16[:,:,:], c16[:,:], c16[:,:,:])'
            @nb.njit(_strdec, parallel=True)
            def HamProd_DEL(SPixA_FTpq, PixA_CFJ, HpDEL):
                
                for p8q8 in nb.prange(FDEL):
                    HpDEL[p8q8,:,:] = PixA_CFJ * SPixA_FTpq[p8q8]
                    
                return HpDEL
            
            SFFTModule_dict['HadProd_DEL'] = HamProd_DEL

            # ** Fill Linear-System [DELTA]
            _strdec = 'f8[:](f8[:,:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def FillLS_DEL(PreDEL, RHb):
                
                for ROW in nb.prange(Fpq):
                    
                    # ** analyze index
                    cROW = Fijab + ROW
                    idx = ROW
                    
                    # ** Fill Linear System [E-component]
                    RHb[cROW] = PreDEL[idx, 0, 0]
                    
                return RHb
            
            SFFTModule_dict['FillLS_DEL'] = FillLS_DEL


        # <*****> regularize matrix <*****> #
        if REGULARIZE_KERNEL:
            _strdec = 'i4[:,:](i4[:,:], i4[:], i4[:])'
            @nb.njit(_strdec, parallel=True)
            def Fill_LAPMAT_NonDiagonal(LAPMAT, RRF, CCF):
                
                for ROW in nb.prange(Fab):
                    for COL in nb.prange(Fab):
                        
                        if (ROW != COL):
                            r1 = RRF[ROW]
                            c1 = CCF[ROW]
                            r2 = RRF[COL]
                            c2 = CCF[COL]
                            
                            if (r2 == r1-1 and c2 == c1):
                                LAPMAT[ROW, COL] = -1
                                
                            if (r2 == r1+1 and c2 == c1):
                                LAPMAT[ROW, COL] = -1
                                
                            if (r2 == r1 and c2 == c1-1):
                                LAPMAT[ROW, COL] = -1
                                
                            if (r2 == r1 and c2 == c1+1):
                                LAPMAT[ROW, COL] = -1
                                
                return LAPMAT

            SFFTModule_dict['fill_lapmat_nondiagonal'] = Fill_LAPMAT_NonDiagonal

            c0 = w0*L1+w1
            _strdec = 'i4[:,:](i4[:,:], i4[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Fill_iREGMAT(iREGMAT, LTLMAT):
                
                for ROW in nb.prange(Fab):
                    for COL in nb.prange(Fab):
                        
                        if (ROW != c0 and COL != c0):
                            iREGMAT[ROW, COL] = LTLMAT[ROW, COL] + LTLMAT[COL, ROW] \
                                                - LTLMAT[c0, ROW] - LTLMAT[c0, COL] \
                                                - LTLMAT[ROW, c0] - LTLMAT[COL, c0] \
                                                + 2*LTLMAT[c0, c0]
                                                
                        if (ROW != c0 and COL == c0):
                            iREGMAT[ROW, COL] = LTLMAT[ROW, c0] + LTLMAT[c0, ROW] \
                                                - 2*LTLMAT[c0, c0]
                                                                        
                        if (ROW == c0 and COL != c0):
                            iREGMAT[ROW, COL] = LTLMAT[COL, c0] + LTLMAT[c0, COL] \
                                                - 2*LTLMAT[c0, c0]
                                                
                        if (ROW == c0 and COL == c0):
                            iREGMAT[ROW, COL] = 2*LTLMAT[c0, c0]            
            
                return iREGMAT
            
            SFFTModule_dict['fill_iregmat'] = Fill_iREGMAT



            if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

                SCALE2 = SCALE**2
                _strdec = 'f8[:,:](i4[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def Fill_REGMAT(iREGMAT, SSTMAT, REGMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            k = int(ROW // Fab)
                            c = int(ROW % Fab)
                            k8 = int(COL // Fab)
                            c8 = int(COL % Fab)
                            
                            REGMAT[ROW, COL] = SCALE2 * SSTMAT[k, k8] * iREGMAT[c, c8]
                
                    return REGMAT
                
                SFFTModule_dict['fill_regmat'] = Fill_REGMAT
            
            
            if SCALING_MODE == 'SEPARATE-VARYING':
                
                c0 = w0*L1+w1
                SCALE2 = SCALE**2
                _strdec = 'f8[:,:](i4[:,:], f8[:,:], f8[:,:], f8[:,:], f8[:,:])'
                @nb.njit(_strdec, parallel=True)
                def Fill_REGMAT(iREGMAT, SSTMAT, CSSTMAT, DSSTMAT, REGMAT):
                    
                    for ROW in nb.prange(Fijab):
                        for COL in nb.prange(Fijab):
                            k = int(ROW // Fab)
                            c = int(ROW % Fab)
                            k8 = int(COL // Fab)
                            c8 = int(COL % Fab)
                            
                            if (c != c0 and c8 != c0):
                                REGMAT[ROW, COL] = SCALE2 * SSTMAT[k, k8] * iREGMAT[c, c8]
                                
                            if (c != c0 and c8 == c0):
                                REGMAT[ROW, COL] = SCALE2 * CSSTMAT[k, k8] * iREGMAT[c, c8]
                                
                            if (c == c0 and c8 != c0):
                                REGMAT[ROW, COL] = SCALE2 * CSSTMAT[k8, k] * iREGMAT[c, c8]
                                
                            if (c == c0 and c8 == c0):
                                REGMAT[ROW, COL] = SCALE2 * DSSTMAT[k, k8] * iREGMAT[c, c8]
                                
                    return REGMAT
                
                SFFTModule_dict['fill_regmat'] = Fill_REGMAT

        # <*****> Tweak Linear-System & Restore Solution <*****> #
        if SCALING_MODE == 'SEPARATE-CONSTANT':

            if KerSpType == 'Polynomial':
                
                _strdec = 'Tuple((f8[:,:], f8[:]))' + '(f8[:,:], f8[:], i4[:], f8[:,:], f8[:])'
                @nb.njit(_strdec, parallel=True)
                def TweakLSPoly(LHMAT, RHb, PresIDX, LHMAT_tweaked, RHb_tweaked):
                    
                    for ROW in nb.prange(NEQt):
                        
                        RHb_tweaked[ROW] = RHb[PresIDX[ROW]]
                        
                        for COL in nb.prange(NEQt):
                            LHMAT_tweaked[ROW, COL] = LHMAT[PresIDX[ROW], PresIDX[COL]]

                    return LHMAT_tweaked, RHb_tweaked
                
                SFFTModule_dict['TweakLS'] = TweakLSPoly

            if KerSpType == 'B-Spline':
                
                _strdec = 'Tuple((f8[:,:], f8[:]))' + '(f8[:,:], f8[:], i4[:], i4[:], f8[:,:], f8[:])'
                @nb.njit(_strdec, parallel=True)
                def TweakLSBSpline(LHMAT, RHb, PresIDX, ij00, LHMAT_tweaked, RHb_tweaked):
                    keyIdx = ij00[0]
                    
                    for ROW in nb.prange(NEQt):
                        for COL in nb.prange(NEQt):

                            if (ROW == keyIdx) and (COL != keyIdx) and (COL < NEQt):
                                cum1 = 0.
                                for ij in nb.prange(Fij):
                                    ridx = ij00[ij]
                                    PresCOL = PresIDX[COL]
                                    cum1 += LHMAT[ridx, PresCOL]

                                LHMAT_tweaked[ROW, COL] = cum1
                
                
                            if (ROW != keyIdx) and (COL == keyIdx):
                                cum2 = 0.
                                for ij in nb.prange(Fij):
                                    PresROW = PresIDX[ROW]
                                    cidx = ij00[ij]
                                    cum2 += LHMAT[PresROW, cidx]
                                    
                                LHMAT_tweaked[ROW, COL] = cum2
                                
                                
                            if (ROW == keyIdx) and (COL == keyIdx):
                                cum3 = 0.
                                for ij in nb.prange(Fij):
                                    for i8j8 in nb.prange(Fij):
                                        ridx = ij00[ij]
                                        cidx = ij00[i8j8]
                                        cum3 += LHMAT[ridx, cidx]
                                        
                                LHMAT_tweaked[ROW, COL] = cum3
                                
                                
                            
                            if (ROW != keyIdx) and (COL != keyIdx):
                                PresROW = PresIDX[ROW]
                                PresCOL = PresIDX[COL]
                                LHMAT_tweaked[ROW, COL] = LHMAT[PresROW, PresCOL]
                                
                    return LHMAT_tweaked, RHb_tweaked    
    
                SFFTModule_dict['TweakLS'] = TweakLSBSpline
            
            
            
            
            _strdec = 'f8[:](f8[:], i4[:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def RestoreSolution(Solution_tweaked, PresIDX, Solution):
                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    Solution[PresROW] = Solution_tweaked[ROW]
                return Solution
            
            SFFTModule_dict['Restore_Solution'] = RestoreSolution




        if (SCALING_MODE == 'SEPARATE-VARYING') and (NEQt < NEQ):
            
            _strdec = 'Tuple((f8[:,:], f8[:]))' + '(f8[:,:], f8[:], i4[:], f8[:,:], f8[:])'
            @nb.njit(_strdec, parallel=True)
            def TweakLS(LHMAT, RHb, PresIDX, LHMAT_tweaked, RHb_tweaked):

                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    RHb_tweaked[ROW] = RHb[PresROW]

                    for COL in nb.prange(NEQt):
                        PresCOL = PresIDX[COL]
                        LHMAT_tweaked[ROW, COL] = LHMAT[PresROW, PresCOL]

                return LHMAT_tweaked, RHb_tweaked
            
            SFFTModule_dict['TweakLS'] = TweakLS
            

            _strdec = 'f8[:](f8[:], i4[:], f8[:])'            
            @nb.njit(_strdec, parallel=True)
            def RestoreSolution(Solution_tweaked, PresIDX, Solution):
                
                for ROW in nb.prange(NEQt):
                    PresROW = PresIDX[ROW]
                    Solution[PresROW] = Solution_tweaked[ROW]

                return Solution

            SFFTModule_dict['Restore_Solution'] = RestoreSolution
        
        # ************************************ Construct Difference ************************************ #

        # <*****> Construct difference in Fourier space <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FDIFF(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, PixA_FDIFF):
            
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0) and (b == 0):
                                PVAL_FKab = SCALE_C
                                
                            if (a != 0) or (b != 0):
                                PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)
            
                            
                            for ij in range(Fij):
                                ijab = ij * Fab + ab
                                PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                
                        for pq in range(Fpq):
                            PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]
                            
                        PixA_FDIFF[ROW, COL] = PixA_FJ[ROW, COL] - PVAL
                        
                return PixA_FDIFF
            
            SFFTModule_dict['Construct_FDIFF'] = Construct_FDIFF

        if SCALING_MODE == 'SEPARATE-VARYING':

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FDIFF(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, ScaSPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, PixA_FDIFF):
                
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0 and b == 0):
                                PVAL_FKab = SCALE_C
                                
                                for ij in range(Fij):
                                    ijab = ij * Fab + ab
                                    PVAL += (a_ijab[ijab] * ScaSPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                    
                                    
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
            
            
            
        # ************************************ Construct Coadd ************************************ #

        # <*****> Construct difference in Fourier space <*****> #
        if SCALING_MODE in ['ENTANGLED', 'SEPARATE-CONSTANT']:

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], f8, f8, c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FCOADD(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, wI, wJ, PixA_FCOADD):
            
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                wI_C = wI + 0.0j
                wJ_C = wJ + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0) and (b == 0):
                                PVAL_FKab = SCALE_C
                                
                            if (a != 0) or (b != 0):
                                PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)
            
                            
                            for ij in range(Fij):
                                ijab = ij * Fab + ab
                                PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                
                        for pq in range(Fpq):
                            PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]
                            
                        PixA_FCOADD[ROW, COL] = wI_C*PixA_FJ[ROW, COL] + wJ_C*PVAL
                        
                return PixA_FCOADD
            
            SFFTModule_dict['Construct_FCOADD'] = Construct_FCOADD

        if SCALING_MODE == 'SEPARATE-VARYING':

            # NOTE: As N0, N1 > MAX_THREADS_PER_BLOCK, here TpB == MAX_THREADS_PER_BLOCK
            _strdec = 'c16[:,:](i4[:,:], i4[:,:], c16[:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:,:,:], c16[:], c16[:,:,:], c16[:,:], f8, f8, c16[:,:])'
            @nb.njit(_strdec, parallel=True)
            def Construct_FCOADD(SREF_ijab, REF_ab, a_ijab, SPixA_FIij, ScaSPixA_FIij, Kab_Wla, Kab_Wmb, b_pq, SPixA_FTpq, PixA_FJ, wI, wJ, PixA_FCOADD):
                
                ZERO_C = 0.0 + 0.0j
                ONE_C = 1.0 + 0.0j
                SCALE_C = SCALE + 0.0j
                
                wI_C = wI + 0.0j
                wJ_C = wJ + 0.0j
                
                for ROW in nb.prange(N0):
                    for COL in nb.prange(N1):
                        
                        PVAL = ZERO_C
                        PVAL_FKab = ZERO_C
                        
                        for ab in range(Fab):
                            a, b = REF_ab[ab]
                            
                            if (a == 0 and b == 0):
                                PVAL_FKab = SCALE_C
                                
                                for ij in range(Fij):
                                    ijab = ij * Fab + ab
                                    PVAL += (a_ijab[ijab] * ScaSPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                    
                                    
                            if (a != 0 or b != 0):
                                PVAL_FKab = SCALE_C * ((Kab_Wla[w0 + a, ROW, COL] * Kab_Wmb[w1 + b, ROW, COL]) - ONE_C)
            
                                for ij in range(Fij):
                                    ijab = ij * Fab + ab
                                    PVAL += (a_ijab[ijab] * SPixA_FIij[ij, ROW, COL]) * PVAL_FKab
                                    
                        
                        for pq in range(Fpq):
                            PVAL += b_pq[pq] * SPixA_FTpq[pq, ROW, COL]
                            
                        PixA_FCOADD[ROW, COL] = wI_C*PixA_FJ[ROW, COL] + wJ_C*PVAL
                        
                return PixA_FCOADD    
                            
            SFFTModule_dict['Construct_FCOADD'] = Construct_FCOADD


        SFFTConfig = (SFFTParam_dict, SFFTModule_dict)
        if VERBOSE_LEVEL in [1, 2]:
            print('--//--//--//--//-- EXIT SFFT COMPILATION --//--//--//--//-- ')

        return SFFTConfig
    

class SingleSFFTConfigure:
    @staticmethod
    def SSC(NX, NY, KerHW=8, KerSpType='Polynomial', KerSpDegree=2, KerIntKnotX=[], KerIntKnotY=[], \
        SEPARATE_SCALING=True, ScaSpType='Polynomial', ScaSpDegree=0, ScaIntKnotX=[], ScaIntKnotY=[], \
        BkgSpType='Polynomial', BkgSpDegree=2, BkgIntKnotX=[], BkgIntKnotY=[], \
        REGULARIZE_KERNEL=False, IGNORE_LAPLACIAN_KERCENT=True, XY_REGULARIZE=None, WEIGHT_REGULARIZE=None, \
        LAMBDA_REGULARIZE=1e-6, BACKEND_4SUBTRACT='Cupy', MAX_THREADS_PER_BLOCK=8, \
        MINIMIZE_GPU_MEMORY_USAGE=False, NUM_CPU_THREADS_4SUBTRACT=8, VERBOSE_LEVEL=2):

        """
        # Compile Functions for SFFT
        #
        # Arguments:
        # -NX: Image size along X (pix)                                                        | e.g., 1024
        # -NY: Image size along Y (pix)                                                        | e.g., 1024
        # -KerHW: Kernel half width for compilation                                            | e.g., 8
        #
        # -KerSpType: Spatial varaition type of matching kernel                                | ['Polynomial', 'B-Spline']
        # -KerSpDegree: Polynomial/B-Spline degree of kernel spatial varaition                 | [0, 1, 2, 3]
        # -KerIntKnotX: Internal knots of kernel B-Spline spatial varaition along X            | e.g., [256., 512., 768.]
        # -KerIntKnotY: Internal knots of kernel B-Spline spatial varaition along Y            | e.g., [256., 512., 768.]
        #
        # -SEPARATE_SCALING: separate convolution scaling or entangled with matching kernel?   | [True, False]
        # -ScaSpType: Spatial varaition type of convolution scaling                            | ['Polynomial', 'B-Spline']
        # -ScaSpDegree: Polynomial/B-Spline degree of convolution scaling                      | [0, 1, 2, 3]
        # -ScaIntKnotX: Internal knots of scaling B-Spline spatial varaition along X           | e.g., [256., 512., 768.]
        # -ScaIntKnotY: Internal knots of scaling B-Spline spatial varaition along Y           | e.g., [256., 512., 768.]
        #
        # -BkgSpType: Spatial varaition type of differential background                        | ['Polynomial', 'B-Spline']
        # -BkgSpDegree: Polynomial/B-Spline degree of background spatial varaition             | [0, 1, 2, 3]
        # -BkgIntKnotX: Internal knots of background B-Spline spatial varaition along X        | e.g., [256., 512., 768.]
        # -BkgIntKnotY: Internal knots of background B-Spline spatial varaition along Y        | e.g., [256., 512., 768.]
        # 
        # -REGULARIZE_KERNEL: Regularize matching kernel by applying penalty on                | [True, False]
        #    kernel's second derivates using Laplacian matrix
        # -IGNORE_LAPLACIAN_KERCENT: zero out the rows of Laplacian matrix                     | [True, False]
        #    corresponding the kernel center pixels by zeros. 
        #    If True, the regularization will not impose any penalty 
        #    on a delta-function-like matching kernel
        # -XY_REGULARIZE: The coordinates at which the matching kernel regularized.            | e.g., np.array([[64., 64.], 
        #    Numpy array of (x, y) with shape (N_points, 2),                                   |                 [256., 256.]]) 
        #    where x in (0.5, NX+0.5) and y in (0.5, NY+0.5)
        # -WEIGHT_REGULARIZE: The weights of the coordinates sampled for regularization.       | e.g., np.array([1.0, 2.0, ...])
        #    Numpy array of weights with shape (N_points)
        #    -WEIGHT_REGULARIZE = None means uniform weights of 1.0
        # -LAMBDA_REGULARIZE: Tunning paramater lambda for regularization                      | e.g., 1e-6
        #    it controls the strength of penalty on kernel overfitting
        #
        # -BACKEND_4SUBTRACT: The backend with which you perform SFFT subtraction              | ['Cupy', 'Numpy']
        # -MAX_THREADS_PER_BLOCK: Maximum Threads per Block for CUDA configuration             | e.g., 8
        # -MINIMIZE_GPU_MEMORY_USAGE: Minimize the GPU Memory Usage?                           | [True, False]
        # -NUM_CPU_THREADS_4SUBTRACT: The number of CPU threads for Numpy-SFFT subtraction     | e.g., 8
        #
        # -VERBOSE_LEVEL: The level of verbosity, can be 0/1/2: QUIET/NORMAL/FULL              | [0, 1, 2]
        #
        """
        
        if BACKEND_4SUBTRACT == 'Cupy':

            SFFTConfig = SingleSFFTConfigure_Cupy.SSCC(NX=NX, NY=NY, KerHW=KerHW, KerSpType=KerSpType, \
                KerSpDegree=KerSpDegree, KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, \
                SEPARATE_SCALING=SEPARATE_SCALING, ScaSpType=ScaSpType, ScaSpDegree=ScaSpDegree, \
                ScaIntKnotX=ScaIntKnotX, ScaIntKnotY=ScaIntKnotY, BkgSpType=BkgSpType, \
                BkgSpDegree=BkgSpDegree, BkgIntKnotX=BkgIntKnotX, BkgIntKnotY=BkgIntKnotY, \
                REGULARIZE_KERNEL=REGULARIZE_KERNEL, IGNORE_LAPLACIAN_KERCENT=IGNORE_LAPLACIAN_KERCENT, \
                XY_REGULARIZE=XY_REGULARIZE, WEIGHT_REGULARIZE=WEIGHT_REGULARIZE, LAMBDA_REGULARIZE=LAMBDA_REGULARIZE, \
                MAX_THREADS_PER_BLOCK=MAX_THREADS_PER_BLOCK, MINIMIZE_GPU_MEMORY_USAGE=MINIMIZE_GPU_MEMORY_USAGE, \
                VERBOSE_LEVEL=VERBOSE_LEVEL)



        if BACKEND_4SUBTRACT == 'Numpy':

            SFFTConfig = SingleSFFTConfigure_Numpy.SSCN(NX=NX, NY=NY, KerHW=KerHW, KerSpType=KerSpType, \
                KerSpDegree=KerSpDegree, KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, \
                SEPARATE_SCALING=SEPARATE_SCALING, ScaSpType=ScaSpType, ScaSpDegree=ScaSpDegree, \
                ScaIntKnotX=ScaIntKnotX, ScaIntKnotY=ScaIntKnotY, BkgSpType=BkgSpType, \
                BkgSpDegree=BkgSpDegree, BkgIntKnotX=BkgIntKnotX, BkgIntKnotY=BkgIntKnotY, \
                REGULARIZE_KERNEL=REGULARIZE_KERNEL, IGNORE_LAPLACIAN_KERCENT=IGNORE_LAPLACIAN_KERCENT, \
                XY_REGULARIZE=XY_REGULARIZE, WEIGHT_REGULARIZE=WEIGHT_REGULARIZE, LAMBDA_REGULARIZE=LAMBDA_REGULARIZE, \
                NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT, MAX_THREADS_PER_BLOCK=MAX_THREADS_PER_BLOCK, MINIMIZE_GPU_MEMORY_USAGE=MINIMIZE_GPU_MEMORY_USAGE, \
                VERBOSE_LEVEL=VERBOSE_LEVEL)

        
        return SFFTConfig