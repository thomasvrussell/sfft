import os
import sys
import numpy as np

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class SingleSFFTConfigure_Pycuda:
    @staticmethod
    def SSCP(NX, NY, KerHW, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, CUDA_DEVICE='0'):

        import pycuda.autoinit
        from pycuda.compiler import SourceModule
        os.environ["CUDA_DEVICE"] = CUDA_DEVICE

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
        Fij = int((DK+1)*(DK+2)/2)        # dof for matching-kernel polynomial index ij 
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

        # * Load SFFT CUDA modules
        #   NOTE: Generally, a kernel function is defined without knowledge about Grid-Block-Thread Management.
        #         However, we need to know the size of threads per block if SharedMemory is called.

        # ************************************ SpatialCoor.cu ************************************ #

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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['SpatialCoor'] = _module

        # ************************************ SpatialPoly.cu ************************************ #

        # ** produce Tij, Iij, Tpq
        _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq}
        _funcstr = r"""
        extern "C" __global__ void kmain(int REF_ij_GPU[%(Fij)s][2], int REF_pq_GPU[%(Fpq)s][2],
            double PixA_CX_GPU[%(N0)s][%(N1)s], double PixA_CY_GPU[%(N0)s][%(N1)s], double PixA_I_GPU[%(N0)s][%(N1)s], 
            double SPixA_Tij_GPU[%(Fij)s][%(N0)s][%(N1)s], double SPixA_Iij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
            double SPixA_Tpq_GPU[%(Fpq)s][%(N0)s][%(N1)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
        
            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fij = %(Fij)s;
            int Fpq = %(Fpq)s;
            
            if (ROW < N0 && COL < N1) {
                for(int ij = 0; ij < Fij; ++ij) {
                    int i = REF_ij_GPU[ij][0];
                    int j = REF_ij_GPU[ij][1];
                    double poly_kterm = pow(PixA_CX_GPU[ROW][COL], i) * pow(PixA_CY_GPU[ROW][COL], j);
                    SPixA_Tij_GPU[ij][ROW][COL] = poly_kterm;
                    SPixA_Iij_GPU[ij][ROW][COL] = PixA_I_GPU[ROW][COL] * poly_kterm;
                }
                    
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['SpatialPoly'] = _module

        # ************************************ HadProd_OMG.cu & FillLS_OMG.cu ************************************ #

        # ** Hadamard Product [𝛀]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['HadProd_OMG'] = _module

        # ** Fill Linear-System [𝛀]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['FillLS_OMG'] = _module

        # ************************************ HadProd_GAM.cu & FillLS_GAM.cu ************************************ #

        # ** Hadamard Product [𝜦]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['HadProd_GAM'] = _module

        # ** Fill Linear-System [𝜦]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['FillLS_GAM'] = _module

        # ************************************ HadProd_PSI.cu & FillLS_PSI.cu ************************************ #

        # ** Hadamard Product [𝜳]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['HadProd_PSI'] = _module

        # ** Fill Linear-System [𝜳]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['FillLS_PSI'] = _module

        # ************************************ HadProd_PHI.cu & FillLS_PHI.cu ************************************ #

        # ** Hadamard Product [𝚽]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['HadProd_PHI'] = _module

        # ** Fill Linear-System [𝚽]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['FillLS_PHI'] = _module

        # ************************************ HadProd_THE.cu & FillLS_THE.cu ************************************ #

        # ** Hadamard Product [𝚯]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['HadProd_THE'] = _module

        # ** Fill Linear-System [𝚯]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['FillLS_THE'] = _module

        # ************************************ HadProd_DEL.cu & FillLS_DEL.cu ************************************ #

        # ** Hadamard Product [𝚫]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['HadProd_DEL'] = _module

        # ** Fill Linear-System [𝚫]
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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['FillLS_DEL'] = _module

        # ************************************ Remove_LSFStripes.cu ************************************ #

        # ** Remove Forbidden Stripes in Linear-System
        _refdict = {'NEQ': NEQ, 'NEQ_FSfree': NEQ_FSfree}
        _funcstr = r"""
        extern "C" __global__ void kmain(double LHMAT_GPU[%(NEQ)s][%(NEQ)s], int IDX_nFS_GPU[%(NEQ_FSfree)s], 
            double LHMAT_FSfree_GPU[%(NEQ_FSfree)s][%(NEQ_FSfree)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            int NEQ_FSfree = %(NEQ_FSfree)s;

            if (ROW < NEQ_FSfree && COL < NEQ_FSfree) {
                int ROW_nFS = IDX_nFS_GPU[ROW];
                int COL_nFS = IDX_nFS_GPU[COL];
                LHMAT_FSfree_GPU[ROW][COL] = LHMAT_GPU[ROW_nFS][COL_nFS];
            }
        }
        """
        _code = _funcstr % _refdict
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['Remove_LSFStripes'] = _module

        # ************************************ Extend_Solution.cu ************************************ #

        # ** Extend Solution from Forbidden-Stripes-Free Linear-System
        _refdict = {'NEQ_FSfree': NEQ_FSfree, 'NEQ': NEQ}
        _funcstr = r"""
        extern "C" __global__ void kmain(double Solution_FSfree_GPU[%(NEQ_FSfree)s], 
            int IDX_nFS_GPU[%(NEQ_FSfree)s], double Solution_GPU[%(NEQ)s]) 
        {    
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int NEQ_FSfree = %(NEQ_FSfree)s;
            
            if (ROW < NEQ_FSfree) {
                int ROW_nFS = IDX_nFS_GPU[ROW];
                Solution_GPU[ROW_nFS] = Solution_FSfree_GPU[ROW];
            }
        }
        """
        _code = _funcstr % _refdict
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['Extend_Solution'] = _module

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
        _module = SourceModule(source=_code, nvcc='nvcc')
        SFFTModule_dict['Construct_FDIFF'] = _module
        
        SFFTConfig = (SFFTParam_dict, SFFTModule_dict)
        print('\n  --//--//--//--//-- EXIT SFFT COMPILATION --//--//--//--//-- ')

        return SFFTConfig
    
class SingleSFFTConfigure_Cupy:
    @staticmethod
    def SSCC(NX, NY, KerHW, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, CUDA_DEVICE='0'):
        
        import cupy as cp
        os.environ["CUDA_DEVICE"] = CUDA_DEVICE

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
        Fij = int((DK+1)*(DK+2)/2)        # dof for matching-kernel polynomial index ij 
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

        # * Load SFFT CUDA modules
        #   NOTE: Generally, a kernel function is defined without knowledge about Grid-Block-Thread Management.
        #               However, we need to know the size of threads per block if SharedMemory is called.

        # ************************************ SpatialCoor.cu ************************************ #

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
        SFFTModule_dict['SpatialCoor'] = _module

        # ************************************ SpatialPoly.cu ************************************ #

        # ** produce Tij, Iij, Tpq
        _refdict = {'N0': N0, 'N1': N1, 'Fij': Fij, 'Fpq': Fpq}
        _funcstr = r"""
        extern "C" __global__ void kmain(int REF_ij_GPU[%(Fij)s][2], int REF_pq_GPU[%(Fpq)s][2],
            double PixA_CX_GPU[%(N0)s][%(N1)s], double PixA_CY_GPU[%(N0)s][%(N1)s], double PixA_I_GPU[%(N0)s][%(N1)s], 
            double SPixA_Tij_GPU[%(Fij)s][%(N0)s][%(N1)s], double SPixA_Iij_GPU[%(Fij)s][%(N0)s][%(N1)s], 
            double SPixA_Tpq_GPU[%(Fpq)s][%(N0)s][%(N1)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
        
            int N0 = %(N0)s;
            int N1 = %(N1)s;
            int Fij = %(Fij)s;
            int Fpq = %(Fpq)s;
            
            if (ROW < N0 && COL < N1) {
                for(int ij = 0; ij < Fij; ++ij) {
                    int i = REF_ij_GPU[ij][0];
                    int j = REF_ij_GPU[ij][1];
                    double poly_kterm = pow(PixA_CX_GPU[ROW][COL], i) * pow(PixA_CY_GPU[ROW][COL], j);
                    SPixA_Tij_GPU[ij][ROW][COL] = poly_kterm;
                    SPixA_Iij_GPU[ij][ROW][COL] = PixA_I_GPU[ROW][COL] * poly_kterm;
                }
                    
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
        SFFTModule_dict['SpatialPoly'] = _module

        # ************************************ HadProd_OMG.cu & FillLS_OMG.cu ************************************ #

        # ** Hadamard Product [𝛀]
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

        # ** Fill Linear-System [𝛀]
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

        # ** Hadamard Product [𝜦]
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

        # ** Fill Linear-System [𝜦]
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

        # ** Hadamard Product [𝜳]
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

        # ** Fill Linear-System [𝜳]
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

        # ** Hadamard Product [𝚽]
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

        # ** Fill Linear-System [𝚽]
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

        # ** Hadamard Product [𝚯]
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

        # ** Fill Linear-System [𝚯]
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

        # ** Hadamard Product [𝚫]
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

        # ** Fill Linear-System [𝚫]
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

        # ************************************ Remove_LSFStripes.cu ************************************ #

        # ** Remove Forbidden Stripes in Linear-System
        _refdict = {'NEQ': NEQ, 'NEQ_FSfree': NEQ_FSfree}
        _funcstr = r"""
        extern "C" __global__ void kmain(double LHMAT_GPU[%(NEQ)s][%(NEQ)s], int IDX_nFS_GPU[%(NEQ_FSfree)s], 
            double LHMAT_FSfree_GPU[%(NEQ_FSfree)s][%(NEQ_FSfree)s]) 
        {
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int COL = blockIdx.y*blockDim.y+threadIdx.y;
            int NEQ_FSfree = %(NEQ_FSfree)s;

            if (ROW < NEQ_FSfree && COL < NEQ_FSfree) {
                int ROW_nFS = IDX_nFS_GPU[ROW];
                int COL_nFS = IDX_nFS_GPU[COL];
                LHMAT_FSfree_GPU[ROW][COL] = LHMAT_GPU[ROW_nFS][COL_nFS];
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['Remove_LSFStripes'] = _module

        # ************************************ Extend_Solution.cu ************************************ #

        # ** Extend Solution from Forbidden-Stripes-Free Linear-System
        _refdict = {'NEQ_FSfree': NEQ_FSfree, 'NEQ': NEQ}
        _funcstr = r"""
        extern "C" __global__ void kmain(double Solution_FSfree_GPU[%(NEQ_FSfree)s], 
            int IDX_nFS_GPU[%(NEQ_FSfree)s], double Solution_GPU[%(NEQ)s]) 
        {    
            int ROW = blockIdx.x*blockDim.x+threadIdx.x;
            int NEQ_FSfree = %(NEQ_FSfree)s;
            
            if (ROW < NEQ_FSfree) {
                int ROW_nFS = IDX_nFS_GPU[ROW];
                Solution_GPU[ROW_nFS] = Solution_FSfree_GPU[ROW];
            }
        }
        """
        _code = _funcstr % _refdict
        _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
        SFFTModule_dict['Extend_Solution'] = _module

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
        print('\n  --//--//--//--//-- EXIT SFFT COMPILATION --//--//--//--//-- ')

        return SFFTConfig

class SingleSFFTConfigure_Numpy:
    @staticmethod
    def SSCN(NX, NY, KerHW, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, NUM_CPU_THREADS=8):

        import numba as nb
        nb.set_num_threads = NUM_CPU_THREADS

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
        Fij = int((DK+1)*(DK+2)/2)        # dof for matching-kernel polynomial index ij 
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

            for ij in nb.prange(Fij):
                i, j = REF_ij[ij]
                PixA_kpoly = np.power(PixA_CX, i) * np.power(PixA_CY, j)
                SPixA_Tij[ij, :, :] = PixA_kpoly   # NOTE UPDATE
                SPixA_Iij[ij, :, :] = PixA_I * PixA_kpoly   # NOTE UPDATE
            
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
        backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8):

        """
        # Arguments:
        # a) NX, NY: Image Size (pixels) along X / Y axis.
        # b) KerHW: Kernel half width for compilation.
        # c) KerPolyOrder: The order of Polynomial Variation for PSF-Matching Kernel.
        # d) BGPolyOrder: The order of Polynomial Variation for Differential Background.
        # e) ConstPhotRatio: Use a constant photometric ratio in image subtraction ?
        # f) backend: Which backend would you like to perform SFFT on ?
        # g) CUDA_DEVICE (backend = Pycuda / Cupy): Which GPU device would you want to perform SFFT on ?
        # h) NUM_CPU_THREADS (backend = Numpy): How many CPU threads would you want to perform SFFT on ?
        
        """

        if backend == 'Pycuda':
            SFFTConfig = SingleSFFTConfigure_Pycuda.SSCP(NX=NX, NY=NY, KerHW=KerHW, \
                KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
                CUDA_DEVICE=CUDA_DEVICE)
        
        if backend == 'Cupy':
            SFFTConfig = SingleSFFTConfigure_Cupy.SSCC(NX=NX, NY=NY, KerHW=KerHW, \
                KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
                CUDA_DEVICE=CUDA_DEVICE)
        
        if backend == 'Numpy':
            SFFTConfig = SingleSFFTConfigure_Numpy.SSCN(NX=NX, NY=NY, KerHW=KerHW, \
                KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
                NUM_CPU_THREADS=NUM_CPU_THREADS)
        
        return SFFTConfig

class BatchSFFTConfigure:
    @staticmethod
    def BSCP(NX, NY, KerHW_CLst, KerPolyOrder=2, BGPolyOrder=2, ConstPhotRatio=True, \
        backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8):
    
        SFFTConfig_Dict = {}
        for KerHW in KerHW_CLst:            
            SFFTConfig = SingleSFFTConfigure.SSC(NX=NX, NY=NY, KerHW=KerHW, \
                KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
                backend=backend, CUDA_DEVICE=CUDA_DEVICE)
            SFFTConfig_Dict[KerHW] = SFFTConfig

        return SFFTConfig_Dict
