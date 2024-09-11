import time
import warnings
import cupy as cp
import numpy as np
from astropy.io import fits
from math import ceil, floor
from astropy.wcs import WCS, FITSFixedWarning

__last_update__ = "2024-09-11"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"

class Cuda_Resampling:
    @staticmethod
    def CR(FITS_obj, FITS_targ, FITS_resamp, METHOD="LANCZOS3", FILL_ZEROPIX=True, VERBOSE_LEVEL=2):

        def Read_WCS(hdr, VERBOSE_LEVEL=2):
            with warnings.catch_warnings():
                if VERBOSE_LEVEL in [0, 1]: behavior = 'ignore'
                if VERBOSE_LEVEL in [2]: behavior = 'default'
                warnings.filterwarnings(behavior, category=FITSFixedWarning)

                if hdr['CTYPE1'] == 'RA---TAN' and 'PV1_0' in hdr:
                    _hdr = hdr.copy()
                    _hdr['CTYPE1'] = 'RA---TPV'
                    _hdr['CTYPE2'] = 'DEC--TPV'
                else: _hdr = hdr
                w = WCS(_hdr)
            return w

        PixA_obj = fits.getdata(FITS_obj, ext=0).T
        PixA_targ = fits.getdata(FITS_targ, ext=0).T

        hdr_obj = fits.getheader(FITS_obj, ext=0)
        hdr_targ = fits.getheader(FITS_targ, ext=0)

        w_obj = Read_WCS(hdr=hdr_obj, VERBOSE_LEVEL=VERBOSE_LEVEL)
        w_targ = Read_WCS(hdr=hdr_targ, VERBOSE_LEVEL=VERBOSE_LEVEL)

        # * maaping target pixel centers to the object frame
        #   todo: not fast
        NTX = int(hdr_targ["NAXIS1"]) 
        NTY = int(hdr_targ["NAXIS2"])

        _RR, _CC = np.mgrid[:NTX, :NTY]
        XX_targ, YY_targ = _RR + 1., _CC + 1.
        XY_targ = np.array([XX_targ.flatten(), YY_targ.flatten()]).T

        # * get the projected coordinates
        #   todo: this is too slow
        XY_proj = w_obj.all_world2pix(w_targ.all_pix2world(XY_targ, 1), 1)
        XX_proj = XY_proj[:, 0].reshape((NTX, NTY))
        YY_proj = XY_proj[:, 1].reshape((NTX, NTY))

        # * padding the object frame
        if METHOD == 'BILINEAR':
            KERHW = (1, 1)

        if METHOD == 'LANCZOS3':
            KERHW = (3, 3)
            
        NOX = int(hdr_obj["NAXIS1"]) 
        NOY = int(hdr_obj["NAXIS2"])

        # find the root index and shift with maximal kernel halfwidth (KERHW)
        # Note: the index ranges (RMIN --- RMAX) and (CMIN --- CMAX) can cover
        #       all pixels that interpolation may be use.

        RMIN = (floor(XX_proj.min()) - 1) - KERHW[0]
        RMAX = (floor(XX_proj.max()) - 1) + KERHW[0]
        RPAD = (-np.min([RMIN, 0]), np.max([RMAX - (NOX - 1), 0]))

        CMIN = (floor(YY_proj.min()) - 1) - KERHW[1]
        CMAX = (floor(YY_proj.max()) - 1) + KERHW[1]
        CPAD = (-np.min([CMIN, 0]), np.max([CMAX - (NOY - 1), 0]))
        PAD_WIDTH = (RPAD, CPAD)

        PixA_Eobj = np.pad(PixA_obj, PAD_WIDTH, mode='constant', constant_values=0.)
        NEOX, NEOY = PixA_Eobj.shape

        # * get the projected coordinates on extended object frame
        XX_Eproj = XX_proj.copy()
        XX_Eproj[:, :] += PAD_WIDTH[0][0]

        YY_Eproj = YY_proj.copy()
        YY_Eproj[:, :] += PAD_WIDTH[1][0]

        RMIN_E = (floor(XX_Eproj.min()) - 1) - KERHW[0]
        RMAX_E = (floor(XX_Eproj.max()) - 1) + KERHW[0]

        CMIN_E = (floor(YY_Eproj.min()) - 1) - KERHW[1]
        CMAX_E = (floor(YY_Eproj.max()) - 1) + KERHW[1]

        assert RMIN_E >= 0 and CMIN_E >= 0 
        assert RMAX_E < NEOX and CMAX_E < NEOY

        # * Cupy configuration
        MaxThreadPerB = 8
        GPUManage = lambda NT: ((NT-1)//MaxThreadPerB + 1, min(NT, MaxThreadPerB))
        BpG_PIX0, TpB_PIX0 = GPUManage(NTX)
        BpG_PIX1, TpB_PIX1 = GPUManage(NTY)
        BpG_PIX, TpB_PIX = (BpG_PIX0, BpG_PIX1), (TpB_PIX0, TpB_PIX1, 1)

        if not XX_Eproj.flags['C_CONTIGUOUS']:
            XX_Eproj = np.ascontiguousarray(XX_Eproj, np.float64)
            XX_Eproj_GPU = cp.array(XX_Eproj)
        else: XX_Eproj_GPU = cp.array(XX_Eproj.astype(np.float64))

        if not YY_Eproj.flags['C_CONTIGUOUS']:
            YY_Eproj = np.ascontiguousarray(YY_Eproj, np.float64)
            YY_Eproj_GPU = cp.array(YY_Eproj)
        else: YY_Eproj_GPU = cp.array(YY_Eproj.astype(np.float64))

        if not PixA_Eobj.flags['C_CONTIGUOUS']:
            PixA_Eobj = np.ascontiguousarray(PixA_Eobj, np.float64)
            PixA_Eobj_GPU = cp.array(PixA_Eobj)
        else: PixA_Eobj_GPU = cp.array(PixA_Eobj.astype(np.float64))

        PixA_resamp_GPU = cp.zeros((NTX, NTY), dtype=np.float64)

        if METHOD == "BILINEAR":
        
            # * perform bilinear resampling using CUDA
            # input: PixA_Eobj | (NEOX, NEOY)
            # input: XX_Eproj, YY_Eproj | (NTX, NTY)
            # output: PixA_resamp | (NTX, NTY)
            
            _refdict = {'NTX': NTX, 'NTY': NTY, 'NEOX': NEOX, 'NEOY': NEOY}
            _funcstr = r"""
            extern "C" __global__ void kmain(double XX_Eproj_GPU[%(NTX)s][%(NTY)s], double YY_Eproj_GPU[%(NTX)s][%(NTY)s], 
                double PixA_Eobj_GPU[%(NEOX)s][%(NEOY)s], double PixA_resamp_GPU[%(NTX)s][%(NTY)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int NTX = %(NTX)s;
                int NTY = %(NTY)s;
                int NEOX = %(NEOX)s;
                int NEOY = %(NEOY)s;

                if (ROW < NTX && COL < NTY) {
                
                    double x = XX_Eproj_GPU[ROW][COL];
                    double y = YY_Eproj_GPU[ROW][COL];

                    int r1 = floor(x) - 1;
                    int c1 = floor(y) - 1;
                    int r2 = r1 + 1;
                    int c2 = c1 + 1;

                    double dx = x - floor(x); 
                    double dy = y - floor(y);

                    double w11 = (1-dx) * (1-dy);
                    double w12 = (1-dx) * dy;
                    double w21 = dx * (1-dy);
                    double w22 = dx * dy;

                    PixA_resamp_GPU[ROW][COL] = w11 * PixA_Eobj_GPU[r1][c1] + w12 * PixA_Eobj_GPU[r1][c2] + 
                        w21 * PixA_Eobj_GPU[r2][c1] + w22 * PixA_Eobj_GPU[r2][c2];
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            resamp_func = _module.get_function('kmain')
            
            t0 = time.time()
            resamp_func(args=(XX_Eproj_GPU, YY_Eproj_GPU, PixA_Eobj_GPU, PixA_resamp_GPU), block=TpB_PIX, grid=BpG_PIX)
            if VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: Cuda resampling takes [%.6f s]' %(time.time() - t0))
            
        if METHOD == "LANCZOS3":
            
            # * perform LANCZOS-3 resampling using CUDA
            # input: XX_Eproj, YY_Eproj | (NTX, NTY)
            # output: PixA_resamp | (NTX, NTY)
            
            _refdict = {'NTX': NTX, 'NTY': NTY}
            _funcstr = r"""
            extern "C" __global__ void kmain(double XX_Eproj_GPU[%(NTX)s][%(NTY)s], double YY_Eproj_GPU[%(NTX)s][%(NTY)s], 
                double LKERNEL_X_GPU[6][%(NTX)s][%(NTY)s], double LKERNEL_Y_GPU[6][%(NTX)s][%(NTY)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int NTX = %(NTX)s;
                int NTY = %(NTY)s;
                
                double PI = 3.141592653589793;
                double PIS = 9.869604401089358;
                
                if (ROW < NTX && COL < NTY) {
                    
                    double x = XX_Eproj_GPU[ROW][COL];
                    double y = YY_Eproj_GPU[ROW][COL];
                    
                    double dx = x - floor(x); 
                    double dy = y - floor(y);
                    
                    // LANCZOS3 weights in x axis
                    double wx0 = 3.0 * sin(PI*(-2.0 - dx)) * sin(PI*(-2.0 - dx)/3.0) / (PIS*(-2.0 - dx) * (-2.0 - dx));
                    double wx1 = 3.0 * sin(PI*(-1.0 - dx)) * sin(PI*(-1.0 - dx)/3.0) / (PIS*(-1.0 - dx) * (-1.0 - dx));
                    double wx2 = 1.0;
                    if (fabs(dx) > 1e-4) {
                        wx2 = 3.0 * sin(PI*(-dx)) * sin(PI*(-dx)/3.0) / (PIS*(-dx) * (-dx));
                    }
                    double wx3 = 3.0 * sin(PI*(1.0 - dx)) * sin(PI*(1.0 - dx)/3.0) / (PIS*(1.0 - dx) * (1.0 - dx));
                    double wx4 = 3.0 * sin(PI*(2.0 - dx)) * sin(PI*(2.0 - dx)/3.0) / (PIS*(2.0 - dx) * (2.0 - dx));
                    double wx5 = 3.0 * sin(PI*(3.0 - dx)) * sin(PI*(3.0 - dx)/3.0) / (PIS*(3.0 - dx) * (3.0 - dx));
                    
                    LKERNEL_X_GPU[0][ROW][COL] = wx0;
                    LKERNEL_X_GPU[1][ROW][COL] = wx1;
                    LKERNEL_X_GPU[2][ROW][COL] = wx2;
                    LKERNEL_X_GPU[3][ROW][COL] = wx3;
                    LKERNEL_X_GPU[4][ROW][COL] = wx4;
                    LKERNEL_X_GPU[5][ROW][COL] = wx5;
                    
                    // LANCZOS3 weights in y axis
                    double wy0 = 3.0 * sin(PI*(-2.0 - dy)) * sin(PI*(-2.0 - dy)/3.0) / (PIS*(-2.0 - dy) * (-2.0 - dy));
                    double wy1 = 3.0 * sin(PI*(-1.0 - dy)) * sin(PI*(-1.0 - dy)/3.0) / (PIS*(-1.0 - dy) * (-1.0 - dy));
                    double wy2 = 1.0;
                    if (fabs(dy) > 1e-4) {
                        wy2 = 3.0 * sin(PI*(-dy)) * sin(PI*(-dy)/3.0) / (PIS*(-dy) * (-dy));
                    }
                    double wy3 = 3.0 * sin(PI*(1.0 - dy)) * sin(PI*(1.0 - dy)/3.0) / (PIS*(1.0 - dy) * (1.0 - dy));
                    double wy4 = 3.0 * sin(PI*(2.0 - dy)) * sin(PI*(2.0 - dy)/3.0) / (PIS*(2.0 - dy) * (2.0 - dy));
                    double wy5 = 3.0 * sin(PI*(3.0 - dy)) * sin(PI*(3.0 - dy)/3.0) / (PIS*(3.0 - dy) * (3.0 - dy));
                    
                    LKERNEL_Y_GPU[0][ROW][COL] = wy0;
                    LKERNEL_Y_GPU[1][ROW][COL] = wy1;
                    LKERNEL_Y_GPU[2][ROW][COL] = wy2;
                    LKERNEL_Y_GPU[3][ROW][COL] = wy3;
                    LKERNEL_Y_GPU[4][ROW][COL] = wy4;
                    LKERNEL_Y_GPU[5][ROW][COL] = wy5;
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            weightkernel_func = _module.get_function('kmain')
            
            _refdict = {'NTX': NTX, 'NTY': NTY, 'NEOX': NEOX, 'NEOY': NEOY}
            _funcstr = r"""
            extern "C" __global__ void kmain(double XX_Eproj_GPU[%(NTX)s][%(NTY)s], double YY_Eproj_GPU[%(NTX)s][%(NTY)s], 
                double LKERNEL_X_GPU[6][%(NTX)s][%(NTY)s], double LKERNEL_Y_GPU[6][%(NTX)s][%(NTY)s], 
                double PixA_Eobj_GPU[%(NEOX)s][%(NEOY)s], double PixA_resamp_GPU[%(NTX)s][%(NTY)s])
            {
                int ROW = blockIdx.x*blockDim.x+threadIdx.x;
                int COL = blockIdx.y*blockDim.y+threadIdx.y;

                int NTX = %(NTX)s;
                int NTY = %(NTY)s;
                int NEOX = %(NEOX)s;
                int NEOY = %(NEOY)s;

                if (ROW < NTX && COL < NTY) {
                
                    double x = XX_Eproj_GPU[ROW][COL];
                    double y = YY_Eproj_GPU[ROW][COL];
                    
                    int r0 = floor(x) - 3;
                    int c0 = floor(y) - 3;
                    
                    for(int i = 0; i < 6; ++i){
                        for(int j = 0; j < 6; ++j){
                            double w = LKERNEL_X_GPU[i][ROW][COL] * LKERNEL_Y_GPU[j][ROW][COL];
                            PixA_resamp_GPU[ROW][COL] += w * PixA_Eobj_GPU[r0 + i][c0 + j];
                        }
                    }
                }
            }
            """
            _code = _funcstr % _refdict
            _module = cp.RawModule(code=_code, backend=u'nvcc', translate_cucomplex=False)
            resamp_func = _module.get_function('kmain')
            
            t0 = time.time()
            LKERNEL_X_GPU = cp.zeros((6, NTX, NTY), dtype=np.float64)
            LKERNEL_Y_GPU = cp.zeros((6, NTX, NTY), dtype=np.float64)
            weightkernel_func(args=(XX_Eproj_GPU, YY_Eproj_GPU, LKERNEL_X_GPU, LKERNEL_Y_GPU), block=TpB_PIX, grid=BpG_PIX)
            resamp_func(args=(XX_Eproj_GPU, YY_Eproj_GPU, LKERNEL_X_GPU, LKERNEL_Y_GPU, PixA_Eobj_GPU, PixA_resamp_GPU), block=TpB_PIX, grid=BpG_PIX)
            if VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: Cuda resampling takes [%.6f s]' %(time.time() - t0))

        # save the resampled image
        PixA_resamp = cp.asnumpy(PixA_resamp_GPU)

        # todo: refine the pixel filling
        PixA_resamp[PixA_resamp == 0.] = np.nan
        with fits.open(FITS_targ) as hdl:
            hdl[0].data[:, :] = PixA_resamp.T
            hdl.writeto(FITS_resamp, overwrite=True)
        if VERBOSE_LEVEL in [1, 2]:
            print('MeLOn CheckPoint: resampled fits file saved at \n # [%s]' %(FITS_resamp))

        return PixA_resamp
