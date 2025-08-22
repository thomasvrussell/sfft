import time
import math
import cupy as cp
import numpy as np
from sfft.utils.ReadWCS import Read_WCS

__last_update__ = "2024-09-22"
__author__ = "Lei Hu <leihu@andrew.cmu.edu> & Shu Liu <shl159@pitt.edu>"

class Cupy_WCS_Transform:
    def __init__(self):
        __author__ = "Shu Liu <shl159@pitt.edu> & Lei Hu <leihu@andrew.cmu.edu>"
        return None

    def read_cd_wcs(self, hdr_wcs):
        """ Read WCS information from FITS header """
        assert hdr_wcs["CTYPE1"] == "RA---TAN"
        assert hdr_wcs["CTYPE2"] == "DEC--TAN"

        N0 = int(hdr_wcs["NAXIS1"])
        N1 = int(hdr_wcs["NAXIS2"])

        CRPIX1 = float(hdr_wcs["CRPIX1"])
        CRPIX2 = float(hdr_wcs["CRPIX2"])

        CRVAL1 = float(hdr_wcs["CRVAL1"])
        CRVAL2 = float(hdr_wcs["CRVAL2"])

        LONPOLE = float(hdr_wcs["LONPOLE"])

        KEYDICT = {
            "N0": N0, "N1": N1, 
            "CRPIX1": CRPIX1, "CRPIX2": CRPIX2,
            "CRVAL1": CRVAL1, "CRVAL2": CRVAL2,
            "LONPOLE": LONPOLE
        }

        CDKEY = Read_WCS.determine_cdkey( hdr_wcs )
        CD1_1 = hdr_wcs[f"{CDKEY}1_1"]
        CD1_2 = hdr_wcs[f"{CDKEY}1_2"] if f"{CDKEY}1_2" in hdr_wcs else 0.
        CD2_1 = hdr_wcs[f"{CDKEY}2_1"] if f"{CDKEY}2_1" in hdr_wcs else 0.
        CD2_2 = hdr_wcs[f"{CDKEY}2_2"]

        CD_GPU = cp.array([
            [CD1_1, CD1_2], 
            [CD2_1, CD2_2]
        ], dtype=cp.float64)
        
        return KEYDICT, CD_GPU

    def read_sip_wcs(self, hdr_wcs):
        """ Read TAN-SIP WCS information from FITS header """
        assert hdr_wcs["CTYPE1"] == "RA---TAN-SIP"
        assert hdr_wcs["CTYPE2"] == "DEC--TAN-SIP"

        N0 = int(hdr_wcs["NAXIS1"])
        N1 = int(hdr_wcs["NAXIS2"])

        CRPIX1 = float(hdr_wcs["CRPIX1"])
        CRPIX2 = float(hdr_wcs["CRPIX2"])

        CRVAL1 = float(hdr_wcs["CRVAL1"])
        CRVAL2 = float(hdr_wcs["CRVAL2"])
        
        LONPOLE = float(hdr_wcs["LONPOLE"])

        A_ORDER = int(hdr_wcs["A_ORDER"])
        B_ORDER = int(hdr_wcs["B_ORDER"])

        KEYDICT = {
            "N0": N0, "N1": N1, 
            "CRPIX1": CRPIX1, "CRPIX2": CRPIX2,
            "CRVAL1": CRVAL1, "CRVAL2": CRVAL2, 
            "LONPOLE": LONPOLE,
            "A_ORDER": A_ORDER, "B_ORDER": B_ORDER
        }

        CDKEY = Read_WCS.determine_cdkey( hdr_wcs )
        CD1_1 = hdr_wcs[f"{CDKEY}1_1"]
        CD1_2 = hdr_wcs[f"{CDKEY}1_2"] if f"{CDKEY}1_2" in hdr_wcs else 0.
        CD2_1 = hdr_wcs[f"{CDKEY}2_1"] if f"{CDKEY}2_1" in hdr_wcs else 0.
        CD2_2 = hdr_wcs[f"{CDKEY}2_2"]

        CD_GPU = cp.array([
            [CD1_1, CD1_2], 
            [CD2_1, CD2_2]
        ], dtype=cp.float64)

        A_SIP_GPU = cp.zeros((A_ORDER+1, A_ORDER+1), dtype=cp.float64)
        B_SIP_GPU = cp.zeros((B_ORDER+1, B_ORDER+1), dtype=cp.float64)

        for p in range(A_ORDER + 1):
            for q in range(0, A_ORDER - p + 1):
                keyword = f"A_{p}_{q}"
                if keyword in hdr_wcs:
                    A_SIP_GPU[p, q] = hdr_wcs[keyword]
        
        for p in range(B_ORDER + 1):
            for q in range(0, B_ORDER - p + 1):
                keyword = f"B_{p}_{q}"
                if keyword in hdr_wcs:
                    B_SIP_GPU[p, q] = hdr_wcs[keyword]
        
        return KEYDICT, CD_GPU, A_SIP_GPU, B_SIP_GPU

    def cd_transform(self, IMAGE_X_GPU, IMAGE_Y_GPU, CD_GPU):
        """CD matrix transformation from image to world"""
        WORLD_X_GPU, WORLD_Y_GPU = cp.matmul(CD_GPU, cp.array([IMAGE_X_GPU, IMAGE_Y_GPU]))
        return WORLD_X_GPU, WORLD_Y_GPU
    
    def cd_transform_inv(self, WORLD_X_GPU, WORLD_Y_GPU, CD_GPU):
        """CD matrix transformation from world to image"""
        # CD_inv_GPU = cp.linalg.inv(CD_GPU)
        CD_inv_GPU = 1. / (CD_GPU[0, 0] * CD_GPU[1, 1] - CD_GPU[0, 1] * CD_GPU[1, 0]) * \
            cp.array([[CD_GPU[1, 1], -CD_GPU[0, 1]], [-CD_GPU[1, 0], CD_GPU[0, 0]]])
        IMAGE_X_GPU, IMAGE_Y_GPU = cp.matmul(CD_inv_GPU, cp.array([WORLD_X_GPU, WORLD_Y_GPU]))
        return IMAGE_X_GPU, IMAGE_Y_GPU
    
    def sip_forward_transform(self, u_GPU, v_GPU, A_SIP_GPU, B_SIP_GPU):
        """Forward SIP transformation from (u, v) to (U, V)"""
        A_ORDER = A_SIP_GPU.shape[0] - 1
        U_GPU = u_GPU + cp.zeros_like(u_GPU)
        for p in range(A_ORDER + 1):
            for q in range(A_ORDER - p + 1):
                U_GPU += A_SIP_GPU[p, q] * u_GPU**p * v_GPU**q
            
        B_ORDER = B_SIP_GPU.shape[0] - 1
        V_GPU = v_GPU + cp.zeros_like(v_GPU)
        for p in range(B_ORDER + 1):
            for q in range(B_ORDER - p + 1):
                V_GPU += B_SIP_GPU[p, q] * u_GPU**p * v_GPU**q
        return U_GPU, V_GPU

    def sip_backward_matrix(self, U_GPU, V_GPU, ORDER):
        """Compute the backward matrix P_UV (FP_UV or GP_UV)"""
        NSAMP = len(U_GPU)
        P_UV_GPU = cp.zeros((NSAMP, (ORDER+1)*(ORDER+2)//2), dtype=cp.float64)
        idx = 0
        for p in range(ORDER + 1):
            for q in range(ORDER - p + 1):
                P_UV_GPU[:, idx] = U_GPU**p * V_GPU**q
                idx += 1
        return P_UV_GPU

    def lstsq_sip_backward_transform(self, u_GPU, v_GPU, U_GPU, V_GPU, A_ORDER, B_ORDER):
        """Compute the backward transformation from (U, V) to (u, v)"""
        FP_UV_GPU = self.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=A_ORDER)
        GP_UV_GPU = self.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=B_ORDER)

        AP_lstsq_GPU = cp.linalg.lstsq(FP_UV_GPU, (u_GPU - U_GPU).reshape(-1, 1), rcond=None)[0]
        BP_lstsq_GPU = cp.linalg.lstsq(GP_UV_GPU, (v_GPU - V_GPU).reshape(-1, 1), rcond=None)[0]
        return FP_UV_GPU, GP_UV_GPU, AP_lstsq_GPU, BP_lstsq_GPU

    def sip_backward_transform(self, U_GPU, V_GPU, FP_UV_GPU, GP_UV_GPU, AP_GPU, BP_GPU):
        """Backward transformation from (U, V) to (u, v)"""
        u_GPU = U_GPU + cp.matmul(FP_UV_GPU, AP_GPU)[:, 0]
        v_GPU = V_GPU + cp.matmul(GP_UV_GPU, BP_GPU)[:, 0]
        return u_GPU, v_GPU

    def random_coord_sampling(self, N0, N1, CRPIX1, CRPIX2, NSAMP=1024, RANDOM_SEED=10086):
        """Random sampling of coordinates"""
        if RANDOM_SEED is not None:
            cp.random.seed(RANDOM_SEED)
        u_GPU = cp.random.uniform(0.5, N0+0.5, NSAMP, dtype=cp.float64) - CRPIX1
        v_GPU = cp.random.uniform(0.5, N1+0.5, NSAMP, dtype=cp.float64) - CRPIX2
        return u_GPU, v_GPU

class Cupy_Resampling:
    def __init__(self, RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=2):
        __author__ = "Lei Hu <leihu@andrew.cmu.edu>"
        self.RESAMP_METHOD = RESAMP_METHOD
        self.VERBOSE_LEVEL = VERBOSE_LEVEL
        return None
    
    def spherical_transformation(self, x_GPU, y_GPU, alpha_p, delta_p, phi_p):
        """Spherical transformation (Spherical projection & Spherical coordinate rotation) using Cupy"""
        # NOTE: we assumpe all variables are in radians!
        
        # * Spherical projection (x, y) > (phi, theta)
        phi_GPU = cp.arctan2(x_GPU, -y_GPU)
        theta_GPU = cp.arctan(1./cp.sqrt(x_GPU**2 + y_GPU**2))

        # * Spherical coordinate rotation (phi, theta) > (alpha, delta)
        delta_GPU = cp.arcsin(
            cp.sin(theta_GPU) * cp.sin(delta_p) + \
            cp.cos(theta_GPU) * cp.cos(delta_p) * cp.cos((phi_GPU - phi_p))
        )
        
        COS_TERM_GPU = (cp.sin(theta_GPU) * cp.cos(delta_p) - \
            cp.cos(theta_GPU) * cp.sin(delta_p) * cp.cos(phi_GPU - phi_p)) / cp.cos(delta_GPU)
        SIN_TERM_GPU = - cp.cos(theta_GPU) * cp.sin(phi_GPU - phi_p) / cp.cos(delta_GPU)
        alpha_GPU = alpha_p + cp.arctan2(SIN_TERM_GPU, COS_TERM_GPU)

        return alpha_GPU, delta_GPU

    def inverse_spherical_transformation(self, alpha_GPU, delta_GPU, alpha_p, delta_p, phi_p):
        """Inverse Spherical transformation (Spherical projection & Spherical coordinate rotation) using Cupy"""
        # NOTE: we assumpe all variables are in radians!
        
        # * Inverse Spherical coordinate rotation (alpha, delta) > (phi, theta)
        theta_GPU = cp.arcsin(cp.sin(delta_GPU) * cp.sin(delta_p) + \
            cp.cos(delta_GPU) * cp.cos(delta_p) * cp.cos(alpha_GPU - alpha_p))

        COS_TERM_GPU = (cp.sin(delta_GPU) * cp.cos(delta_p) - \
            cp.cos(delta_GPU) * cp.sin(delta_p) * cp.cos(alpha_GPU - alpha_p)) / cp.cos(theta_GPU)
        SIN_TERM_GPU = - cp.cos(delta_GPU) * cp.sin(alpha_GPU - alpha_p) / cp.cos(theta_GPU)
        phi_GPU = phi_p + cp.arctan2(SIN_TERM_GPU, COS_TERM_GPU)

        # * Inverse Spherical projection (phi, theta) > (x, y)
        r_GPU = 1./cp.tan(theta_GPU)
        x_GPU = r_GPU * cp.sin(phi_GPU)
        y_GPU = - r_GPU * cp.cos(phi_GPU)

        return x_GPU, y_GPU

    def resamp_projection_sip(self, hdr_obj, hdr_targ, NSAMP=1024, RANDOM_SEED=10086):
        """Project the target pixel centers to the object frame using Cupy for TAN-SIP WCS"""
        NTX = int(hdr_targ["NAXIS1"]) 
        NTY = int(hdr_targ["NAXIS2"])

        XX_targ_GPU, YY_targ_GPU = cp.meshgrid(
            cp.arange(0, NTX) + 1., 
            cp.arange(0, NTY) + 1., 
            indexing='ij'
        )
        
        CWT = Cupy_WCS_Transform()
        if True:
            # read header | target
            KEYDICT, CD_GPU, A_SIP_GPU, B_SIP_GPU = CWT.read_sip_wcs(hdr_wcs=hdr_targ)
            CRPIX1_targ, CRPIX2_targ = KEYDICT["CRPIX1"], KEYDICT["CRPIX2"]
            CRVAL1_targ, CRVAL2_targ = KEYDICT["CRVAL1"], KEYDICT["CRVAL2"]
            LONPOLE_targ = KEYDICT["LONPOLE"]
            
            # forward transformation 0: (p1, p2) > (u, v) | target
            u_GPU, v_GPU = XX_targ_GPU.flatten() - CRPIX1_targ, YY_targ_GPU.flatten() - CRPIX2_targ
            
            # forward transformation 1: (u, v) > (x, y) | target
            U_GPU, V_GPU = CWT.sip_forward_transform(u_GPU=u_GPU, v_GPU=v_GPU, A_SIP_GPU=A_SIP_GPU, B_SIP_GPU=B_SIP_GPU)
            x_GPU, y_GPU = CWT.cd_transform(IMAGE_X_GPU=U_GPU, IMAGE_Y_GPU=V_GPU, CD_GPU=CD_GPU)

            # forward transformation 2 & 3 to sky: (x, y) > (phi, theta) > (alpha, delta) | target
            ra_GPU, dec_GPU = self.spherical_transformation(
                x_GPU=cp.deg2rad(x_GPU), 
                y_GPU=cp.deg2rad(y_GPU), 
                alpha_p=cp.deg2rad(CRVAL1_targ), 
                delta_p=cp.deg2rad(CRVAL2_targ),
                phi_p=cp.deg2rad(LONPOLE_targ)
            )   # in radians!
        
        if True:
            # read header | object
            KEYDICT, CD_GPU, A_SIP_GPU, B_SIP_GPU = CWT.read_sip_wcs(hdr_wcs=hdr_obj)
            CRPIX1_obj, CRPIX2_obj = KEYDICT["CRPIX1"], KEYDICT["CRPIX2"]
            CRVAL1_obj, CRVAL2_obj = KEYDICT["CRVAL1"], KEYDICT["CRVAL2"]
            LONPOLE_obj = KEYDICT["LONPOLE"]

            # inverse transformation 3 & 2: (alpha, delta) > (phi, theta) > (x, y) | object
            x_GPU, y_GPU = self.inverse_spherical_transformation(
                alpha_GPU=ra_GPU, 
                delta_GPU=dec_GPU, 
                alpha_p=cp.deg2rad(CRVAL1_obj), 
                delta_p=cp.deg2rad(CRVAL2_obj),
                phi_p=cp.deg2rad(LONPOLE_obj)
            )   # in radians!

            # inverse transformation 1: (x, y) > (u, v) | object
            # inverse CD transformation
            U_GPU, V_GPU = CWT.cd_transform_inv(
                WORLD_X_GPU=cp.rad2deg(x_GPU), 
                WORLD_Y_GPU=cp.rad2deg(y_GPU), 
                CD_GPU=CD_GPU
            )

            # inverse SIP transformation
            def FIT_INVERSE_SIP(KEYDICT, A_SIP_GPU, B_SIP_GPU, NSAMP, RANDOM_SEED):
                N0, N1 = KEYDICT["N0"], KEYDICT["N1"]
                CRPIX1, CRPIX2 = KEYDICT["CRPIX1"], KEYDICT["CRPIX2"]
                A_ORDER, B_ORDER = KEYDICT["A_ORDER"], KEYDICT["B_ORDER"]
                
                # sampling random coordinates
                u_GPU, v_GPU = CWT.random_coord_sampling(N0=N0, N1=N1, 
                    CRPIX1=CRPIX1, CRPIX2=CRPIX2, NSAMP=NSAMP, RANDOM_SEED=RANDOM_SEED)
                
                # fit polynomial form backward transformation | object
                U_GPU, V_GPU = CWT.sip_forward_transform(u_GPU=u_GPU, v_GPU=v_GPU, A_SIP_GPU=A_SIP_GPU, B_SIP_GPU=B_SIP_GPU)
                AP_lstsq_GPU, BP_lstsq_GPU = CWT.lstsq_sip_backward_transform(u_GPU=u_GPU, v_GPU=v_GPU, 
                    U_GPU=U_GPU, V_GPU=V_GPU, A_ORDER=A_ORDER, B_ORDER=B_ORDER)[2:4]
                return AP_lstsq_GPU, BP_lstsq_GPU

            AP_lstsq_GPU, BP_lstsq_GPU = FIT_INVERSE_SIP(
                KEYDICT=KEYDICT, A_SIP_GPU=A_SIP_GPU, B_SIP_GPU=B_SIP_GPU,
                NSAMP=NSAMP, RANDOM_SEED=RANDOM_SEED
            )

            FP_UV_GPU = CWT.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=KEYDICT["A_ORDER"])
            GP_UV_GPU = CWT.sip_backward_matrix(U_GPU=U_GPU, V_GPU=V_GPU, ORDER=KEYDICT["B_ORDER"])

            u_GPU, v_GPU = CWT.sip_backward_transform(U_GPU=U_GPU, V_GPU=V_GPU, 
                FP_UV_GPU=FP_UV_GPU, GP_UV_GPU=GP_UV_GPU, AP_GPU=AP_lstsq_GPU, BP_GPU=BP_lstsq_GPU)

            # inverse transformation 0: (u, v) > (p1, p2) | object
            XX_proj_GPU = CRPIX1_obj + u_GPU.reshape((NTX, NTY))
            YY_proj_GPU = CRPIX2_obj + v_GPU.reshape((NTX, NTY))

        return XX_proj_GPU, YY_proj_GPU

    def resamp_projection_astropy(self, hdr_obj, hdr_targ):
        """Mapping the target pixel centers to the object frame using Astropy"""
        # * read object WCS and target WCS
        w_obj = Read_WCS.RW(hdr=hdr_obj, VERBOSE_LEVEL=self.VERBOSE_LEVEL)
        w_targ = Read_WCS.RW(hdr=hdr_targ, VERBOSE_LEVEL=self.VERBOSE_LEVEL)
        
        # * maaping target pixel centers to the object frame
        NTX = int(hdr_targ["NAXIS1"]) 
        NTY = int(hdr_targ["NAXIS2"])
        
        XX_targ, YY_targ = np.meshgrid(np.arange(0, NTX)+1., np.arange(0, NTY)+1., indexing='ij')
        XY_targ = np.array([XX_targ.flatten(), YY_targ.flatten()]).T
        XY_proj = w_obj.all_world2pix(w_targ.all_pix2world(XY_targ, 1), 1)

        XX_proj = XY_proj[:, 0].reshape((NTX, NTY))
        YY_proj = XY_proj[:, 1].reshape((NTX, NTY))

        XX_proj_GPU = cp.array(XX_proj, dtype=np.float64)
        YY_proj_GPU = cp.array(YY_proj, dtype=np.float64)

        return XX_proj_GPU, YY_proj_GPU

    def frame_extension(self, XX_proj_GPU, YY_proj_GPU, PixA_obj_GPU, PAD_FILL_VALUE=0., NAN_FILL_VALUE=0.):
        """Extend the object frame for resampling"""
        NTX, NTY = XX_proj_GPU.shape
        NOX, NOY = PixA_obj_GPU.shape

        # * padding the object frame
        if self.RESAMP_METHOD == 'BILINEAR':
            KERHW = (1, 1)

        if self.RESAMP_METHOD == 'LANCZOS3':
            KERHW = (3, 3)

        # find the root index and shift with maximal kernel halfwidth (KERHW)
        # Note: the index ranges (RMIN --- RMAX) and (CMIN --- CMAX) can cover
        #       all pixels that interpolation may be use.

        RMIN = (math.floor(XX_proj_GPU.min().item()) - 1) - KERHW[0]
        RMAX = (math.floor(XX_proj_GPU.max().item()) - 1) + KERHW[0]
        RPAD = (-np.min([RMIN, 0]), np.max([RMAX - (NOX - 1), 0]))

        CMIN = (math.floor(YY_proj_GPU.min().item()) - 1) - KERHW[1]
        CMAX = (math.floor(YY_proj_GPU.max().item()) - 1) + KERHW[1]
        CPAD = (-np.min([CMIN, 0]), np.max([CMAX - (NOY - 1), 0]))

        PAD_WIDTH = (RPAD, CPAD)    
        PixA_Eobj_GPU = cp.pad(PixA_obj_GPU, PAD_WIDTH, mode='constant', constant_values=PAD_FILL_VALUE)
        if NAN_FILL_VALUE is not None:
            PixA_Eobj_GPU[cp.isnan(PixA_Eobj_GPU)] = NAN_FILL_VALUE
        NEOX, NEOY = PixA_Eobj_GPU.shape
        
        XX_Eproj_GPU = XX_proj_GPU + PAD_WIDTH[0][0]
        YY_Eproj_GPU = YY_proj_GPU + PAD_WIDTH[1][0]

        RMIN_E = (math.floor(XX_Eproj_GPU.min().item()) - 1) - KERHW[0]
        RMAX_E = (math.floor(XX_Eproj_GPU.max().item()) - 1) + KERHW[0]

        CMIN_E = (math.floor(YY_Eproj_GPU.min().item()) - 1) - KERHW[1]
        CMAX_E = (math.floor(YY_Eproj_GPU.max().item()) - 1) + KERHW[1]

        assert RMIN_E >= 0 and CMIN_E >= 0 
        assert RMAX_E < NEOX and CMAX_E < NEOY

        EProjDict = {}
        EProjDict["NTX"] = NTX
        EProjDict["NTY"] = NTY

        EProjDict["NOX"] = NOX
        EProjDict["NOY"] = NOY

        EProjDict["NEOX"] = NEOX
        EProjDict["NEOY"] = NEOY
        EProjDict["KERHW"] = KERHW

        EProjDict["XX_Eproj_GPU"] = XX_Eproj_GPU
        EProjDict["YY_Eproj_GPU"] = YY_Eproj_GPU

        return PixA_Eobj_GPU, EProjDict
    
    def resampling(self, PixA_Eobj_GPU, EProjDict, PIXEL_SCALE_FACTOR=1., THREAD_PER_BLOCK=8, USE_SHARED_MEMORY=False):
        """Resampling the object frame to the target frame using Cupy"""
        NTX = EProjDict["NTX"]
        NTY = EProjDict["NTY"]

        NEOX = EProjDict["NEOX"]
        NEOY = EProjDict["NEOY"]
        KERHW = EProjDict["KERHW"]

        XX_Eproj_GPU = EProjDict["XX_Eproj_GPU"]
        YY_Eproj_GPU = EProjDict["YY_Eproj_GPU"]

        # * Cupy configuration
        GPUManage = lambda NT: ((NT-1)//THREAD_PER_BLOCK + 1, min(NT, THREAD_PER_BLOCK))
        BpG_PIX0, TpB_PIX0 = GPUManage(NTX)
        BpG_PIX1, TpB_PIX1 = GPUManage(NTY)
        BpG_PIX, TpB_PIX = (BpG_PIX0, BpG_PIX1), (TpB_PIX0, TpB_PIX1, 1)

        PixA_resamp_GPU = cp.zeros((NTX, NTY), dtype=np.float64)

        if self.RESAMP_METHOD == "BILINEAR":

            if USE_SHARED_MEMORY:
                print("MeLOn WARNING: Shared memory is not implemented for bilinear resampling!")

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
            PixA_resamp_GPU *= PIXEL_SCALE_FACTOR
            
            if self.VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: Cuda resampling takes [%.6f s]' %(time.time() - t0))

        return PixA_resamp_GPU

class Cupy_ZoomRotate:
    @staticmethod
    def CZR(PixA_obj_GPU, ZOOM_SCALE_X=1., ZOOM_SCALE_Y=1., OUTSIZE_PARIRY_X='UNCHANGED', OUTSIZE_PARIRY_Y='UNCHANGED', \
        PATTERN_ROTATE_ANGLE=0.0, RESAMP_METHOD='BILINEAR', PAD_FILL_VALUE=0., NAN_FILL_VALUE=0., \
        THREAD_PER_BLOCK=8, USE_SHARED_MEMORY=False, VERBOSE_LEVEL=2):
        """ Zoom and rotate an image using Cupy """
        __author__ = "Lei Hu <leihu@andrew.cmu.edu>"
        
        # check inputs
        assert ZOOM_SCALE_X > 0.0
        assert ZOOM_SCALE_Y > 0.0
        assert 0.0 <= PATTERN_ROTATE_ANGLE < 360.0
        NAXIS1_ORI, NAXIS2_ORI = PixA_obj_GPU.shape

        # * determine the number of pixels of the zoomed image
        def NPIX_ORI2ZOOMED(NPIX_ORI, ZOOM_SCALE, OUTSIZE_PARIRY):
            if OUTSIZE_PARIRY == 'UNCHANGED':
                if NPIX_ORI % 2 == 0: OUTSIZE_UPARIRY = 'EVEN'
                if NPIX_ORI % 2 == 1: OUTSIZE_UPARIRY = 'ODD'
            if OUTSIZE_PARIRY == 'ODD': OUTSIZE_UPARIRY = 'ODD'
            if OUTSIZE_PARIRY == 'EVEN': OUTSIZE_UPARIRY = 'EVEN'
            HALFWIDTH_ORI = NPIX_ORI/2.0  # unit of original pixel
            if OUTSIZE_UPARIRY == 'EVEN':
                NPIX_ZOOMED = 2 * math.ceil(HALFWIDTH_ORI/ZOOM_SCALE)
            if OUTSIZE_UPARIRY == 'ODD':
                NPIX_ZOOMED = 2 * math.ceil((HALFWIDTH_ORI - ZOOM_SCALE/2.0)/ZOOM_SCALE) + 1
            return NPIX_ZOOMED

        NAXIS1_ZOOMED = NPIX_ORI2ZOOMED(NPIX_ORI=NAXIS1_ORI, ZOOM_SCALE=ZOOM_SCALE_X, OUTSIZE_PARIRY=OUTSIZE_PARIRY_X)
        NAXIS2_ZOOMED = NPIX_ORI2ZOOMED(NPIX_ORI=NAXIS2_ORI, ZOOM_SCALE=ZOOM_SCALE_Y, OUTSIZE_PARIRY=OUTSIZE_PARIRY_Y)

        def BACKWARD_TRANSFORM(X_ROTATED_GPU, Y_ROTATED_GPU):
            # * prepare for the frame transform
            CRPIX1_ZOOMED = 0.5 + NAXIS1_ZOOMED/2.
            CRPIX2_ZOOMED = 0.5 + NAXIS2_ZOOMED/2.

            ALPHA_INV = cp.deg2rad(-PATTERN_ROTATE_ANGLE)
            ROTMAT_INV_GPU = cp.array([
                [cp.cos(ALPHA_INV), -cp.sin(ALPHA_INV)], 
                [cp.sin(ALPHA_INV), cp.cos(ALPHA_INV)]
            ])

            # * frame transform from rotated to zoomed
            COORD_ROTATED_GPU = cp.array([
                X_ROTATED_GPU - CRPIX1_ZOOMED,
                Y_ROTATED_GPU - CRPIX2_ZOOMED
            ])
            
            COORD_ZOOMED_GPU = cp.dot(ROTMAT_INV_GPU, COORD_ROTATED_GPU)
            COORD_ZOOMED_GPU[0, :] += CRPIX1_ZOOMED
            COORD_ZOOMED_GPU[1, :] += CRPIX2_ZOOMED
            X_ZOOMED_GPU, Y_ZOOMED_GPU = COORD_ZOOMED_GPU
            
            # * frame transform from zoomed to original
            OFFSET_X = ZOOM_SCALE_X * (0.5 + NAXIS1_ZOOMED / 2.0) - (0.5 + NAXIS1_ORI / 2.0)
            OFFSET_Y = ZOOM_SCALE_Y * (0.5 + NAXIS2_ZOOMED / 2.0) - (0.5 + NAXIS2_ORI / 2.0)

            X_ORI_GPU = X_ZOOMED_GPU * ZOOM_SCALE_X - OFFSET_X
            Y_ORI_GPU = Y_ZOOMED_GPU * ZOOM_SCALE_Y - OFFSET_Y

            return X_ORI_GPU, Y_ORI_GPU

        XX_targ_GPU, YY_targ_GPU = cp.meshgrid(
            cp.arange(0, NAXIS1_ZOOMED) + 1., 
            cp.arange(0, NAXIS2_ZOOMED) + 1., 
            indexing='ij'
        )

        X_ORI_GPU, Y_ORI_GPU = BACKWARD_TRANSFORM(X_ROTATED_GPU=XX_targ_GPU.flatten(), Y_ROTATED_GPU=YY_targ_GPU.flatten())
        XX_proj_GPU = X_ORI_GPU.reshape((NAXIS1_ZOOMED, NAXIS2_ZOOMED))
        YY_proj_GPU = Y_ORI_GPU.reshape((NAXIS1_ZOOMED, NAXIS2_ZOOMED))

        GWR = Cupy_Resampling(RESAMP_METHOD=RESAMP_METHOD, VERBOSE_LEVEL=VERBOSE_LEVEL)
        PixA_Eobj_GPU, EProjDict = GWR.frame_extension(
            XX_proj_GPU=XX_proj_GPU, YY_proj_GPU=YY_proj_GPU, PixA_obj_GPU=PixA_obj_GPU,
            PAD_FILL_VALUE=PAD_FILL_VALUE, NAN_FILL_VALUE=NAN_FILL_VALUE
        )
        
        PIXEL_SCALE_FACTOR = ZOOM_SCALE_X * ZOOM_SCALE_Y
        PixA_resamp_GPU = GWR.resampling(
            PixA_Eobj_GPU=PixA_Eobj_GPU, EProjDict=EProjDict, PIXEL_SCALE_FACTOR=PIXEL_SCALE_FACTOR, 
            THREAD_PER_BLOCK=THREAD_PER_BLOCK, USE_SHARED_MEMORY=USE_SHARED_MEMORY
        )

        return PixA_resamp_GPU
