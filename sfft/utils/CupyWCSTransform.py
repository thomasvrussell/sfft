import cupy as cp
from sfft.utils.ReadWCS import Read_WCS

__last_update__ = "2024-09-19"
__author__ = "Shu Liu <shl159@pitt.edu> & Lei Hu <leihu@andrew.cmu.edu>"

class Cupy_WCS_Transform:
    def __init__(self):
        return None

    def read_cd_wcs(self, hdr_wcs):
        """
        # * Note on the CD matrix transformation:
        The sky coordinate (x, y) relative to reference point can be connected with
        the image coordinate (u, v) relative to reference point by the CD matrix:
            [x]   [CD1_1 CD1_2] [u]
            [y] = [CD2_1 CD2_2] [v]
        where CD1_1, CD1_2, CD2_1, CD2_2 are stored in the FITS header.

        """
        KEYDICT, CD = Read_WCS.read_cd_wcs( hdr_wcs )
        CD_GPU = cp.array( CD, dtype=cp.float64 )
        return KEYDICT, CD_GPU

    def read_sip_wcs(self, hdr_wcs):
        """
        # * Note on the SIP transformation:
        The image coordinate (u, v) relative to reference point and we have undistorted image coordinate (U, V):
            U = u + f(u, v)
            V = v + g(u, v)
        where f(u, v) and g(u, v) are the SIP distortion functions with polynomial form:
            f(u, v) = sum_{p, q} A_{pq} u^p v^q, p + q <= A_ORDER
            g(u, v) = sum_{p, q} B_{pq} u^p v^q, p + q <= B_ORDER
        These coefficients are stored in the FITS header and define a foward transformation from (u, v) to (U, V).

        The sky coordinate (x, y) relative to reference point can be connected with (U, V) by the reversible CD matrix:
            [x]   [CD1_1 CD1_2] [U]
            [y] = [CD2_1 CD2_2] [V]
        So the forward transformation & CD matrix (with considering the reference point) is equivalent to a pix2world function.

        The reverse question is how to obtain the backward transformation from undistorted (U, V) to distorted (u, v).
        Once we have the backward transformation, we can combine a reversed CD matrix
        (with considering the reference point) to get the world2pix function.

        A simple approach (not accurate) is to assume the forward transformation also has a polynomial form:
            u = U + fp(U, V)
            v = V + gp(U, V)
        where
            fp(U, V) = sum_{p, q} AP_{pq} U^p V^q, p + q <= A_ORDER
            gp(U, V) = sum_{p, q} BP_{pq} U^p V^q, p + q <= B_ORDER

        The coefficients AP_{pq} and BP_{pq} can be obtained by solving the linear equations separately.
            [u - U] = [U^p*V^q] [AP_{pq}]

            shapes:
            b: [u - U]   | (N, 1),
            A: [U^p*V^q] | (N, (A_ORDER+1)*(A_ORDER+2)/2)
            x: [AP_{pq}] | ((A_ORDER+1)*(A_ORDER+2)/2, 1)

            [v - V] = [U^p*V^q] [BP_{pq}]
            b: [v - V]   | (N, 1),
            A: [U^p*V^q] | (N, (B_ORDER+1)*(B_ORDER+2)/2)
            x: [BP_{pq}] | ((B_ORDER+1)*(B_ORDER+2)/2, 1)

            The two matrices encode the "backward" transformation in fp(U, V) and gp(U, V).
            Let's denote the two matrices as FP_UV and GP_UV, respectively.

        # * A WCS example with SIP distortion
        CTYPE1  = 'RA---TAN-SIP'
        CTYPE2  = 'DEC--TAN-SIP'

        CRPIX1  =               2044.0
        CRPIX2  =               2044.0

        CD1_1   = 2.65458074767927E-05
        CD1_2   = -1.2630331175158E-05
        CD2_1   = 1.38917634264203E-05
        CD2_2   = 2.57785917553667E-05

        CRVAL1  =    9.299707734492053
        CRVAL2  =  -43.984663860136145

        A_ORDER =                    4
        A_0_2   =     -4.774423702E-10
        A_0_3   =     -2.462744787E-14
        A_0_4   =       2.28913156E-17
        A_1_1   =      1.076615801E-09
        A_1_2   =     -9.939904553E-14
        A_1_3   =     -5.845336863E-17
        A_2_0   =     -3.717865118E-10
        A_2_1   =     -4.966118494E-15
        A_2_2   =      6.615859199E-17
        A_3_0   =     -3.817793356E-15
        A_3_1   =      3.310020049E-17
        A_4_0   =     -5.166048303E-19

        B_ORDER =                    4
        B_0_2   =      1.504792792E-09
        B_0_3   =     -2.662833066E-15
        B_0_4   =     -8.383877471E-20
        B_1_1   =      1.204553316E-10
        B_1_2   =       6.57748238E-14
        B_1_3   =     -6.998508554E-18
        B_2_0   =      4.136013664E-10
        B_2_1   =      5.339208582E-14
        B_2_2   =      6.063403412E-17
        B_3_0   =      1.486316541E-14
        B_3_1   =     -9.102668806E-17
        B_4_0   =     -5.174323631E-17

        """
        assert hdr_wcs["CTYPE1"] == "RA---TAN-SIP"
        assert hdr_wcs["CTYPE2"] == "DEC--TAN-SIP"

        KEYDICT, CD = Read_WCS.read_linear_part_of_wcs( hdr_wcs )
        A_ORDER = int(hdr_wcs["A_ORDER"])
        B_ORDER = int(hdr_wcs["B_ORDER"])
        KEYDICT['A_ORDER'] = A_ORDER
        KEYDICT['B_ORDER'] = B_ORDER

        CD_GPU = cp.array( CD, dtype=cp.float64 )
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
        Nsamp = len(U_GPU)
        P_UV_GPU = cp.zeros((Nsamp, (ORDER+1)*(ORDER+2)//2), dtype=cp.float64)
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

    def random_coord_sampling(self, N0, N1, CRPIX1, CRPIX2, Nsamp=1024, RANDOM_SEED=10086):
        """Random sampling of coordinates"""
        if RANDOM_SEED is not None:
            cp.random.seed(RANDOM_SEED)
        u_GPU = cp.random.uniform(0.5, N0+0.5, Nsamp, dtype=cp.float64) - CRPIX1
        v_GPU = cp.random.uniform(0.5, N1+0.5, Nsamp, dtype=cp.float64) - CRPIX2
        return u_GPU, v_GPU
