import numpy as np
from astropy import units as u
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
# version: Aug 4, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.2"

class Symmetric_Match:
    @staticmethod
    def SM(XY_A, XY_B, tol, return_distance=False):

        """
        a) XY_A, XY_B are two collections of points for matching with shape (NUM_A, 2) and (NUM_B, 2)
        b) The function func_query(PO, POR) returns the matched points for PO in POR.
           For any given po in PO, assign the index of the nearest point in POR (respect to po), 
           and if the nearest distance still exceeds the tolerance tol, assign the trivial index POR.shape[0].
        c) A point pair < a, b > is called symmetrically matched iff they are nearest to each other.
        """

        def func_query(PO, POR):
            # Find k-nearest (here k=1) objects in POR
            # [0] distance to the POR object
            # [1] index of the POR object, if index = POR.shape[0] 
            #     the query fails within given tolerance.
            Tree = cKDTree(POR)
            DIST, IDX = Tree.query(PO, k=1, distance_upper_bound=tol)
            return DIST, IDX
        
        NUM_A, NUM_B = XY_A.shape[0], XY_B.shape[0]
        DIST_alpha, IDX_alpha = func_query(PO=XY_A, POR=XY_B)
        DIST_beta, IDX_beta = func_query(PO=XY_B, POR=XY_A)
        
        MATCH_IDX_A, MATCH_IDX_B, MATCH_DIST = [], [], []
        if NUM_A < NUM_B:
            for i in np.where((IDX_alpha < NUM_B))[0]:
                j = IDX_alpha[i]
                if IDX_beta[j] == i:
                    MATCH_IDX_A.append(i)
                    MATCH_IDX_B.append(j)
                    MATCH_DIST.append(DIST_alpha[i])
        else:
            for v in np.where((IDX_beta < NUM_A))[0]:
                u = IDX_beta[v]
                if IDX_alpha[u] == v:
                    MATCH_IDX_A.append(u)
                    MATCH_IDX_B.append(v)
                    MATCH_DIST.append(DIST_beta[v])
        
        Symm = np.array([MATCH_IDX_A, MATCH_IDX_B]).T
        MATCH_DIST = np.array(MATCH_DIST)
        
        if return_distance:
            return Symm, MATCH_DIST
        else:
            return Symm

class Sky_Symmetric_Match:
    @staticmethod
    def SSM(RD_A, RD_B, tol, return_distance=False):
        
        # NOTE: SkyCoord default frame is icrs, only having subtle difference with (equinox='J2000', frame='fk5')
        Skycoors_A = SkyCoord(ra=RD_A[:, 0], dec=RD_A[:, 1], unit=(u.deg, u.deg))
        Skycoors_B = SkyCoord(ra=RD_B[:, 0], dec=RD_B[:, 1], unit=(u.deg, u.deg))
        
        NUM_A, NUM_B = RD_A.shape[0], RD_B.shape[0]
        if NUM_A < NUM_B:
            _matchres = match_coordinates_sky(Skycoors_A.frame, Skycoors_B.frame)[:2]
            IDX_alpha, DIST_alpha = _matchres[0], _matchres[1].arcsec
            AIDX_cone = np.where(DIST_alpha < tol)[0]
            BIDX_conep = IDX_alpha[AIDX_cone]
            DIST_conep = DIST_alpha[AIDX_cone]
            
            IDX_beta = match_coordinates_sky(Skycoors_B[BIDX_conep].frame, Skycoors_A.frame)[0]
            SYMM_MASK = AIDX_cone == IDX_beta
            MATCH_IDX_A = AIDX_cone[SYMM_MASK]
            MATCH_IDX_B = BIDX_conep[SYMM_MASK]
            MATCH_DIST = DIST_conep[SYMM_MASK]
        else:
            _matchres = match_coordinates_sky(Skycoors_B.frame, Skycoors_A.frame)[:2]
            IDX_beta, DIST_beta = _matchres[0], _matchres[1].arcsec
            BIDX_cone = np.where(DIST_beta < tol)[0]
            AIDX_conep = IDX_beta[BIDX_cone]
            DIST_conep = DIST_beta[BIDX_cone]
            
            IDX_alpha = match_coordinates_sky(Skycoors_A[AIDX_conep].frame, Skycoors_B.frame)[0]
            SYMM_MASK = BIDX_cone == IDX_alpha
            MATCH_IDX_B = BIDX_cone[SYMM_MASK]
            MATCH_IDX_A = AIDX_conep[SYMM_MASK]
            MATCH_DIST = DIST_conep[SYMM_MASK]
        
        Symm = np.array([MATCH_IDX_A, MATCH_IDX_B]).T

        if return_distance:
            return Symm, MATCH_DIST
        else:
            return Symm
