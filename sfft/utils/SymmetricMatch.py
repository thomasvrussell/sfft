import numpy as np
from astropy import units as u
from scipy.spatial import cKDTree
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

# MeLon Notes
# @ Simple Symmetric Match Algorithm
#   a) POA, POB are two collections of points for matching with shape (Num_A, 2) and (Num_B, 2)
#   b) Function COSE(PO, POR) return an array attached to PO, that is, for any given po in PO, 
#      just assign the index of the nearest point (respect to po) from POR, and note if the nearest 
#      distance still exceeds the tolerance tol, then assign the trivial index POR.shape[0].
#   c) A pair < a, b > is called symmetric match iff COSE(PO, POR)[a] = b and COSE(POR, PO)[b] = a
#      where a | b is a index of PO | POR, it actually means a-th of PO and b-th POR 'stay nearest each other'.
#   d) To find the collection of all symmetric pairs, the economical way is check one by one from the shorter one of PO & POR.
#      Such collection has shape (Num_symmetric_matched_pairs, 2) 

class Symmetric_Match:
    @staticmethod
    def SM(POA, POB, tol):
        Na, Nb = POA.shape[0], POB.shape[0]
        def COSE(PO, POR):
            Tree = cKDTree(POR)
            # [0] means distance [1] means index, return IND = POR.shape[0] means fail to query
            IND = Tree.query(PO, k=1, distance_upper_bound=tol)[1]
            return IND
        cosA = COSE(PO=POA, POR=POB)
        cosB = COSE(PO=POB, POR=POA)
        Matcha, Matchb = [], []
        if Na < Nb:
            for i in np.where((cosA < Nb))[0]:
                j = cosA[i]
                if cosB[j] == i:
                    Matcha.append(i)
                    Matchb.append(j)
        else:
            for v in np.where((cosB < Na))[0]:
                u = cosB[v]
                if cosA[u] == v:
                    Matcha.append(u)
                    Matchb.append(v)
        Symm = np.array([Matcha, Matchb]).T     

        return Symm


class Sky_Symmetric_Match:
    @staticmethod
    def SSM(RD_A, RD_B, tol):
        # NOTE SkyCoord default frame: icrs, only subtle difference w.r.t (equinox='J2000', frame='fk5')
        Skycoors_A = SkyCoord(ra=RD_A[:, 0], dec=RD_A[:, 1], unit=(u.deg, u.deg))
        Skycoors_B = SkyCoord(ra=RD_B[:, 0], dec=RD_B[:, 1], unit=(u.deg, u.deg))

        if RD_A.shape[0] < RD_B.shape[0]:
            idx_alpha, distance = match_coordinates_sky(Skycoors_A.frame, Skycoors_B.frame)[:2]
            Aidx_cone = np.where(distance.arcsec < tol)[0]
            Bidx_cone = idx_alpha[Aidx_cone]
            idx_beta = match_coordinates_sky(Skycoors_B[Bidx_cone].frame, Skycoors_A.frame)[0]
            symm_mask = Aidx_cone == idx_beta
            Aidx_cross = Aidx_cone[symm_mask]
            Bidx_cross = Bidx_cone[symm_mask]
        else:
            idx_beta, distance = match_coordinates_sky(Skycoors_B.frame, Skycoors_A.frame)[:2]
            Bidx_cone = np.where(distance.arcsec < tol)[0]
            Aidx_cone = idx_beta[Bidx_cone]
            idx_alpha = match_coordinates_sky(Skycoors_A[Aidx_cone].frame, Skycoors_B.frame)[0]
            symm_mask = Bidx_cone == idx_alpha
            Bidx_cross = Bidx_cone[symm_mask]
            Aidx_cross = Aidx_cone[symm_mask]
        Symm = np.array([Aidx_cross, Bidx_cross]).T

        return Symm
