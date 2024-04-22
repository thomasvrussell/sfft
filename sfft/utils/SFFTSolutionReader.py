import numpy as np
from copy import deepcopy
from astropy.io import fits
# version: Feb 24, 2023

__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class Read_SFFTSolution:

    def __init__(self):

        """
        * Notes on the representation of a spatially-varying kernel (SVK)
        # [1] Using SFFT Dictionary
        #     > SVK_xy = sum_ab (Ac_xyab * K_ab), where Ac_xyab = sum_ij (ac_ijab * x^i * y^j)
        #     > Sfft_dict[(i, j)][a, b] = ac_ijab
        #  
        #     NOTE: K_ab is modified delta basis (see the sfft paper)
        #           Ac (ac) is the symbol A (a) with circle hat in sfft paper, note that ac = a/(N0*N1).
        #           Although (x, y) are the integer indices of a pixel in sfft paper, we actually used a 
        #           scaled image coordinate (i.e., ScaledFortranCoor) of the pixel center in our implementation.
        #           Here, above equations allows for all possible image coordinates (not just pixel center).
        #
        # [2] Using 'Standard' Dictionary
        #     > SVK_xy = sum_ab (B_xyab * D_ab), where B_xyab = sum_ij (b_ijab * x^i * y^j)
        #     > Standard_dict[(i, j)][a, b] = b_ijab
        #     
        #     NOTE D_ab is the standard Cartesian-delta basis
        #          Likewise, (x, y) is ScaledFortranCoor of all possible image coordinates.
        #
        # P.S. Coordinate Example: Given Pixel Location at (row=3, column=5) in an image of size (1024, 2048)
        #      FortranCoor = (4.0, 6.0) | ScaledFortranCoor = (4.0/1024, 6.0/2048)
        #
        # P.S. In principle, a and b are in the range of [-w0, w0 + 1) and [-w1 , w1+ 1), respectively.
        #      However, Sfft_dict[(i, j)] or Standard_dict[(i, j)] has to be a matrix with 
        #      non-negative indices starting from 0. In practice, the indices were tweaked 
        #      as [a, b] > [a + w0, b + w1].
        #
        """

        pass

    def FromArray(self, Solution, N0, N1, L0, L1, DK, Fpq):

        DKx, DKy = DK, DK
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)
        
        REF_ij = np.array([(i, j) for i in range(DKx+1) for j in range(DKy+1-i)]).astype(int)
        REF_ab = np.array([(a_pos - w0, b_pos - w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(int)
        
        SREF_ijab = np.array([(ij, ab) for ij in range(REF_ij.shape[0]) for ab in range(REF_ab.shape[0])]).astype(int)
        Fijab = SREF_ijab.shape[0]

        # initialize SFFT Dictionary 
        Sfft_dict = {}
        for i in range(DKx+1):
            for j in range(DKy+1-i):
                Sfft_dict[(i, j)] = np.zeros((L0, L1)).astype(float)

        # fill the SFFT Dictionary
        a_ijab = Solution[:-Fpq]      # drop differential background
        ac_ijab = a_ijab / (N0*N1)    # a > ac

        for idx in range(Fijab):
            ij, ab = SREF_ijab[idx]
            a, b = REF_ab[ab]
            i, j = REF_ij[ij]

            # add offsets w0/w1 for non-negative indices
            Sfft_dict[(i, j)][a+w0, b+w1] = ac_ijab[idx]   
        
        return Sfft_dict

    def FromFITS(self, FITS_Solution):

        phr = fits.getheader(FITS_Solution, ext=0)
        N0, N1 = int(phr['N0']), int(phr['N1'])
        L0, L1 = int(phr['L0']), int(phr['L1'])
        DK = int(phr['DK'])
        Fpq = int(phr['FPQ'])

        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        Sfft_dict = self.FromArray(Solution=Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)

        return Sfft_dict

class SVKDict_ST2SFFT:
    @staticmethod
    def convert(DKx, DKy, Standard_dict):
        # convert Standard-dict into Sfft-dict (Basis change)
        L0, L1 = Standard_dict[(0, 0)].shape
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)
        Sfft_dict = deepcopy(Standard_dict)
        for i in range(DKx+1):
            for j in range(DKy+1-i):
                Sfft_dict[(i, j)][0+w0, 0+w1] = np.sum(Standard_dict[(i, j)])
        return Sfft_dict

class SVKDict_SFFT2ST:
    @staticmethod
    def convert(DKx, DKy, Sfft_dict):
        # convert Sfft-dict into Standard-dict (Basis change)
        L0, L1 = Sfft_dict[(0, 0)].shape
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)
        Standard_dict = deepcopy(Sfft_dict)
        for i in range(DKx+1):
            for j in range(DKy+1-i):
                Standard_dict[(i, j)][0+w0, 0+w1] = 2 * Sfft_dict[(i, j)][0+w0, 0+w1] - \
                    np.sum(Sfft_dict[(i, j)])
        return Standard_dict

class Realize_MatchingKernel:

    def __init__(self, XY_q):
        self.XY_q = XY_q

    def FromArray(self, Solution, N0, N1, L0, L1, DK, Fpq):

        DKx, DKy = DK, DK
        # convert requested coordinates
        sXY_q = self.XY_q.astype(float)   # input requested coordinates in FortranCoor [global convention]
        sXY_q[:, 0] /= N0                 # convert to ScaledFortranCoor [local convention]
        sXY_q[:, 1] /= N1                 # convert to ScaledFortranCoor [local convention]

        # read SFFTSolution and convert to standard dictionary
        Sfft_dict = Read_SFFTSolution().FromArray(Solution=Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)
        Standard_dict = SVKDict_SFFT2ST.convert(DKx=DKx, DKy=DKy, Sfft_dict=Sfft_dict)

        # realize kernels at given coordinates
        B = np.array([sXY_q[:, 0]**i * sXY_q[:, 1]**j for i in range(DKx+1) for j in range(DKy+1-i)])
        SSS = np.array([Standard_dict[(i, j)] for i in range(DKx+1) for j in range(DKy+1-i)])
        KerStack = np.tensordot(B, SSS, (0, 0))   # (Num_request, NX_kernel, NY_kernel)
        
        return KerStack
    
    def FromFITS(self, FITS_Solution):

        phr = fits.getheader(FITS_Solution, ext=0)
        N0, N1 = int(phr['N0']), int(phr['N1'])
        L0, L1 = int(phr['L0']), int(phr['L1'])
        DK = int(phr['DK'])
        Fpq = int(phr['FPQ'])

        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        KerStack = self.FromArray(Solution=Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)

        return KerStack

class Realize_FluxScaling:

    def __init__(self, XY_q):
        self.XY_q = XY_q

    def FromArray(self, Solution, N0, N1, L0, L1, DK, Fpq):

        DKx, DKy = DK, DK
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)

        # convert requested coordinates
        sXY_q = self.XY_q.astype(float)   # input requested coordinates in FortranCoor [global convention]
        sXY_q[:, 0] /= N0                 # convert to ScaledFortranCoor [local convention]
        sXY_q[:, 1] /= N1                 # convert to ScaledFortranCoor [local convention]

        # read SFFTSolution and convert to standard dictionary
        Sfft_dict = Read_SFFTSolution().FromArray(Solution=Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)
        
        # calculate the flux scaling through SFFT convolution at given coordinates
        # NOTE: flux scaling Ac_xy00 = sum_ij (ac_ij00 * x^i * y^j) and ac_ijab = Sfft_dict[(i, j)][a, b].
        #       in practice, ac_ij00 = Sfft_dict[(i, j)][w0, w1]  # w0, w1 are offsets
        # NOTE: the flux scaling is consistent with the sum of matching kernel at any given coordinate.
        #       it indicates the amplification of the overall flux by the convolution in the image subtraction.

        SFFT_FSCAL_q = np.zeros(self.XY_q.shape[0]).astype(float)
        for i in range(DKx+1):
            for j in range(DKy+1-i):
                SFFT_FSCAL_q += Sfft_dict[(i, j)][w0, w1] * sXY_q[:, 0]**i * sXY_q[:, 1]**j

        return SFFT_FSCAL_q
    
    def FromFITS(self, FITS_Solution):

        phr = fits.getheader(FITS_Solution, ext=0)
        N0, N1 = int(phr['N0']), int(phr['N1'])
        L0, L1 = int(phr['L0']), int(phr['L1'])
        DK = int(phr['DK'])
        Fpq = int(phr['FPQ'])

        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        SFFT_FSCAL_q = self.FromArray(Solution=Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq)

        return SFFT_FSCAL_q
