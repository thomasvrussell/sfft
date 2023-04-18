import numpy as np
from copy import deepcopy
from astropy.io import fits
from scipy.interpolate import BSpline
# version: Apr 17, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class Read_SFFTSolution:

    def __init__(self):

        """
        # Notes on the representation of a spatially-varying kernel (SVK)
        # (1) Using SFFT Dictionary: Sfft_dict[(i, j)][a, b] = ac_ijab
        #     Polynomial: SVK_xy = sum_ab (Ac_xyab * K_ab), where Ac_xyab = sum_ij (ac_ijab * x^i * y^j)
        #     B-Spline:   SVK_xy = sum_ab (Ac_xyab * K_ab), where Ac_xyab = sum_ij (ac_ijab * B_i(x) * B_j(y))
        #          
        #     NOTE: K_ab is modified delta basis (see the SFFT paper).
        #           Ac (ac) is the symbol A (a) with circle hat in SFFT paper, note that ac = a/(N0*N1).
        #           B_i(x) is the i-th B-Spline base function along X
        #           B_j(x) is the j-th B-Spline base function along Y
        #    
        #     NOTE: Although (x, y) are the integer indices of a pixel in SFFT paper,
        #           we actually use a scaled image coordinate (i.e., ScaledFortranCoor) of
        #           pixel center in our implementation. Note that above equations allow for 
        #           all possible image coordinates (not just pixel center).
        #
        # (2) Using 'Standard' Dictionary: Standard_dict[(i, j)][a, b] = b_ijab
        #     Polynomial: SVK_xy = sum_ab (B_xyab * D_ab), where B_xyab = sum_ij (b_ijab * x^i * y^j)
        #     B-Spline:   SVK_xy = sum_ab (B_xyab * D_ab), where B_xyab = sum_ij (b_ijab * B_i(x) * B_j(y))
        #
        #     NOTE: K_ab is standard Cartesian delta basis. Likewise, 
        #           (x, y) is ScaledFortranCoor that allows for all possible image coordinates.
        #
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
    
    def FromArray(self, Solution, KerSpType, N0, N1, DK, L0, L1, Fi, Fj, Fpq):

        if KerSpType == 'Polynomial':
            REF_ij = np.array([(i, j) for i in range(DK+1) for j in range(DK+1-i)]).astype(int)
        
        if KerSpType == 'B-Spline':
            REF_ij = np.array([(i, j) for i in range(Fi) for j in range(Fj)]).astype(np.int32)

        w0, w1 = int((L0 - 1) / 2), int((L1 - 1) / 2)
        REF_ab = np.array([(a_pos - w0, b_pos - w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(int)
        SREF_ijab = np.array([(ij, ab) for ij in range(REF_ij.shape[0]) for ab in range(REF_ab.shape[0])]).astype(int)
        Fijab = SREF_ijab.shape[0]
        
        Sfft_dict = {}
        if KerSpType == 'Polynomial':
            for i in range(DK+1):
                for j in range(DK+1-i):
                    Sfft_dict[(i, j)] = np.zeros((L0, L1)).astype(float)
        
        if KerSpType == 'B-Spline':
            for i in range(Fi):
                for j in range(Fj):
                    Sfft_dict[(i, j)] = np.zeros((L0, L1)).astype(float)
        
        a_ijab = Solution[:-Fpq]      # drop differential background
        ac_ijab = a_ijab / (N0*N1)    # convert a to ac

        for idx in range(Fijab):
            ij, ab = SREF_ijab[idx]
            a, b = REF_ab[ab]
            i, j = REF_ij[ij]

            # add offsets w0/w1 for non-negative indices
            Sfft_dict[(i, j)][a+w0, b+w1] = ac_ijab[idx]   
        
        return Sfft_dict

    def FromFITS(self, FITS_Solution):
        
        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        phdr = fits.getheader(FITS_Solution, ext=0)
        
        KerSpType = phdr['KSPTYPE']
        N0, N1 = int(phdr['N0']), int(phdr['N1'])
        w0, w1 = int(phdr['W0']), int(phdr['W1'])
        DK = int(phdr['DK'])
        
        L0, L1 = int(phdr['L0']), int(phdr['L1'])
        Fi, Fj = int(phdr['FI']), int(phdr['FJ'])
        Fpq = int(phdr['FPQ'])

        Sfft_dict = self.FromArray(Solution=Solution, KerSpType=KerSpType, \
            N0=N0, N1=N1, w0=w0, w1=w1, DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq)

        return Sfft_dict

class SVKDict_ST2SFFT:
    @staticmethod
    def convert(KerSpType, DK, Fi, Fj, Standard_dict):

        L0, L1 = Sfft_dict[(0, 0)].shape
        w0, w1 = int((L0 - 1) / 2), int((L1 - 1) / 2)

        Sfft_dict = deepcopy(Standard_dict)
        if KerSpType == 'Polynomial':
            for i in range(DK+1):
                for j in range(DK+1-i):
                    Sfft_dict[(i, j)][0 + w0, 0 + w1] = np.sum(Standard_dict[(i, j)])

        if KerSpType == 'B-Spline':
            for i in range(Fi):
                for j in range(Fj):
                    Sfft_dict[(i, j)][0 + w0, 0 + w1] = np.sum(Standard_dict[(i, j)])

        return Sfft_dict

class SVKDict_SFFT2ST:
    @staticmethod
    def convert(KerSpType, DK, Fi, Fj, Sfft_dict):

        L0, L1 = Sfft_dict[(0, 0)].shape
        w0, w1 = int((L0-1)/2), int((L1-1)/2)

        Standard_dict = deepcopy(Sfft_dict)
        if KerSpType == 'Polynomial':
            for i in range(DK+1):
                for j in range(DK+1-i):
                    Standard_dict[(i, j)][0 + w0, 0 + w1] = 2 * Sfft_dict[(i, j)][0 + w0, 0 + w1] - \
                        np.sum(Sfft_dict[(i, j)])
                    
        if KerSpType == 'B-Spline':
            for i in range(Fi):
                for j in range(Fj):
                    Standard_dict[(i, j)][(0 + w0, 0 + w1)] = 2 * Sfft_dict[(i, j)][(0 + w0, 0 + w1)] - \
                        np.sum(Sfft_dict[(i, j)])

        return Standard_dict

class Realize_MatchingKernel:

    def __init__(self, XY_q, VERBOSE_LEVEL=2):
        self.XY_q = XY_q
        self.VERBOSE_LEVEL = VERBOSE_LEVEL

    def FromArray(self, Solution, KerSpType, KerIntKnotX, KerIntKnotY, N0, N1, DK, L0, L1, Fi, Fj, Fpq):
        
        # convert requested coordinates
        sXY_q = self.XY_q.astype(float)   # input requested coordinates in FortranCoor [global convention]
        sXY_q[:, 0] /= N0                 # convert to ScaledFortranCoor [local convention]
        sXY_q[:, 1] /= N1                 # convert to ScaledFortranCoor [local convention]

        # read SFFTSolution and convert to standard dictionary
        Sfft_dict = Read_SFFTSolution().FromArray(Solution=Solution, KerSpType=KerSpType, \
            N0=N0, N1=N1, DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq)
        Standard_dict = SVKDict_SFFT2ST.convert(KerSpType=KerSpType, \
            DK=DK, Fi=Fi, Fj=Fj, Sfft_dict=Sfft_dict)

        # realize kernels at given coordinates
        if KerSpType == 'Polynomial':

            B = np.array([sXY_q[:, 0]**i * sXY_q[:, 1]**j for i in range(DK+1) for j in range(DK+1-i)])
            SSS = np.array([Standard_dict[(i, j)] for i in range(DK+1) for j in range(DK+1-i)])
            KerStack = np.tensordot(B, SSS, (0, 0))   # (Num_request, NX_kernel, NY_kernel)
        
        if KerSpType == 'B-Spline':

            def Create_BSplineBasis_Req(N, IntKnot, BSplineDegree, ReqCoord):
                BSplineBasis_Req = []
                #PixCoord = (1.0+np.arange(N))/N
                Knot = np.concatenate(([0.5]*(BSplineDegree+1), IntKnot, [N+0.5]*(BSplineDegree+1)))/N
                Nc = len(IntKnot) + BSplineDegree + 1    # number of control points/coeffcients
                for idx in range(Nc):
                    Coeff = (np.arange(Nc) == idx).astype(float)
                    BaseFunc = BSpline(t=Knot, c=Coeff, k=BSplineDegree, extrapolate=False)
                    BSplineBasis_Req.append(BaseFunc(ReqCoord))
                BSplineBasis_Req = np.array(BSplineBasis_Req)
                return BSplineBasis_Req

            KerSplBasisX_q = Create_BSplineBasis_Req(N=N0, IntKnot=KerIntKnotX, BSplineDegree=DK, ReqCoord=sXY_q[:, 0])
            KerSplBasisY_q = Create_BSplineBasis_Req(N=N1, IntKnot=KerIntKnotY, BSplineDegree=DK, ReqCoord=sXY_q[:, 1])

            B = np.array([KerSplBasisX_q[i] * KerSplBasisY_q[j] for i in range(Fi) for j in range(Fj)])
            SSS = np.array([Standard_dict[(i, j)] for i in range(Fi) for j in range(Fj)])
            KerStack = np.tensordot(B, SSS, (0, 0))

        return KerStack
    
    def FromFITS(self, FITS_Solution):

        phdr = fits.getheader(FITS_Solution, ext=0)
        KerHW = phdr['KERHW']
        KerSpType = phdr['KSPTYPE']
        KerIntKnotX = [phdr['KIKX%d' %i] for i in range(phdr['NKIKX'])]
        KerIntKnotY = [phdr['KIKY%d' %i] for i in range(phdr['NKIKY'])]

        N0, N1 = int(phdr['N0']), int(phdr['N1'])
        DK = int(phdr['DK'])
        
        L0, L1 = int(phdr['L0']), int(phdr['L1'])
        Fi, Fj = int(phdr['FI']), int(phdr['FJ'])
        Fpq = int(phdr['FPQ'])

        if self.VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- SFFT CONFIGURATION --//--//--//--//-- ')
            if KerSpType == 'Polynomial':
                print('\n ---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(DK, KerHW))
            if KerSpType == 'B-Spline':
                print('\n ---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                    %(len(KerIntKnotX), len(KerIntKnotY), DK, KerHW))
        
        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        KerStack = self.FromArray(Solution=Solution, KerSpType=KerSpType, \
            KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, N0=N0, N1=N1, \
            DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq)

        return KerStack

class Realize_FluxScaling:

    def __init__(self, XY_q, VERBOSE_LEVEL=2):
        self.XY_q = XY_q
        self.VERBOSE_LEVEL = VERBOSE_LEVEL

    def FromArray(self, Solution, KerSpType, KerIntKnotX, KerIntKnotY, N0, N1, DK, L0, L1, Fi, Fj, Fpq):

        # read SFFTSolution and convert to standard dictionary
        Sfft_dict = Read_SFFTSolution().FromArray(Solution=Solution, KerSpType=KerSpType, \
            N0=N0, N1=N1, DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq)
        
        # calculate flux scaling at given coordinates
        # > Recall that SFFT Dictionary: Sfft_dict[(i, j)][a, b] = ac_ijab
        #    Polynomial: SVK_xy = sum_ab (Ac_xyab * K_ab), where Ac_xyab = sum_ij (ac_ijab * x^i * y^j)
        #    B-Spline:   SVK_xy = sum_ab (Ac_xyab * K_ab), where Ac_xyab = sum_ij (ac_ijab * B_i(x) * B_j(y))
        #    NOTE the flux scaling equals to the sum of matching kernel at any given coordinate,
        #         which means the amplification of the overall flux by the convolution in the image subtraction.
        #         In SFFT, it is fully controlled by Ac_xy00

        sXY_q = self.XY_q.astype(float)   # input requested coordinates in FortranCoor [global convention]
        sXY_q[:, 0] /= N0                 # convert to ScaledFortranCoor [local convention]
        sXY_q[:, 1] /= N1                 # convert to ScaledFortranCoor [local convention]

        w0, w1 = int((L0 - 1) / 2), int((L1 - 1) / 2)
        SFFT_FSCAL_q = np.zeros(self.XY_q.shape[0]).astype(float)
        if KerSpType == 'Polynomial':
            for i in range(DK+1):
                for j in range(DK+1-i):
                    SFFT_FSCAL_q += Sfft_dict[(i, j)][w0, w1] * sXY_q[:, 0]**i * sXY_q[:, 1]**j
        
        if KerSpType == 'B-Spline':

            def Create_BSplineBasis_Req(N, IntKnot, BSplineDegree, ReqCoord):
                BSplineBasis_Req = []
                #PixCoord = (1.0+np.arange(N))/N
                Knot = np.concatenate(([0.5]*(BSplineDegree+1), IntKnot, [N+0.5]*(BSplineDegree+1)))/N
                Nc = len(IntKnot) + BSplineDegree + 1    # number of control points/coeffcients
                for idx in range(Nc):
                    Coeff = (np.arange(Nc) == idx).astype(float)
                    BaseFunc = BSpline(t=Knot, c=Coeff, k=BSplineDegree, extrapolate=False)
                    BSplineBasis_Req.append(BaseFunc(ReqCoord))
                BSplineBasis_Req = np.array(BSplineBasis_Req)
                return BSplineBasis_Req

            KerSplBasisX_q = Create_BSplineBasis_Req(N=N0, IntKnot=KerIntKnotX, BSplineDegree=DK, ReqCoord=sXY_q[:, 0])
            KerSplBasisY_q = Create_BSplineBasis_Req(N=N1, IntKnot=KerIntKnotY, BSplineDegree=DK, ReqCoord=sXY_q[:, 1])

            for i in range(Fi):
                for j in range(Fj):
                    SFFT_FSCAL_q += Sfft_dict[(i, j)][w0, w1] * KerSplBasisX_q[i] * KerSplBasisY_q[j]

        return SFFT_FSCAL_q
    
    def FromFITS(self, FITS_Solution):

        phdr = fits.getheader(FITS_Solution, ext=0)
        KerHW = phdr['KERHW']
        KerSpType = phdr['KSPTYPE']
        KerIntKnotX = [phdr['KIKX%d' %i] for i in range(phdr['NKIKX'])]
        KerIntKnotY = [phdr['KIKY%d' %i] for i in range(phdr['NKIKY'])]

        N0, N1 = int(phdr['N0']), int(phdr['N1'])
        DK = int(phdr['DK'])
        
        L0, L1 = int(phdr['L0']), int(phdr['L1'])
        Fi, Fj = int(phdr['FI']), int(phdr['FJ'])
        Fpq = int(phdr['FPQ'])

        if self.VERBOSE_LEVEL in [1, 2]:
            print('\n --//--//--//--//-- SFFT CONFIGURATION --//--//--//--//-- ')
            if KerSpType == 'Polynomial':
                print('\n ---//--- Polynomial Kernel | KerSpDegree %d | KerHW %d ---//---' %(DK, KerHW))
            if KerSpType == 'B-Spline':
                print('\n ---//--- B-Spline Kernel | Internal Knots %d,%d | KerSpDegree %d | KerHW %d ---//---' \
                    %(len(KerIntKnotX), len(KerIntKnotY), DK, KerHW))

        Solution = fits.getdata(FITS_Solution, ext=0)[0]
        SFFT_FSCAL_q = self.FromArray(Solution=Solution, KerSpType=KerSpType, \
            KerIntKnotX=KerIntKnotX, KerIntKnotY=KerIntKnotY, \
            N0=N0, N1=N1, DK=DK, L0=L0, L1=L1, Fi=Fi, Fj=Fj, Fpq=Fpq)

        return SFFT_FSCAL_q
