import os
import sys
import numpy as np
from sympy import *
import os.path as pa
from copy import deepcopy
from astropy.io import fits
from astropy.convolution import convolve
from sfft.utils.meta.MultiProc import Multi_Proc

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

# * Symbol Convention NOTE
#    X (x) / Y (y) ----- ScaledFortranCoor
#    e.g. pixel (3, 5) center corresponds (x, y) = (4.0/N0, 6.0/N1)
#    NOTE: Without special definition, MeLOn convention refers to X (x) / Y (y) as FortranCoor.
#
# * Express spatial-varying kernel 
#    a. SFFT DictForm
#        SVK_xy = sum_ab (A_xyab * K_ab), where
#        A_xyab = sum_ij (a_ijab * x^i * y^j)
#        NOTE K_ab is sfft-delta basis
#        NOTE dictionary is given in a form a[(i, j)][a, b] = a_ijab
#        NOTE here a is literally sfft_å, keep in mind that sfft_a = N0*N1*sfft_å = N0*N1*a
#        NOTE this formula is slightly modified from that in sfft paper, so that the new form (with decimal subscripts xy) 
#             allows us to derive the kernel at any given ScaledFortranCoor, not just confined to pixel centers. 
#
#    b. Standard DictForm
#        SVK_xy = sum_ab (B_xyab * D_ab), where 
#        B_xyab = sum_ij (b_ijab * x^i * y^j)
#        NOTE D_ab is Cartesian-delta basis
#        NOTE dictionary is given in a form b[(i, j)][a, b] = b_ijab
#
#    c. PSFEx DictForm
#        SVK_xy = sum_ab (B_xyab * D_ab), where 
#        B_xyab = sum_ij (c_ijab * ((N0*x - refx)/scax)^i * ((N1*y - refy)/scay)^j)
#        NOTE D_ab is Cartesian-delta basis
#        NOTE dictionary is given in a form c[(i, j)][a, b] = c_ijab with 
#        parameters scax, scay, refx, refy, which all are capsulated in PSFMODEL.
#        where scax & scay are scale factors, while refx & refy are FortranCoor of reference point.

class ReadPSFMODEL:
    @staticmethod
    def RP(PSFMODEL):

        # * Remarks on PSFEx 
        # @ PSFEx produce PSFMODEL with shape (psf_varsize, psf_ysize, psf_xsize),
        #      where psf_varsize = int((d+1)*(d+2)/2), d is the fitting polynomial degree.
        #
        # @ Visit the header[1] of .psf, and define 
        #      poly_deg = hdr['POLDEG1']   (that is, d)
        #      refx, refy = hdr['POLZERO1'], hdr['POLZERO2']
        #      scax, scay = hdr['POLSCAL1'], hdr['POLSCAL2']
        #      rx = (FortranCoor_x - refx) / scax
        #      ry = (FortranCoor_y - refy) / scay
        #
        # @ PSFEx order is a reversed standard i-j order !
        #     Standard-Order: [(i, j) for i in range(DKx+1) for j in range(DKy+1-i)] 
        #     PSFEx-Order:    [(i, j) for j in range(DKy+1) for i in range(DKx+1-j)]
        #                     say, DKx=DKy= 2, that is [1, rx, rx^2, ry, rx*ry, ry^2]

        hdl = fits.open(PSFMODEL)
        hdr = hdl[1].header
        pd = int(hdr['POLDEG1'])
        
        DKx, DKy = pd, pd
        print('MeLOn CheckPoint: POLDEG = %d !' %pd)
        refx, refy = float(hdr['POLZERO1']), float(hdr['POLZERO2'])
        scax, scay = float(hdr['POLSCAL1']), float(hdr['POLSCAL2'])
        PSF_Array = hdl[1].data['PSF_MASK'][0]
        L0, L1 = PSF_Array[0].shape
        PParams = (pd, refx, refy, scax, scay)
        hdl.close()

        L_ = 0
        PSFEx_dict = {}
        for j in range(DKy+1):
            for i in range(DKx+1-j):
                PSFEx_dict[(i, j)] = PSF_Array[L_].copy() 
                L_ += 1

        return PSFEx_dict, PParams


class SVKDictConverter:
    def __init__(self, DKx, DKy):
        self.DKx = DKx
        self.DKy = DKy
    
    def PSF2ST(self, N0, N1, PSFEx_dict, PParams):

        L0, L1 = PSFEx_dict[(0, 0)].shape
        pd, refx, refy, scax, scay = PParams
        if self.DKx != pd or self.DKy != pd:
            sys.exit('MeLOn ERROR: POLDEG is inconsistent with given DKx & DKy !')

        if pd == 2:
            PCUBE, Standard_dict = [], {}
            for i in range(self.DKx+1):
                for j in range(self.DKy+1-i):
                    PCUBE.append(PSFEx_dict[(i, j)])
            PCUBE = np.array(PCUBE)

            c00ab, c01ab, c02ab, c10ab, c11ab, c20ab = PCUBE
            print('MeLOn Report: Robust Calculation with ready-made equations !')
            Standard_dict[(0, 0)] = (refx**2 * c20ab * scay**2 + refx * refy * c11ab * scax * scay \
                - refx * c10ab * scax * scay**2 + refy**2 * c02ab * scax**2 - refy * c01ab * scax**2 * scay \
                    + c00ab * scax**2 * scay**2) / (scax**2 * scay**2)
            Standard_dict[(0, 1)] = (-N1 * refx * c11ab * scay - 2 * N1 * refy * c02ab * scax + \
                N1 * c01ab * scax * scay)/(scax * scay**2)
            Standard_dict[(0, 2)] = N1**2 * c02ab / scay**2
            Standard_dict[(1, 0)] = (-2 * N0 * refx * c20ab * scay - N0 * refy * c11ab * scax + N0 * c10ab * scax * scay) / (scax**2 * scay)
            Standard_dict[(1, 1)] = N0 * N1 * c11ab / (scax * scay)
            Standard_dict[(2, 0)] = N0**2 * c20ab / scax**2

        if pd != 2:
            # Initialization
            Standard_dict = {}
            for i in range(self.DKx+1):
                for j in range(self.DKy+1-i):
                    Standard_dict[(i, j)] = np.zeros((L0, L1)).astype(float)
            
            print('MeLOn Report: Stupid & Slow Calculation with sympy equations !')
            for aa in range(L0):
                for bb in range(L1):
                    Eq = 0.0
                    x, y = symbols('x, y')
                    for ip in range(self.DKx+1):
                        for jp in range(self.DKy+1-ip):
                            Eq += PSFEx_dict[(ip, jp)][aa, bb] * (((N0*x-refx)/scax)**ip) * (((N1*y-refy)/scay)**jp)
                    
                    CoDict = {}
                    a = Poly(Eq, x, y)
                    for k, (i, j) in enumerate(a.monoms()):
                        CoDict[(i, j)] = a.coeffs()[k]
                    for i in range(self.DKx+1):
                        for j in range(self.DKy+1-i):
                            if (i, j) in CoDict.keys():
                                Standard_dict[(i, j)][aa, bb] = CoDict[(i, j)]

        return Standard_dict
    
    def ST2SFFT(self, Standard_dict):
        L0, L1 = Standard_dict[(0, 0)].shape
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)

        # * Convert Standard-dict into Sfft-dict (Basis Change)
        Sfft_dict = deepcopy(Standard_dict)   # NOTE WARNING always use deepcopy for list & dictionary
        for i in range(self.DKx+1):
            for j in range(self.DKy+1-i):
                Sfft_dict[(i, j)][0+w0, 0+w1] = np.sum(Standard_dict[(i, j)])
        return Sfft_dict

    def SFFT2ST(self, Sfft_dict):
        L0, L1 = Sfft_dict[(0, 0)].shape
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)

        # * Convert Sfft-dict into Standard-dict (Basis Change)
        Standard_dict = deepcopy(Sfft_dict)
        for i in range(self.DKx+1):
            for j in range(self.DKy+1-i):
                Standard_dict[(i, j)][0+w0, 0+w1] = 2 * Sfft_dict[(i, j)][0+w0, 0+w1] - np.sum(Sfft_dict[(i, j)])
        return Standard_dict


class SFFTDictCoverter:
    def __init__(self, DKx, DKy, L0, L1):
        self.DKx = DKx
        self.DKy = DKy
        self.L0 = L0
        self.L1 = L1
        self.w0 = int((L0-1)/2)
        self.w1 = int((L1-1)/2)
        self.REF_ij = np.array([(i, j) for i in range(self.DKx+1) for j in range(self.DKy+1-i)]).astype(int)
        self.REF_ab = np.array([(a_pos - self.w0, b_pos - self.w1) for a_pos in range(L0) for b_pos in range(L1)]).astype(int)
        self.SREF_ijab = np.array([(ij, ab) for ij in range(self.REF_ij.shape[0]) for ab in range(self.REF_ab.shape[0])]).astype(int)
        self.Fijab = self.SREF_ijab.shape[0]
    
    def Dict2Flatten(self, Sfft_dict):
        a_ijab = []
        for idx in range(self.Fijab):
            ij, ab = self.SREF_ijab[idx]
            a, b = self.REF_ab[ab]
            i, j = self.REF_ij[ij]
            a_ijab.append(Sfft_dict[(i, j)][a+self.w0, b+self.w1])
        a_ijab = np.array(a_ijab)
        return a_ijab

    def Flatten2Dict(self, a_ijab):
        Sfft_dict = {}
        for i in range(self.DKx+1):
            for j in range(self.DKy+1-i):
                Sfft_dict[(i, j)] = np.zeros((self.L0, self.L1)).astype(float)   

        for idx in range(self.Fijab):
            ij, ab = self.SREF_ijab[idx]
            a, b = self.REF_ab[ab]
            i, j = self.REF_ij[ij]
            Sfft_dict[(i, j)][a+self.w0, b+self.w1] = a_ijab[idx]
        return Sfft_dict


class ReadSFFTSolution:
    @staticmethod
    def RSS(FITS_Solution):

        phr = fits.getheader(FITS_Solution, ext=0)
        N0, N1 = int(phr['N0']), int(phr['N1'])
        L0, L1 = int(phr['L0']), int(phr['L1'])
        DK, DB = int(phr['DK']), int(phr['DB'])
        Fij = int(phr['FIJ'])
        Fpq = int(phr['FPQ'])

        # NOTE the information differential background is dropped here
        a_ijab = (fits.getdata(FITS_Solution, ext=0)[0])[:-Fpq]
        a_ijab /= N0*N1   # NOTE sfft_a to sfft_å (i.e., convert to a_ijab here)
        Sfft_dict = SFFTDictCoverter(DKx=DK, DKy=DK, L0=L0, L1=L1).Flatten2Dict(a_ijab)

        return a_ijab, Sfft_dict


class SVKRealization:
    def __init__(self, N0, N1, DKx, DKy, XY_q):
        self.N0 = N0
        self.N1 = N1
        self.DKx = DKx
        self.DKy = DKy
        self.XY_q = XY_q.copy()       # Input FortranCoor [general convention]
        self.XY_q[:, 0] /= N0         # Convert to ScaledFortranCoor [local convention]
        self.XY_q[:, 1] /= N1         # Convert to ScaledFortranCoor [local convention]

    def ST(self, Standard_dict):
        B = np.array([self.XY_q[:, 0]**i * self.XY_q[:, 1]**j for i in range(self.DKx+1) for j in range(self.DKy+1-i)])
        SSS = np.array([Standard_dict[(i, j)] for i in range(self.DKx+1) for j in range(self.DKy+1-i)])
        KStack = np.tensordot(B, SSS, (0, 0))
        return KStack 

    def PSF(self, PSFEx_dict, PParams):
        pd, refx, refy, scax, scay = PParams
        if self.DKx != pd or self.DKy != pd:
            sys.exit('MeLOn ERROR: POLDEG is inconsistent with given DKx & DKy !')
        
        C = np.array([((self.XY_q[:, 0] * self.N0 - refx) / scax)**i * ((self.XY_q[:, 1] * self.N1 - refy) / scay)**j 
            for i in range(self.DKx+1) for j in range(self.DKy+1-i)])
        SSS = np.array([PSFEx_dict[(i, j)] for i in range(self.DKx+1) for j in range(self.DKy+1-i)])
        KStack = np.tensordot(C, SSS, (0, 0))      # These kernels are PSF at given positions XY_q
        return KStack 
    
    def SFFT(self, Sfft_dict):
        Standard_dict = SVKDictConverter(DKx=self.DKx, DKy=self.DKy).SFFT2ST(Sfft_dict=Sfft_dict)
        KStack = self.ST(Standard_dict=Standard_dict)
        return KStack


# @Spatial-Varying Convolution based on sfft 
# * Baisc definitions 
#   SVConv(R, g_ijab) = sum_ijab (g_ijab * T^ij * (R © K_ab))
#   SFFT_SVConv(R, g_ijab) = sum_ijab (g_ijab * (R^ij © K_ab))
#   NOTE SVConv(1, g_ijab) = sum_ij00 (g_ij00* T^ij)
#
# * Final Form accepted by this program
#   Since we can use directly use Reconstruction code to calculte SV-convolution, 
#   It's necessary to convert given c_ijab to b_ijab by frame-transformation and 
#   then a_ijab via basis-change, finally, we need reform a_ijab as a flatten for sfft.
#
# * WHAT kinds of inputs are avilable for this program
#   [ST] given Standard-dict
#   [PSFEx] convolve with PSFEx-PSF by given psf-model directly
#
# * Bias of SFFT-SVConv
#    SFFT-SVConv is slightly biased (towards overestimate) due to the approaximation: 
#    the bias can be trivial for most cases as it is typically ~ 0.01-0.5 mmag.
#    The bias is the error of relative flux zeropoint estimated by SFFT.
#    REF: Hartung (2012) developed a CUDA approach to precisely calculate SVConv. 
#    Frankly, our scenario is deprecated, it is just an alternative.
   
class SFFT_SVConv:
    @staticmethod
    def SSVC(PixA_obj, Standard_dict=None, Sfft_dict=None, PSFMODEL=None, DK=2, \
        backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8):
        
        N0, N1 = PixA_obj.shape
        if np.sum(np.isnan(PixA_obj)) != 0:
            sys.exit('MeLOn ERROR: Input Image has NaN !')

        if Standard_dict is not None: mode = 'ST-DICT'
        if Sfft_dict is not None: mode = 'SFFT-DICT'
        if PSFMODEL is not None: mode = 'PSFEx-MODEL'
        
        if mode == 'ST-DICT':
            L0, L1 = Standard_dict[(0, 0)].shape

        if mode == 'SFFT-DICT':
            L0, L1 = Sfft_dict[(0, 0)].shape
        
        if mode == 'PSFEx-MODEL':
            # Read PSFMODEL, then Convert PSFEx-dict into Standard-dict
            PSFEx_dict, PParams = ReadPSFMODEL.RP(PSFMODEL=PSFMODEL)
            if DK != PParams[0]: sys.exit('MeLOn ERROR: POLDEG is inconsistent with given DK !')
            Standard_dict = SVKDictConverter(DKx=DK, DKy=DK).PSF2ST(N0=N0, N1=N1, PSFEx_dict=PSFEx_dict, PParams=PParams)
            L0, L1 = PSFEx_dict[(0, 0)].shape
        
        assert L0 == L1   # FIXME Slightly Modify SFFTConfigure if you want to allow L0 != L1
        KerHW = int((L0-1)/2)
        print('MeLOn CheckPoint: Convolution Kernel-HalfWidth [%d]' %KerHW)
        
        if mode in ['ST-DICT', 'PSFEx-MODEL']:
            # Convert Standard-dict into Sfft-dict
            Sfft_dict = SVKDictConverter(DKx=DK, DKy=DK).ST2SFFT(Standard_dict=Standard_dict)

        # Convert Sfft-dict to be flatten-form
        a_ijab = SFFTDictCoverter(DKx=DK, DKy=DK, L0=L0, L1=L1).Dict2Flatten(Sfft_dict=Sfft_dict)

        # Construct Soultion in sfft
        # @ add trivial background terms for sfft calculation
        # @ convert a (sfft_å) to sfft_a = N0*N1*a
        DB = 2  # trivial
        Fpq = int((DB+1)*(DB+2)/2)
        Solution = list(a_ijab * N0*N1) + [0.0]*Fpq
        Solution = np.array(Solution).astype(np.float64)
        
        from SFFTConfigure import SingleSFFTConfigure
        from SFFTSubtract import ElementalSFFTSubtract

        SFFTConfig = SingleSFFTConfigure.SSC(NX=N0, NY=N1, KerHW=KerHW, \
            KerPolyOrder=DK, BGPolyOrder=DB, ConstPhotRatio=True, \
            backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)  # NOTE ConstPhotRatio does not matter
        
        _PixA_J = np.zeros(PixA_obj.shape).astype(np.float64)
        _PixA_DIFF = ElementalSFFTSubtract.ESS(PixA_I=PixA_obj, PixA_J=_PixA_J, \
            SFFTConfig=SFFTConfig, SFFTSolution=Solution, Subtract=True, \
            backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)[1]
        PixA_SFFT_SVConv = _PixA_J - _PixA_DIFF

        """
        #  THIS IS WARONG !!!!    [FYI]
        #  Emperically, we can calculate bias correction map BCMap = SVConv(1) / SFFT-SVConv(1), 
        #  then BCMap * SFFT-SVConv(IMG) is basically unbiased.

        PixA_1 = np.ones(PixA_obj.shape).astype(np.float64)
        _PixA_J = np.zeros(PixA_obj.shape).astype(np.float64)
        _PixA_DIFF = ElementalSFFTSubtract.ESS(PixA_I=PixA_1, PixA_J=_PixA_J, \
            SFFTConfig=SFFTConfig, SFFTSolution=Solution, Subtract=True, \
            backend=backend, CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)[1]
        RHS1 = _PixA_J - _PixA_DIFF

        LHS1 = np.zeros(PixA_obj.shape).astype(np.float64)
        PixA_X, PixA_Y = np.mgrid[:N0, :N1]
        PixA_CX = ((PixA_X+1.0) / N0).astype(np.float64)
        PixA_CY = ((PixA_Y+1.0) / N1).astype(np.float64)
        for i in range(DK+1):
            for j in range(DK+1-i):
                a_ij00 = Sfft_dict[(i, j)][w0, w1]
                LHS1 += a_ij00 * PixA_CX**i * PixA_CY**j

        PixA_BCMap = LHS1 / RHS1
        PixA_SFFT_SVConv =  PixA_BCMap * PixA_SFFT_SVConv
        
        """

        return PixA_SFFT_SVConv


class GRID_SVConv:
    @staticmethod
    def GSVC(PixA_obj, AllocatedL, KStack, nproc=32):
        
        """
        # Typcical Example of AllocatedL & KStack
        TiHW = 10
        N0, N1 = 1024, 1024

        lab = 0
        TiN = 2*TiHW+1
        XY_TiC = []
        
        AllocatedL = np.zeros((N0, N1))
        for xs in np.arange(0, N0, TiN):
            xe = np.min([xs+TiN, N0])
            for ys in np.arange(0, N1, TiN):
                ye = np.min([ys+TiN, N1])
                AllocatedL[xs: xe, ys: ye] = lab
                x_q = 0.5 + xs + (xe - xs) / 2.0   # tile-center (x)
                y_q = 0.5 + ys + (ye - ys) / 2.0   # tile-center (y)
                XY_TiC.append([x_q, y_q])
                lab += 1

        XY_TiC = np.array(XY_TiC)
        
        """

        # * Allocated LabelMap has same shape with PixA_obj to show the image segmentation (Compact-Box!)
        #   Kernel-Stack gives corresponding convolution kernel for each label
        #   For each segment, we would extract Esegment according to the label with a extended boundary
        #   Perform convolution and then send the values within this segment to output image.
        
        N0, N1 = PixA_obj.shape
        PixA_GRID_SVConv = np.zeros((N0, N1)).astype(float)
        Nseg, L0, L1 = KStack.shape
        
        w0 = int((L0-1)/2)
        w1 = int((L1-1)/2)
        IBx, IBy = w0+1, w1+1
        def func_conv(idx):
            Ker = KStack[idx]
            lX, lY = np.where(AllocatedL == idx)
            xs, xe = lX.min(), lX.max()
            ys, ye = lY.min(), lY.max()

            xEs, xEe = np.max([0, xs-IBx]), np.min([N0-1, xe+IBx])
            yEs, yEe = np.max([0, ys-IBy]), np.min([N1-1, ye+IBy])
            PixA_Emini = PixA_obj[xEs: xEe+1, yEs: yEe+1]
            PixA_ = convolve(PixA_Emini, Ker, boundary='extend', normalize_kernel=False)
            fragment = PixA_[xs-xEs: (xs-xEs)+(xe+1-xs), ys-yEs: (ys-yEs)+(ye+1-ys)]
            xyrg = xs, xe+1, ys, ye+1
            return xyrg, fragment

        taskid_lst = np.arange(Nseg)
        mydict = Multi_Proc.MP(taskid_lst=taskid_lst, func=func_conv, nproc=nproc, mode='mp')
        for idx in taskid_lst:
            xyrg, fragment = mydict[idx]
            PixA_GRID_SVConv[xyrg[0]: xyrg[1], xyrg[2]: xyrg[3]] = fragment

        return PixA_GRID_SVConv

