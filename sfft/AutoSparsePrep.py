import os
import sys
import warnings
import fastremap
import numpy as np
import os.path as pa
from astropy.io import fits
import scipy.ndimage as ndimage
from astropy.table import Table, Column, hstack
from sfft.utils.SymmetricMatch import Symmetric_Match
from sfft.utils.HoughMorphClassifier import Hough_MorphClassifier

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class Auto_SparsePrep:
    def __init__(self, FITS_REF, FITS_SCI):

        self.FITS_REF = FITS_REF
        self.FITS_SCI = FITS_SCI
    
    def Hough(self, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', DETECT_THRESH=2.0, \
        BoundarySIZE=30, BeltHW=0.2, MAGD_THRESH=0.12, StarExt_iter=4):
        
        # ********************* Determine SubSources ********************* #

        # * Use Hough-MorphClassifer to identify GoodSources in REF & SCI
        def main_hough(FITS_obj):
            # NOTE Byproducts: SATLEVEL & FWHM Estimation & SEGMENTATION map
            # NOTE DETECT_THRESH can affect SEGMENTATION map
            Hmc = Hough_MorphClassifier.MakeCatalog(FITS_obj=FITS_obj, GAIN_KEY=GAIN_KEY, \
                SATUR_KEY=SATUR_KEY, BACK_TYPE='MANUAL', BACK_VALUE='0.0', BACK_SIZE=64, \
                BACK_FILTERSIZE=3, DETECT_THRESH=DETECT_THRESH, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
                BACKPHOTO_TYPE='LOCAL', CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, \
                BoundarySIZE=BoundarySIZE, AddSNR=False)   # FIXME SEx-Configuration is Customizable
            Hc = Hough_MorphClassifier.Classifier(AstSEx=Hmc[0], BeltHW=BeltHW, Return_HPS=False)
            
            AstSEx_GS = Hmc[0][Hc[2]]
            SATLEVEL = fits.getheader(FITS_obj, ext=0)[SATUR_KEY]
            FHWM, PixA_SEG = Hc[0], Hmc[1][0].astype(int)
            return AstSEx_GS, SATLEVEL, FHWM, PixA_SEG

        AstSEx_GSr, SATLEVEL_REF, FWHM_REF, PixA_SEGr = main_hough(self.FITS_REF)
        AstSEx_GSs, SATLEVEL_SCI, FWHM_SCI, PixA_SEGs = main_hough(self.FITS_SCI)
        XY_GSr = np.array([AstSEx_GSr['X_IMAGE'], AstSEx_GSr['Y_IMAGE']]).T
        XY_GSs = np.array([AstSEx_GSs['X_IMAGE'], AstSEx_GSs['Y_IMAGE']]).T

        # * Determine Matched-GoodSources [MGS]
        tol = np.sqrt((FWHM_REF/3.0)**2 + (FWHM_SCI/3.0)**2)   # FIXME Customizable
        Symm = Symmetric_Match.SM(POA=XY_GSr, POB=XY_GSs, tol=tol)
        AstSEx_MGSr = AstSEx_GSr[Symm[:, 0]]
        AstSEx_MGSs = AstSEx_GSs[Symm[:, 1]]

        # * Apply constrain on magnitude-difference to obtain SubSources [SS]
        MAG_MGSr = np.array(AstSEx_MGSr['MAG_AUTO'])
        MAG_MGSs = np.array(AstSEx_MGSs['MAG_AUTO'])
        MAGD = MAG_MGSs - MAG_MGSr
        MAGDm = np.median(MAGD)
        Avmask = np.abs(MAGD - MAGDm) < MAGD_THRESH

        AstSEx_SSr = AstSEx_MGSr[Avmask]
        AstSEx_SSs = AstSEx_MGSs[Avmask]
        for coln in AstSEx_SSr.colnames:
            AstSEx_SSr[coln].name = coln + '_REF'
            AstSEx_SSs[coln].name = coln + '_SCI'
        AstSEx_SS = hstack([AstSEx_SSr, AstSEx_SSs])
        AstSEx_SS.add_column(Column(1+np.arange(len(AstSEx_SS)), name='SEGLABEL'), index=0)
        print('MeLOn CheckPoint: Number of SubSources out of Matched-GoodSources [%d / %d] !' \
            %(len(AstSEx_SS), len(MAGD)))

        # ********************* Determine ActiveMask ********************* #

        PixA_REF = fits.getdata(self.FITS_REF, ext=0).T
        if np.issubdtype(PixA_REF.dtype, np.integer):
            PixA_REF = PixA_REF.astype(np.float64)
        
        PixA_SCI = fits.getdata(self.FITS_SCI, ext=0).T
        if np.issubdtype(PixA_SCI.dtype, np.integer):
            PixA_SCI = PixA_SCI.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SatMask_REF = PixA_REF >= SATLEVEL_REF
            SatMask_SCI = PixA_SCI >= SATLEVEL_SCI

        # * Relabel SS-islands & Flip labels of NotSS-islands        
        SEGL_SSr = np.array(AstSEx_SS['SEGLABEL_REF']).astype(int)
        SEGL_SSs = np.array(AstSEx_SS['SEGLABEL_SCI']).astype(int)
        SEGL_SS = np.array(AstSEx_SS['SEGLABEL']).astype(int)

        Mappings_REF = {}
        for label_o, label_n in zip(SEGL_SSr, SEGL_SS): 
            Mappings_REF[label_o] = -label_n
        fastremap.remap(PixA_SEGr, Mappings_REF, preserve_missing_labels=True, in_place=True)   # NOTE UPDATE
        PixA_SEGr *= -1

        Mappings_SCI = {}
        for label_o, label_n in zip(SEGL_SSs, SEGL_SS):
            Mappings_SCI[label_o] = -label_n
        fastremap.remap(PixA_SEGs, Mappings_SCI, preserve_missing_labels=True, in_place=True)   # NOTE UPDATE
        PixA_SEGs *= -1

        # * Define ProhibitedZone
        NaNmask_U = None
        NaNmask_REF = np.isnan(PixA_REF)
        NaNmask_SCI = np.isnan(PixA_SCI)
        ProZone = np.logical_or(PixA_SEGr < 0, PixA_SEGs < 0)
        if NaNmask_REF.any() or NaNmask_SCI.any():
            NaNmask_U = np.logical_or(NaNmask_REF, NaNmask_SCI)
            ProZone[NaNmask_U] = True

        # * Make SFFTLmap > ActiveMask > PixA_mREF & PixA_mSCI
        SFFTLmap = np.max(np.array([PixA_SEGr, PixA_SEGs]), axis=0)
        SFFTLmap[ProZone] = 0      # NOTE After-burn equivalent operation
        struct0 = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct0, StarExt_iter)
        SFFTLmap = ndimage.grey_dilation(SFFTLmap, footprint=struct)
        SFFTLmap[ProZone] = -128
        ActiveMask = SFFTLmap > 0

        # NOTE: The preparation can guarantee that mREF & mSCI are NaN-Free !
        ActivePROP = np.sum(ActiveMask) / (ActiveMask.shape[0] * ActiveMask.shape[1])
        print('MeLOn CheckPoint: ActiveMask Pixel Proportion [%s]' %('{:.2%}'.format(ActivePROP)))

        PixA_mREF = PixA_REF.copy()
        PixA_mSCI = PixA_SCI.copy()
        PixA_mREF[~ActiveMask] = 0.0      # NOTE REF has been sky-subtracted
        PixA_mSCI[~ActiveMask] = 0.0      # NOTE SCI has been sky-subtracted

        # * Create sfft-prep master dictionary
        SFFTPrepDict = {}
        SFFTPrepDict['PixA_REF'] = PixA_REF
        SFFTPrepDict['PixA_SCI'] = PixA_SCI
        SFFTPrepDict['REF-SAT-Mask'] = SatMask_REF
        SFFTPrepDict['SCI-SAT-Mask'] = SatMask_SCI
        SFFTPrepDict['Union-NaN-Mask'] = NaNmask_U

        SFFTPrepDict['SATLEVEL_REF'] = SATLEVEL_REF
        SFFTPrepDict['SATLEVEL_SCI'] = SATLEVEL_SCI
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        SFFTPrepDict['SExCatalog-SubSource'] = AstSEx_SS
        SFFTPrepDict['SFFT-LabelMap'] = SFFTLmap
        SFFTPrepDict['Active-Mask'] = ActiveMask
        SFFTPrepDict['PixA_mREF'] = PixA_mREF
        SFFTPrepDict['PixA_mSCI'] = PixA_mSCI

        return SFFTPrepDict
    
    def Refinement(self, SFFTPrepDict, trSubtract=False, trConvdSide=None, \
        SFFTConfig=None, backend='Pycuda', CUDA_DEVICE='0', NUM_CPU_THREADS=8, \
        RATIO_THRESH=3.0, XY_PriorBan=None):

        # * Read from input SFFTPrepDict, note they will be updated in the refinement 
        AstSEx_SS = SFFTPrepDict['SExCatalog-SubSource']
        SFFTLmap = SFFTPrepDict['SFFT-LabelMap']
        ActiveMask = SFFTPrepDict['Active-Mask']
        PixA_mREF = SFFTPrepDict['PixA_mREF']
        PixA_mSCI = SFFTPrepDict['PixA_mSCI']

        SurvMask_TS = None
        if trSubtract:
            from sfft.sfftcore.SFFTSubtract import ElementalSFFTSubtract
            
            # * Trigger a trial SFFT Element-Subtraction on mREF & mSCI
            if trConvdSide == 'REF':
                PixA_trDIFF = ElementalSFFTSubtract.ESS(PixA_I=PixA_mREF, PixA_J=PixA_mSCI, \
                    SFFTConfig=SFFTConfig, SFFTSolution=None, Subtract=True, backend=backend, \
                    CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)[1]

            if trConvdSide == 'SCI':
                PixA_trDIFF = ElementalSFFTSubtract.ESS(PixA_I=PixA_mSCI, PixA_J=PixA_mREF, \
                    SFFTConfig=SFFTConfig, SFFTSolution=None, Subtract=True, backend=backend, \
                    CUDA_DEVICE=CUDA_DEVICE, NUM_CPU_THREADS=NUM_CPU_THREADS)[1]

            # * Estimate expected variance of SubSources on difference
            #   NOTE hypothesis: SubSources are stationary
            Gr = np.array(AstSEx_SS['FLUXERR_AUTO_REF'])
            Gs = np.array(AstSEx_SS['FLUXERR_AUTO_SCI'])
            if trConvdSide == 'REF':
                dm = np.median(AstSEx_SS['MAG_AUTO_SCI'] - AstSEx_SS['MAG_AUTO_REF'])
                VARr = (Gr/(10**(dm/2.5)))**2
                VARs = Gs**2
            if trConvdSide == 'SCI':
                dm = np.median(AstSEx_SS['MAG_AUTO_REF'] - AstSEx_SS['MAG_AUTO_SCI'])
                VARr = Gr**2
                VARs = (Gs/(10**(dm/2.5)))**2
            ExpDVAR_SS = VARr + VARs

            # * Find variables in trial-difference and identify bad SubSources
            SEGL_SS = np.array(AstSEx_SS['SEGLABEL']).astype(int)
            DFSUM_SS = ndimage.labeled_comprehension(PixA_trDIFF, SFFTLmap, SEGL_SS, np.sum, float, 0.0)
            RATIO_SS = DFSUM_SS / np.sqrt(np.clip(ExpDVAR_SS, a_min=0.1, a_max=None))
            SurvMask_TS = np.abs(RATIO_SS) < RATIO_THRESH
            if np.sum(SurvMask_TS) < 0.4 * len(AstSEx_SS):
                print('MeLOn WARNING: SubSource Rejection (%d / %d) by Trial-Subtraction is SKIPPED' \
                    %(np.sum(~SurvMask_TS), len(AstSEx_SS)))
                SurvMask_TS = np.ones(len(AstSEx_SS)).astype(bool)       # NOTE UPDATE
            SEGL_bSS_TS = SEGL_SS[~SurvMask_TS]
            
            # * UPDATE SubSource-Catalog AstSEx_SS [trial-subtraction]
            AstSEx_SS.add_column(Column(ExpDVAR_SS, name='ExpDVAR'))
            AstSEx_SS.add_column(Column(DFSUM_SS, name='DFSUM'))
            AstSEx_SS.add_column(Column(RATIO_SS, name='RATIO'))
            AstSEx_SS.add_column(Column(SurvMask_TS, name='SurvMask_TS'))
            print('MeLOn CheckPoint: trial-difference helps to reject SubSources [%d / %d] !' \
                    %(np.sum(~SurvMask_TS), len(AstSEx_SS)))
        
        SurvMask_PB = None
        if XY_PriorBan is not None:

            SEGL_bSS_PB = np.unique([SFFTLmap[int(_x - 0.5), int(_y - 0.5)] for _x, _y in XY_PriorBan])
            SEGL_bSS_PB = SEGL_bSS_PB[SEGL_bSS_PB > 0] 
            SurvMask_PB = ~np.in1d(SEGL_SS, SEGL_bSS_PB)

            # * UPDATE SubSource-Catalog AstSEx_SS [prior-ban]
            AstSEx_SS.add_column(Column(SurvMask_PB, name='SurvMask_PB'))
            print('MeLOn CheckPoint: prior-ban helps to reject SubSources [%d / %d] !' \
                    %(np.sum(~SurvMask_PB), len(AstSEx_SS)))

        # * Post Update Operations
        SurvMask = None
        if SurvMask_TS is not None and SurvMask_PB is not None:
            SurvMask = np.logical_and(SurvMask_TS, SurvMask_PB)
        if SurvMask_TS is not None and SurvMask_PB is None:
            SurvMask = SurvMask_TS.copy()
        if SurvMask_TS is None and SurvMask_PB is not None:
            SurvMask = SurvMask_PB.copy()
        
        # * UPDATE SFFTLmap > ActiveMask > PixA_mREF & PixA_mSCI
        if SurvMask is not None:
            _mappings = {}
            SEGL_bSS = SEGL_SS[~SurvMask]
            for label in SEGL_bSS: _mappings[label] = -64
            fastremap.remap(SFFTLmap, _mappings, preserve_missing_labels=True, in_place=True)   # NOTE UPDATE
            _mask = SFFTLmap == -64
            
            ActiveMask[_mask] = False      #  NOTE UPDATE, always be SFFTLmap > 0
            PixA_mREF[_mask] = 0.0         #  NOTE UPDATE
            PixA_mSCI[_mask] = 0.0         #  NOTE UPDATE
            AstSEx_SS.add_column(Column(SurvMask, name='SurvMask'))
            print('MeLOn CheckPoint: Total SubSource Rejections [%d / %d] !' \
                    %(np.sum(~SurvMask), len(AstSEx_SS)))
        
        # NOTE: The preparation can guarantee that mREF & mSCI are NaN-Free !
        ActivePROP = np.sum(ActiveMask) / (ActiveMask.shape[0] * ActiveMask.shape[1])
        print('MeLOn CheckPoint: After-Refinement ActiveMask Pixel Proportion [%s]' %('{:.2%}'.format(ActivePROP)))
    
        return SFFTPrepDict
