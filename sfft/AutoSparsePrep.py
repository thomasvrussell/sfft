import warnings
import fastremap
import numpy as np
from astropy.io import fits
import scipy.ndimage as ndimage
from astropy.table import Column, hstack
from sfft.utils.SymmetricMatch import Symmetric_Match
from sfft.utils.HoughMorphClassifier import Hough_MorphClassifier

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.1"

class Auto_SparsePrep:
    def __init__(self, FITS_REF, FITS_SCI):

        self.FITS_REF = FITS_REF
        self.FITS_SCI = FITS_SCI
    
    def AutoMask(self, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', DETECT_THRESH=2.0, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, BoundarySIZE=30, \
        Hough_FRLowerLimit=0.1, BeltHW=0.2, MatchTolFactor=3.0, \
        MAGD_THRESH=0.12, StarExt_iter=4, XY_PriorBan=None):
        
        # ********************* Determine SubSources ********************* #

        # * Use Hough-MorphClassifer to identify GoodSources in REF & SCI
        def main_hough(FITS_obj):
            # NOTE Byproducts: SATLEVEL & FWHM Estimation & SEGMENTATION map
            # NOTE DETECT_THRESH can affect SEGMENTATION map
            Hmc = Hough_MorphClassifier.MakeCatalog(FITS_obj=FITS_obj, GAIN_KEY=GAIN_KEY, \
                SATUR_KEY=SATUR_KEY, BACK_TYPE='MANUAL', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, \
                DETECT_THRESH=DETECT_THRESH, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, \
                BACKPHOTO_TYPE='LOCAL', CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, \
                BoundarySIZE=BoundarySIZE, AddSNR=False)          # FIXME SEx-Configuration is Customizable

            Hc = Hough_MorphClassifier.Classifier(AstSEx=Hmc[0], \
                Hough_FRLowerLimit=Hough_FRLowerLimit, BeltHW=BeltHW, Return_HPS=False)
            
            AstSEx_GS = Hmc[0][Hc[2]]
            SATLEVEL = fits.getheader(FITS_obj, ext=0)[SATUR_KEY]
            FHWM, PixA_SEG = Hc[0], Hmc[1][0].astype(int)
            return AstSEx_GS, SATLEVEL, FHWM, PixA_SEG

        AstSEx_GSr, SATLEVEL_REF, FWHM_REF, PixA_SEGr = main_hough(self.FITS_REF)
        AstSEx_GSs, SATLEVEL_SCI, FWHM_SCI, PixA_SEGs = main_hough(self.FITS_SCI)
        XY_GSr = np.array([AstSEx_GSr['X_IMAGE'], AstSEx_GSr['Y_IMAGE']]).T
        XY_GSs = np.array([AstSEx_GSs['X_IMAGE'], AstSEx_GSs['Y_IMAGE']]).T

        # * Determine Matched-GoodSources [MGS]
        #   @ Given precise WCS, one can use a high MatchTolFactor ~3.0
        #   @ For very sparse fields where WCS can be inaccurate, 
        #     one can loosen the tolerance with a low MatchTolFactor ~1.0
        
        tol = np.sqrt((FWHM_REF/MatchTolFactor)**2 + (FWHM_SCI/MatchTolFactor)**2)
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
        print('\nMeLOn CheckPoint: Number of SubSources out of Matched-GoodSources [%d / %d] !' \
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

        # * Make SFFTLmap 
        SFFTLmap = np.max(np.array([PixA_SEGr, PixA_SEGs]), axis=0)
        SFFTLmap[ProZone] = 0             # NOTE After-burn equivalent operation
        struct0 = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct0, StarExt_iter)
        SFFTLmap = ndimage.grey_dilation(SFFTLmap, footprint=struct)
        SFFTLmap[ProZone] = -128

        if XY_PriorBan is not None:
            # ** Identify the PriorBanned SubSources and decorate AstSEx_SS
            SEGL_PB = np.unique([SFFTLmap[int(_x - 0.5), int(_y - 0.5)] for _x, _y in XY_PriorBan])
            SEGL_PB = SEGL_PB[SEGL_PB > 0] 
            PBMASK_SS = np.in1d(SEGL_SS, SEGL_PB)
            AstSEx_SS.add_column(Column(PBMASK_SS, name='MASK_PriorBan'))

            print('MeLOn CheckPoint: Find [%d] PriorBanned / Not-Working SubSources out of [%d] Auto-Determined SubSources!\n' \
                  %(np.sum(PBMASK_SS), len(AstSEx_SS)) + \
                  'MeLOn CheckPoint: There are [%d] given PriorBan coordinates!' %(XY_PriorBan.shape[0]))
            
            # ** Update Label Map (SFFTLmap) by inactivating the PriorBanned SubSources
            _mappings = {label: -64 for label in SEGL_SS[PBMASK_SS]}
            fastremap.remap(SFFTLmap, _mappings, preserve_missing_labels=True, in_place=True)   

        # * Create ActiveMask
        #   NOTE: The preparation can guarantee that mREF & mSCI are NaN-Free !
        ActiveMask = SFFTLmap > 0
        ActivePROP = np.sum(ActiveMask) / (ActiveMask.shape[0] * ActiveMask.shape[1])
        print('MeLOn CheckPoint: ActiveMask Pixel Proportion [%s]' %('{:.2%}'.format(ActivePROP)))

        # * Produce masked images PixA_mREF & PixA_mSCI
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
    