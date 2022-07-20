import warnings
import fastremap
import numpy as np
from astropy.io import fits
import scipy.ndimage as ndimage
from astropy.table import Column, hstack
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.SymmetricMatch import Symmetric_Match
from sfft.utils.HoughMorphClassifier import Hough_MorphClassifier
# version: Jul 20, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.2"

class Auto_SparsePrep:
    def __init__(self, FITS_REF, FITS_SCI, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        BACK_TYPE='MANUAL', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, \
        DETECT_THRESH=2.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, \
        BACKPHOTO_TYPE='LOCAL', ONLY_FLAGS=[0], BoundarySIZE=30):

        self.FITS_REF = FITS_REF
        self.FITS_SCI = FITS_SCI

        self.GAIN_KEY = GAIN_KEY
        self.SATUR_KEY = SATUR_KEY

        self.BACK_TYPE = BACK_TYPE
        self.BACK_VALUE = BACK_VALUE
        self.BACK_SIZE = BACK_SIZE
        self.BACK_FILTERSIZE = BACK_FILTERSIZE

        self.DETECT_THRESH = DETECT_THRESH
        self.DETECT_MINAREA = DETECT_MINAREA
        self.DETECT_MAXAREA = DETECT_MAXAREA
        self.DEBLEND_MINCONT = DEBLEND_MINCONT
        self.BACKPHOTO_TYPE = BACKPHOTO_TYPE

        self.ONLY_FLAGS = ONLY_FLAGS
        self.BoundarySIZE = BoundarySIZE

    def run_image_mask(self, AstSEx_SS, PixA_SEGr, PixA_SEGs, StarExt_iter, XY_PriorBan):
        
        PixA_REF = fits.getdata(self.FITS_REF, ext=0).T
        if np.issubdtype(PixA_REF.dtype, np.integer):
            PixA_REF = PixA_REF.astype(np.float64)
        
        PixA_SCI = fits.getdata(self.FITS_SCI, ext=0).T
        if np.issubdtype(PixA_SCI.dtype, np.integer):
            PixA_SCI = PixA_SCI.astype(np.float64)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            SATLEVEL_REF = fits.getheader(self.FITS_REF, ext=0)[self.SATUR_KEY]
            SATLEVEL_SCI = fits.getheader(self.FITS_SCI, ext=0)[self.SATUR_KEY]
            SatMask_REF = PixA_REF >= SATLEVEL_REF
            SatMask_SCI = PixA_SCI >= SATLEVEL_SCI

        # * Relabel SS-islands & Flip labels of NotSS-islands
        #   WARNING NOTE: PixA_SEGr & PixA_SEGs will be updated!
        SEGL_SSr = np.array(AstSEx_SS['SEGLABEL_REF']).astype(int)
        SEGL_SSs = np.array(AstSEx_SS['SEGLABEL_SCI']).astype(int)
        SEGL_SS = np.array(AstSEx_SS['SEGLABEL']).astype(int)

        Mappings_REF = {}
        for label_o, label_n in zip(SEGL_SSr, SEGL_SS): 
            Mappings_REF[label_o] = -label_n
        fastremap.remap(PixA_SEGr, Mappings_REF, preserve_missing_labels=True, in_place=True)
        PixA_SEGr *= -1

        Mappings_SCI = {}
        for label_o, label_n in zip(SEGL_SSs, SEGL_SS):
            Mappings_SCI[label_o] = -label_n
        fastremap.remap(PixA_SEGs, Mappings_SCI, preserve_missing_labels=True, in_place=True)
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
        SFFTLmap[ProZone] = 0     # NOTE an after-burn equivalent operation
        struct0 = ndimage.generate_binary_structure(2, 1)
        struct = ndimage.iterate_structure(struct0, StarExt_iter)
        SFFTLmap = ndimage.grey_dilation(SFFTLmap, footprint=struct)
        SFFTLmap[ProZone] = -128

        if XY_PriorBan is not None:
            # ** Identify the Prior-Banned SubSources and decorate AstSEx_SS
            #    NOTE: Unlike Prior-Selection, the Prior-Ban coordinates do not have to be accurate.
            #    WARNING NOTE: AstSEx_SS will be updated in this step!

            SEGL_PB = np.unique([SFFTLmap[int(_x - 0.5), int(_y - 0.5)] for _x, _y in XY_PriorBan])
            SEGL_PB = SEGL_PB[SEGL_PB > 0] 
            PBMASK_SS = np.in1d(SEGL_SS, SEGL_PB)
            AstSEx_SS.add_column(Column(PBMASK_SS, name='MASK_PriorBan'))
            print('MeLOn CheckPoint: Find / Given [%d / %d] Prior-Banned in Current [%d] SubSources!' \
                %(np.sum(PBMASK_SS), XY_PriorBan.shape[0], len(AstSEx_SS)))

            # ** Update Label Map (SFFTLmap) by inactivating the Prior-Banned SubSources
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

        SFFTPrepDict['SExCatalog-SubSource'] = AstSEx_SS
        SFFTPrepDict['SFFT-LabelMap'] = SFFTLmap
        SFFTPrepDict['Active-Mask'] = ActiveMask
        SFFTPrepDict['PixA_mREF'] = PixA_mREF
        SFFTPrepDict['PixA_mSCI'] = PixA_mSCI

        return SFFTPrepDict

    def HoughAutoMask(self, Hough_FRLowerLimit=0.1, BeltHW=0.2, PS_ELLIPThresh=0.3, \
        MatchTol=None, MatchTolFactor=3.0, MAGD_THRESH=0.12, StarExt_iter=4, XY_PriorBan=None):
        # - Hough Transform is used to determine subtraction-sources

        # * Use Hough-MorphClassifer to identify GoodSources in REF & SCI
        #   NOTE: Although Sparse-Flavor-SFFT assume the processed images being sky-subtracted, it does not means
        #         that we should always preserve zero-background (i.e., BACK_TYPE='MANUAL' & BACK_VALUE='0.0').
        #         e.g., if you want to detect point sources in bright diffuse light (galaxy outskirts), then using 
        #               BACK_TYPE='AUTO' becomes necessary even the unmasked region will finally be zero-filled as well.
        
        def main_hough(FITS_obj):
            # NOTE: SEGMENTATION map is sensitive to DETECT_THRESH (here we use a cold default value of 2.0)
            Hmc = Hough_MorphClassifier.MakeCatalog(FITS_obj=FITS_obj, GAIN_KEY=self.GAIN_KEY, \
                SATUR_KEY=self.SATUR_KEY, BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, BACK_SIZE=self.BACK_SIZE, \
                BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, DETECT_MINAREA=self.DETECT_MINAREA, \
                DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, \
                CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, ONLY_FLAGS=self.ONLY_FLAGS, \
                BoundarySIZE=self.BoundarySIZE, AddSNR=False)
            Hc = Hough_MorphClassifier.Classifier(AstSEx=Hmc[0], Hough_FRLowerLimit=Hough_FRLowerLimit, \
                BeltHW=BeltHW, PS_ELLIPThresh=PS_ELLIPThresh, Return_HPS=False)
            AstSEx_GS = Hmc[0][Hc[2]]
            FHWM = Hc[0]
            PixA_SEG = Hmc[1][0].astype(int)
            return AstSEx_GS, FHWM, PixA_SEG

        AstSEx_GSr, FWHM_REF, PixA_SEGr = main_hough(self.FITS_REF)
        AstSEx_GSs, FWHM_SCI, PixA_SEGs = main_hough(self.FITS_SCI)
        print('MeLOn CheckPoint: Hough FWHM Estimate [FWHM_REF = %.3f pix] & [FWHM_SCI = %.3f pix]!' %(FWHM_REF, FWHM_SCI))

        XY_GSr = np.array([AstSEx_GSr['X_IMAGE'], AstSEx_GSr['Y_IMAGE']]).T
        XY_GSs = np.array([AstSEx_GSs['X_IMAGE'], AstSEx_GSs['Y_IMAGE']]).T

        UMatchTol = MatchTol
        if MatchTol is None:
            # - Given precise WCS, one can use a high MatchTolFactor ~3.0
            # - For very sparse fields where WCS is inaccurate, one can loosen MatchTolFactor to ~1.0
            # - WARNING NOTE: it can go wrong when the estimated FWHM is highly inaccurate.

            UMatchTol = np.sqrt((FWHM_REF / MatchTolFactor)**2 + (FWHM_SCI / MatchTolFactor)**2)
        print('MeLOn CheckPoint: Used Matching Tolerance [tol = %.3f pix]!' %UMatchTol)

        # * Determine Matched-GoodSources [MGS]
        Symm = Symmetric_Match.SM(POA=XY_GSr, POB=XY_GSs, tol=UMatchTol)
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

        # create unified segmentation label
        AstSEx_SS.add_column(Column(1+np.arange(len(AstSEx_SS)), name='SEGLABEL'), index=0)
        print('\nMeLOn CheckPoint: Number of SubSources out of Matched-GoodSources [%d / %d]!' \
                %(len(AstSEx_SS), len(MAGD)))

        # * Run Image Masking
        SFFTPrepDict = self.run_image_mask(AstSEx_SS=AstSEx_SS, PixA_SEGr=PixA_SEGr, \
            PixA_SEGs=PixA_SEGs, StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan)
        
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        return SFFTPrepDict

    def SemiAutoMask(self, XY_PriorSelect=None, BeltHW=0.2, PS_ELLIPThresh=0.3, \
        MatchTol=None, MatchTolFactor=3.0, StarExt_iter=4, XY_PriorBan=None):
        # - We directly use given Prior-Selection to determine subtraction-sources

        def main_phot(FITS_obj):
            PL = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', \
                  'FLAGS', 'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE']
            PYSEX_OP = PY_SEx.PS(FITS_obj=FITS_obj, PL=PL, GAIN_KEY=self.GAIN_KEY, SATUR_KEY=self.SATUR_KEY, \
                BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, BACK_SIZE=self.BACK_SIZE, \
                BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, DETECT_MINAREA=self.DETECT_MINAREA, \
                DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, \
                CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, ONLY_FLAGS=self.ONLY_FLAGS, \
                XBoundary=self.BoundarySIZE, YBoundary=self.BoundarySIZE, MDIR=None)
            AstSEx = PYSEX_OP[0]
            PixA_SEG = PYSEX_OP[1][0].astype(int)

            def crude_fwhm_estimate(AstSEx, PS_ELLIPThresh, BeltHW):
                # this is just the same as in HoughMorphClassifier (when Hough fails)
                A_IMAGE = np.array(AstSEx['A_IMAGE'])
                B_IMAGE = np.array(AstSEx['B_IMAGE'])
                MA_FR = np.array([AstSEx['MAG_AUTO'], AstSEx['FLUX_RADIUS']]).T
                ELLIP = (A_IMAGE - B_IMAGE)/(A_IMAGE + B_IMAGE)
                MASK_ELLIP = ELLIP < PS_ELLIPThresh
                BPmask = AstSEx['MAGERR_AUTO'] < np.percentile(AstSEx['MAGERR_AUTO'], 75)
                Rv = np.percentile(MA_FR[BPmask, 1], 25)
                MASK_FRM = np.abs(MA_FR[:, 1]  - Rv) < BeltHW
                MASK_PS = np.logical_and(MASK_FRM, MASK_ELLIP)
                FWHM = round(np.median(AstSEx[MASK_PS]['FWHM_IMAGE']), 6)
                return FWHM
            FWHM = crude_fwhm_estimate(AstSEx, PS_ELLIPThresh, BeltHW)
            return AstSEx, FWHM, PixA_SEG
        
        AstSExr, FWHM_REF, PixA_SEGr = main_phot(self.FITS_REF)
        AstSExs, FWHM_SCI, PixA_SEGs = main_phot(self.FITS_SCI)
        print('MeLOn CheckPoint: Crude FWHM Estimate [FWHM_REF = %.3f pix] & [FWHM_SCI = %.3f pix]!' %(FWHM_REF, FWHM_SCI))

        XYr = np.array([AstSExr['X_IMAGE'], AstSExr['Y_IMAGE']]).T
        XYs = np.array([AstSExs['X_IMAGE'], AstSExs['Y_IMAGE']]).T

        UMatchTol = MatchTol
        if MatchTol is None:
            # - Given precise WCS, one can use a high MatchTolFactor ~3.0
            # - For very sparse fields where WCS is inaccurate, one can loosen MatchTolFactor to ~1.0
            # - WARNING NOTE: it can go wrong when the estimated FWHM is highly inaccurate.

            UMatchTol = np.sqrt((FWHM_REF / MatchTolFactor)**2 + (FWHM_SCI / MatchTolFactor)**2)
        print('MeLOn CheckPoint: Used Matching Tolerance [tol = %.3f pix]!' %UMatchTol)

        # * Cross Match REF & SCI catalogs and Combine them
        _Symm = Symmetric_Match.SM(POA=XYr, POB=XYs, tol=UMatchTol)
        AstSEx_Mr = AstSExr[_Symm[:, 0]]
        AstSEx_Ms = AstSExs[_Symm[:, 1]]

        for coln in AstSEx_Mr.colnames:
            AstSEx_Mr[coln].name = coln + '_REF'
            AstSEx_Ms[coln].name = coln + '_SCI'
        AstSEx_iSS = hstack([AstSEx_Mr, AstSEx_Ms])

        X_IMAGE_REF_SCI_MEAN = np.array(AstSEx_iSS['X_IMAGE_REF']/2.0 + AstSEx_iSS['X_IMAGE_SCI']/2.0)
        Y_IMAGE_REF_SCI_MEAN = np.array(AstSEx_iSS['Y_IMAGE_REF']/2.0 + AstSEx_iSS['Y_IMAGE_SCI']/2.0)
        AstSEx_iSS.add_column(Column(X_IMAGE_REF_SCI_MEAN, name='X_IMAGE_REF_SCI_MEAN'))
        AstSEx_iSS.add_column(Column(Y_IMAGE_REF_SCI_MEAN, name='Y_IMAGE_REF_SCI_MEAN'))

        # * Only Preserve the Objects in the Prior-Selection 
        #   WARNING NOTE: here we still use the same tolerance!
        #   WARNING NOTE: we match the prior selection to the mean image coordinates

        XY_iSS = np.array([AstSEx_iSS['X_IMAGE_REF_SCI_MEAN'], AstSEx_iSS['Y_IMAGE_REF_SCI_MEAN']]).T
        _Symm = Symmetric_Match.SM(POA=XY_PriorSelect, POB=XY_iSS, tol=UMatchTol)
        AstSEx_SS = AstSEx_iSS[_Symm[:, 1]]
        
        # record matching and create unified segmentation label
        AstSEx_SS.add_column(Column(_Symm[:, 0], name='INDEX_PRIOR_SELECTION'), index=0)
        AstSEx_SS.add_column(Column(1+np.arange(len(AstSEx_SS)), name='SEGLABEL'), index=0)
        print('MeLOn CheckPoint: Find / Given [%d / %d] Prior-Selected in [%d] Matched-Sources!' \
               %(len(AstSEx_SS), XY_PriorSelect.shape[0], len(AstSEx_iSS)))
        
        # * Run Image Masking
        SFFTPrepDict = self.run_image_mask(AstSEx_SS=AstSEx_SS, PixA_SEGr=PixA_SEGr, \
            PixA_SEGs=PixA_SEGs, StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan)
        
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        return SFFTPrepDict
