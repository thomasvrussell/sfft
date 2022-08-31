import warnings
import fastremap
import numpy as np
from astropy.io import fits
import scipy.ndimage as ndimage
from astropy.table import Column, hstack
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.SymmetricMatch import Symmetric_Match
from sfft.utils.HoughMorphClassifier import Hough_MorphClassifier
from sfft.utils.WeightedQuantile import TopFlatten_Weighted_Quantile
# version: Aug 31, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.3"

class Auto_SparsePrep:
    def __init__(self, FITS_REF, FITS_SCI, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        BACK_TYPE='MANUAL', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, \
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
            _message = 'Find / Given [%d / %d] Prior-Banned ' %(np.sum(PBMASK_SS), XY_PriorBan.shape[0])
            _message += 'in current [%d] SubSources!' %(len(AstSEx_SS))
            print('MeLOn CheckPoint: %s' %_message)

            # ** Update Label Map (SFFTLmap) by inactivating the Prior-Banned SubSources
            _mappings = {label: -64 for label in SEGL_SS[PBMASK_SS]}
            fastremap.remap(SFFTLmap, _mappings, preserve_missing_labels=True, in_place=True)
        else:
            _message = 'Find / Given [0 / 0] Prior-Banned '
            _message += 'in current [%d] SubSources!' %(len(AstSEx_SS))
            print('MeLOn CheckPoint: %s' %_message)
        
        # * Create ActiveMask
        #   NOTE: The preparation can guarantee that mREF & mSCI are NaN-Free !
        ActiveMask = SFFTLmap > 0
        ActivePROP = np.sum(ActiveMask) / (ActiveMask.shape[0] * ActiveMask.shape[1])
        print('MeLOn CheckPoint: Active-Mask (Non-Prior-Banned SubSources) Pixel Proportion [%s]' \
               %('{:.2%}'.format(ActivePROP)))

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

    def HoughAutoMask(self, Hough_FRLowerLimit=0.1, Hough_peak_clip=0.7, BeltHW=0.2, PS_ELLIPThresh=0.3, \
        MatchTol=None, MatchTolFactor=3.0, COARSE_VAR_REJECTION=True, CVREJ_MAGD_THRESH=0.12, \
        ELABO_VAR_REJECTION=False, EVREJ_RATIO_THREH=5.0, EVREJ_SAFE_MAGDEV=0.04, StarExt_iter=4, XY_PriorBan=None):
        # - Hough Transform is used to determine subtraction-sources

        # * Use Hough-MorphClassifer to identify GoodSources in REF & SCI
        def main_hough(FITS_obj):
            # NOTE: SEGMENTATION map is sensitive to DETECT_THRESH
            Hmc = Hough_MorphClassifier.MakeCatalog(FITS_obj=FITS_obj, GAIN_KEY=self.GAIN_KEY, \
                SATUR_KEY=self.SATUR_KEY, BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, BACK_SIZE=self.BACK_SIZE, \
                BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, DETECT_MINAREA=self.DETECT_MINAREA, \
                DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, \
                CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, ONLY_FLAGS=self.ONLY_FLAGS, \
                BoundarySIZE=self.BoundarySIZE, AddSNR=False)

            Hc = Hough_MorphClassifier.Classifier(AstSEx=Hmc[0], Hough_FRLowerLimit=Hough_FRLowerLimit, \
                Hough_peak_clip=Hough_peak_clip, BeltHW=BeltHW, PS_ELLIPThresh=PS_ELLIPThresh, Return_HPS=False)
            AstSEx_GS, FHWM, PixA_SEG = Hmc[0][Hc[2]], Hc[0], Hmc[1][0].astype(int)
            return AstSEx_GS, FHWM, PixA_SEG

        AstSEx_GSr, FWHM_REF, PixA_SEGr = main_hough(self.FITS_REF)
        AstSEx_GSs, FWHM_SCI, PixA_SEGs = main_hough(self.FITS_SCI)
        _message = 'Estimated [FWHM_REF = %.3f pix] & [FWHM_SCI = %.3f pix]!' %(FWHM_REF, FWHM_SCI)
        print('\nMeLOn CheckPoint: %s' %_message)

        # * Determine Matched-GoodSources (abbr. MGS)
        XY_GSr = np.array([AstSEx_GSr['X_IMAGE'], AstSEx_GSr['Y_IMAGE']]).T
        XY_GSs = np.array([AstSEx_GSs['X_IMAGE'], AstSEx_GSs['Y_IMAGE']]).T
        
        UMatchTol = MatchTol
        if MatchTol is None:
            # - Given precise WCS, one can use a high MatchTolFactor ~ 3.0
            # - For very sparse fields where WCS is inaccurate, one can loosen MatchTolFactor to ~ 1.0
            # - WARNING NOTE: it can go wrong when the estimated FWHM is highly inaccurate.
            UMatchTol = np.sqrt((FWHM_REF / MatchTolFactor)**2 + (FWHM_SCI / MatchTolFactor)**2)
        print('MeLOn CheckPoint: Tolerance [%.3f pix] For Source Cross-Match!' %UMatchTol)

        Symm = Symmetric_Match.SM(XY_A=XY_GSr, XY_B=XY_GSs, tol=UMatchTol, return_distance=False)
        AstSEx_MGSr = AstSEx_GSr[Symm[:, 0]]
        AstSEx_MGSs = AstSEx_GSs[Symm[:, 1]]
        NUM_MGS = Symm.shape[0]

        # * Determine Magnitude Offset from MGS
        MAG_MGSr = np.array(AstSEx_MGSr['MAG_AUTO'])
        MAG_MGSs = np.array(AstSEx_MGSs['MAG_AUTO'])
        FLUX_MGSr = np.array(AstSEx_MGSr['FLUX_AUTO'])
        FLUX_MGSs = np.array(AstSEx_MGSs['FLUX_AUTO'])

        MAGD_MGS = MAG_MGSs - MAG_MGSr
        MAG_OFFSET0 = np.median(MAGD_MGS)

        NTE_4MO = 30  # NUM_TOP_END for MAG_OFFSET, default value used herein.
        MAG_OFFSETr = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_MGS, weights=FLUX_MGSr, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        MAG_OFFSETs = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_MGS, weights=FLUX_MGSs, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        MAG_OFFSET = (MAG_OFFSETr + MAG_OFFSETs)/2.0
        
        if np.abs(MAG_OFFSET - MAG_OFFSET0) > 0.05:
            _warn_message = 'Magnitude Offset Estimate --- WEIGHTED-MEDIAN IS HIGHLY DEVIATED FROM MEDIAN '
            _warn_message += '[median: %.3f mag] >>> [weighted-median: %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
        else:
            _message = 'Magnitude Offset Estimate --- '
            _message += '[median: %.3f mag] >>> [weighted-median: %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
            print('\nMeLOn CheckPoint: %s' %_message)

        # * Apply a coarse variable rejection (abbr. CVREJ)
        if COARSE_VAR_REJECTION:
            CVREJ_MASK = np.abs(MAGD_MGS - MAG_OFFSET) > CVREJ_MAGD_THRESH
            AstSEx_iSSr = AstSEx_MGSr[~CVREJ_MASK]
            AstSEx_iSSs = AstSEx_MGSs[~CVREJ_MASK]
            _message = 'Coarse Variable Rejection [magnitude deviation > %.3f mag] ' %CVREJ_MAGD_THRESH
            _message += 'on Matched-GoodSources [%d / %d]!' %(np.sum(CVREJ_MASK), NUM_MGS)
            print('\nMeLOn CheckPoint: %s' %_message)
        else:
            AstSEx_iSSr = AstSEx_MGSr
            AstSEx_iSSs = AstSEx_MGSs
            print('\nMeLOn CheckPoint: SKIP Coarse Variable Rejection!')

        # * Apply a more elaborate variable rejection (abbr. EVREJ)
        if ELABO_VAR_REJECTION:
            warnings.warn('\nMeLOn WARNING: Elaborate Variable Rejection requires CORRECT GAIN in FITS HEADER!')
            # TODO: Incorporate weight-maps of the input image pair for more accurate SExtractor FLUXERR.

            """
            # Remarks on the Elaborate Variable Rejection
            # [1] The coarse variable rejection by CVREJ_MAGD_THRESH can only exclude the most drastic variable sources.
            #     As a result, the possible missing varaibles, especially at the bright end, can strongly affect the subtraction. 
            #
            # [2] Elaborate Variable Rejection (EVREJ) is designed to further identify variables by taking their 
            #     photometric uncertainties into account. For stationary objects, the flux change between two images 
            #     is a predictable probability distribution propagated from original photometric uncertatinties.
            #     EVREJ identify the outliers of the distribution as a variable. 
            #     NOTE: Bright transients (compared with its host) can be also identified by EVREJ.
            #
            # [3] However, the scenario in [2] is very ideal so that EVREJ can overkill. This motivated us to add a
            #     protection on EVREJ by -EVREJ_SAFE_MAGDEV, which is used to handle the imperfections listed below.
            #
            #     (a) distribution mean: The photometric scaling can vary across the field (says, ~1%, or ~0.01mag). 
            #         However, we only use a constant MAG_OFFSET for EVREJ. The error of MAG_OFFSET, including the ignorance 
            #         of spatial variation and the uncertainty of the constant MAG_OFFSET itself, makes the calculated flux 
            #         change only an approximation of the 'real' value.
            #
            #     (b) distribution variance: i) inaccurate SExtractor FLUXERR, especially at the bright end 
            #         (Poission Noise Dominated). e.g., A mosaic input image used an average GAIN value, 
            #         making FLUXERR calculation in SExtractor not accurate. ii) the error of MAG_OFFSET 
            #         will affect the scaling of one FLUXERR.
            #
            """

            MAG_iSSr = np.array(AstSEx_iSSr['MAG_AUTO'])
            MAG_iSSs = np.array(AstSEx_iSSs['MAG_AUTO'])
            FLUX_iSSr = np.array(AstSEx_iSSr['FLUX_AUTO'])
            FLUX_iSSs = np.array(AstSEx_iSSs['FLUX_AUTO'])
            FLUXERR_iSSr = np.array(AstSEx_iSSr['FLUXERR_AUTO'])
            FLUXERR_iSSs = np.array(AstSEx_iSSs['FLUXERR_AUTO'])
            MAGD_iSS = MAG_iSSs - MAG_iSSr

            # *** Note on the flux scaling *** 
            # (1) MAG_OFFSET = MAG_SCI - MAG_REF = -2.5 * np.log10(FLUX_SCI/FLUX_REF)
            # (2) FLUX_SCAL = FLUX_SCI/FLUX_REF = 10**(MAG_OFFSET/-2.5)

            FLUX_SCAL = 10**(MAG_OFFSET/-2.5)
            sFLUX_iSSr = FLUX_SCAL * FLUX_iSSr
            sFLUXERR_iSSr = FLUX_SCAL * FLUXERR_iSSr

            _DATA = FLUX_iSSs - sFLUX_iSSr
            _SIGMA = np.sqrt(sFLUXERR_iSSr**2 + FLUXERR_iSSs**2)
            _OUTMASK = np.abs(_DATA) > EVREJ_RATIO_THREH * _SIGMA
            _SAFEMASK = np.abs(MAGD_iSS - MAG_OFFSET) <= EVREJ_SAFE_MAGDEV
            
            EVREJ_MASK = np.logical_and(_OUTMASK, ~_SAFEMASK)
            AstSEx_SSr = AstSEx_iSSr[~EVREJ_MASK]
            AstSEx_SSs = AstSEx_iSSs[~EVREJ_MASK]

            EVREJ_PERC = np.sum(EVREJ_MASK) / NUM_MGS
            _message = 'Elaborate Variable Rejection [flux deviation > %.2f sigma] & ' %EVREJ_RATIO_THREH
            _message += '[magnitude deviation > %.3f mag] ' %EVREJ_SAFE_MAGDEV
            _message += 'on Matched-GoodSources [%d / %d] ' %(np.sum(EVREJ_MASK), NUM_MGS)
            _message += 'or [%s]!' %('{:.2%}'.format(EVREJ_PERC))
            print('\nMeLOn CheckPoint: %s' %_message)

            if EVREJ_PERC > 0.1:
                _warn_message = '[%s] Matched-GoodSources ARE REJECTED BY EVREJ!' %('{:.2%}'.format(EVREJ_PERC))
                warnings.warn('\nMeLOn IMPORTANT WARNING: %s' %_warn_message)
        else:
            AstSEx_SSr = AstSEx_iSSr
            AstSEx_SSs = AstSEx_iSSs
            print('\nMeLOn CheckPoint: SKIP Elaborate Variable Rejection!')

        # * Combine catalogs for survived MGS (i.e., SubSources)
        for coln in AstSEx_SSr.colnames:
            AstSEx_SSr[coln].name = coln + '_REF'
            AstSEx_SSs[coln].name = coln + '_SCI'
        AstSEx_SS = hstack([AstSEx_SSr, AstSEx_SSs])
        
        # create unified segmentation label
        AstSEx_SS.add_column(Column(1+np.arange(len(AstSEx_SS)), name='SEGLABEL'), index=0)
        print('\nMeLOn CheckPoint: SubSources out of Matched-GoodSources [%d / %d]!' %(len(AstSEx_SS), NUM_MGS))
        
        # * Run Image Masking (P.S. some SubSources might be marked as invalid by the Prior-Ban list)
        SFFTPrepDict = self.run_image_mask(AstSEx_SS=AstSEx_SS, PixA_SEGr=PixA_SEGr, \
            PixA_SEGs=PixA_SEGs, StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan)
        
        # * Append Additional information
        SFFTPrepDict['MAG_OFFSET'] = MAG_OFFSET
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        return SFFTPrepDict

    def SemiAutoMask(self, XY_PriorSelect=None, MatchTol=None, MatchTolFactor=3.0, StarExt_iter=4, XY_PriorBan=None):
        # - We directly use given Prior-Selection to determine subtraction-sources

        def func4phot(FITS_obj):
            # run SExtractor photometry
            PL = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', \
                  'FLAGS', 'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE']
            PYSEX_OP = PY_SEx.PS(FITS_obj=FITS_obj, PL=PL, GAIN_KEY=self.GAIN_KEY, SATUR_KEY=self.SATUR_KEY, \
                BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, BACK_SIZE=self.BACK_SIZE, \
                BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, DETECT_MINAREA=self.DETECT_MINAREA, \
                DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, \
                CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, ONLY_FLAGS=self.ONLY_FLAGS, \
                XBoundary=self.BoundarySIZE, YBoundary=self.BoundarySIZE, MDIR=None)
            AstSEx, PixA_SEG = PYSEX_OP[0], PYSEX_OP[1][0].astype(int)
            
            # *** Note on the crude estimate of FWHM ***
            # our strategy is to calculate a median FWHM_IMAGE weighted by the source flux, meanwhile, 
            # we modulate the weights by applying a penalty on the sources with large FWHM_IMAGE.

            NTE_4FWHM = 30  # NUM_TOP_END for FWHM, default value used herein.
            _values, _weights = np.array(AstSEx['FWHM_IMAGE']), np.array(AstSEx['FLUX_AUTO'])
            _weights /= np.clip(_values, a_min=1.0, a_max=None)**2  
            FWHM = TopFlatten_Weighted_Quantile.TFWQ(values=_values, weights=_weights, \
                quantiles=[0.5], NUM_TOP_END=NTE_4FWHM)[0]
            return AstSEx, FWHM, PixA_SEG
        
        AstSExr, FWHM_REF, PixA_SEGr = func4phot(self.FITS_REF)
        AstSExs, FWHM_SCI, PixA_SEGs = func4phot(self.FITS_SCI)
        _message = 'Estimated [FWHM_REF = %.3f pix] & [FWHM_SCI = %.3f pix]!' %(FWHM_REF, FWHM_SCI)
        print('\nMeLOn CheckPoint: %s' %_message)

        # * Cross Match REF & SCI catalogs
        XYr = np.array([AstSExr['X_IMAGE'], AstSExr['Y_IMAGE']]).T
        XYs = np.array([AstSExs['X_IMAGE'], AstSExs['Y_IMAGE']]).T

        UMatchTol = MatchTol
        if MatchTol is None:
            # - Given precise WCS, one can use a high MatchTolFactor ~ 3.0
            # - For very sparse fields where WCS is inaccurate, one can loosen MatchTolFactor to ~ 1.0
            # - WARNING NOTE: it can go wrong when the estimated FWHM is highly inaccurate.
            UMatchTol = np.sqrt((FWHM_REF / MatchTolFactor)**2 + (FWHM_SCI / MatchTolFactor)**2)
        print('MeLOn CheckPoint: Tolerance [%.3f pix] For Source Cross-Match!' %UMatchTol)
        _Symm = Symmetric_Match.SM(XY_A=XYr, XY_B=XYs, tol=UMatchTol, return_distance=False)
        AstSEx_Mr = AstSExr[_Symm[:, 0]]
        AstSEx_Ms = AstSExs[_Symm[:, 1]]

        # * Determine Magnitude Offset from Matched Sources
        MAG_Mr = np.array(AstSEx_Mr['MAG_AUTO'])
        MAG_Ms = np.array(AstSEx_Ms['MAG_AUTO'])
        FLUX_Mr = np.array(AstSEx_Mr['FLUX_AUTO'])
        FLUX_Ms = np.array(AstSEx_Ms['FLUX_AUTO'])

        MAGD_M = MAG_Ms - MAG_Mr
        MAG_OFFSET0 = np.median(MAGD_M)

        NTE_4MO = 30  # NUM_TOP_END for MAG_OFFSET, default value used herein.
        MAG_OFFSETr = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_M, weights=FLUX_Mr, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        MAG_OFFSETs = TopFlatten_Weighted_Quantile.TFWQ(values=MAGD_M, weights=FLUX_Ms, \
            quantiles=[0.5], NUM_TOP_END=NTE_4MO)[0]
        MAG_OFFSET = (MAG_OFFSETr + MAG_OFFSETs)/2.0

        if np.abs(MAG_OFFSET - MAG_OFFSET0) > 0.05:
            _warn_message = 'Magnitude Offset Estimate --- HIGHLY DEVIATED FROM DIRECT MEDIAN '
            _warn_message += '[%.3f mag -> %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
            warnings.warn('\nMeLOn WARNING: %s' %_warn_message)
        else:
            _message = 'Magnitude Offset Estimate --- '
            _message += '[%.3f mag -> %.3f mag]!' %(MAG_OFFSET0, MAG_OFFSET)
            print('\nMeLOn CheckPoint: %s' %_message)
        
        # * Combine catalogs
        for coln in AstSEx_Mr.colnames:
            AstSEx_Mr[coln].name = coln + '_REF'
            AstSEx_Ms[coln].name = coln + '_SCI'
        AstSEx_iSS = hstack([AstSEx_Mr, AstSEx_Ms])

        X_IMAGE_REF_SCI_MEAN = np.array(AstSEx_iSS['X_IMAGE_REF'] + AstSEx_iSS['X_IMAGE_SCI'])/2.0
        Y_IMAGE_REF_SCI_MEAN = np.array(AstSEx_iSS['Y_IMAGE_REF'] + AstSEx_iSS['Y_IMAGE_SCI'])/2.0
        AstSEx_iSS.add_column(Column(X_IMAGE_REF_SCI_MEAN, name='X_IMAGE_REF_SCI_MEAN'))
        AstSEx_iSS.add_column(Column(Y_IMAGE_REF_SCI_MEAN, name='Y_IMAGE_REF_SCI_MEAN'))

        # * Only Preserve the Objects in the Prior-Selection (i.e., SubSources)
        #   WARNING NOTE: here we still use the same cross-match tolerance!
        #   WARNING NOTE: we match the prior selection to the mean image coordinates.

        XY_iSS = np.array([AstSEx_iSS['X_IMAGE_REF_SCI_MEAN'], AstSEx_iSS['Y_IMAGE_REF_SCI_MEAN']]).T
        _Symm = Symmetric_Match.SM(XY_A=XY_PriorSelect, XY_B=XY_iSS, tol=UMatchTol, return_distance=False)
        AstSEx_SS = AstSEx_iSS[_Symm[:, 1]]
        
        # record matching and create unified segmentation label
        AstSEx_SS.add_column(Column(_Symm[:, 0], name='INDEX_PRIOR_SELECTION'), index=0)
        AstSEx_SS.add_column(Column(1+np.arange(len(AstSEx_SS)), name='SEGLABEL'), index=0)
        print('MeLOn CheckPoint: Find / Given [%d / %d] Prior-Selected in [%d] Matched-Sources!' \
               %(len(AstSEx_SS), XY_PriorSelect.shape[0], len(AstSEx_iSS)))
        
        # * Run Image Masking (P.S. some SubSources might be marked as invalid by the Prior-Ban list)
        SFFTPrepDict = self.run_image_mask(AstSEx_SS=AstSEx_SS, PixA_SEGr=PixA_SEGr, \
            PixA_SEGs=PixA_SEGs, StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan)
        
        SFFTPrepDict['MAG_OFFSET'] = MAG_OFFSET
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        return SFFTPrepDict
