import skimage
import warnings
import numpy as np
from pkg_resources import parse_version
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.HoughDetection import Hough_Detection
from sfft.utils.WeightedQuantile import TopFlatten_Weighted_Quantile

# version: Feb 10, 2023

__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class Hough_MorphClassifier:
    
    def __init__(self):
        
        """
        # MeLOn Notes
        # @ Point-Source Extractor
        #   A) A PSFEx suggested Morphological Classifier, based on a 2D distribution diagram
        #      FLUX_RADIUS [X-axis] - MAG_AUTO [Y-axis], A universal but naive approach.
        #      We first draw all isolated sources on the plane, and the typical distribution will form a 'Y' shape.
        #      A nearly vertical branch, A nearly horizontal branch and their cross with a tail at faint side.
        #
        #      Here I give rough conclusions with comments, note I have compared with Legacy Survety Tractor Catalog.
        #      a. The point sources would be distributed around the nearly vertical stright line. {vertical branch}
        #         NOTE At the faint end, point sources no longer cling to the line, being diffuse in the cross and tail.
        #      b. Close to the bright end of the stright-line are saturated or slight-nonlinear sources, with a deviated direction.
        #      c. The right side of the line are typically extended structure, mainly including various galaxies. {horizontal branch}
        #         NOTE At the faint end, likewise, extended sources also exist, being diffuse in the cross and tail.
        #      d. Scattering points are located at very left side of the line, they are generally hotpix, cosmic-ray or 
        #         some small-scale artifacts. Keep in mind, they are typically outlier-like and away from the cross and tail.
        #
        #      METHOD: For simplicity, we only crudely divide the diagram into 3 regions, w.r.t. the vertical line.
        #              they are, Radius-Mid (FR-M), Radius-Large (FR-L) and Radius-Small (FR-S).
        #
        #   B) 3 hierarchic groups                
        #      > "Good" Sources:
        #        + NOT in FR-S region (union of FR-M & FR-L)
        #          NOTE Good Sources is consist of the vertical & horizontal branches with their cross (not the tail), 
        #               which is roughly equivalent to the set of REAL Point-Sources & Extended Sources 
        #               with rejection the samples in the tail (at faint & small-radius end).
        #          NOTE Good Sources are commonly used as FITTING Candidates in Image Subtraction.
        #               It is acceptable to lose the samples in the tail.
        #         
        #        >> {subgroup} "Point-like" Sources:
        #               + Restricted into FR-M Region   |||   Should be located around the Hough-Line
        #               + Basically Circular-Shape      |||   PSFEx-ELLIPTICITY = (A-B) / (A+B) < PointSource_MINELLIP
        #
        #                 NOTE At cross region, this identification criteria mis-include some extended source
        #                      On the flip side, some REAL PointSource samples are missing in the tail.
        #                 NOTE Point Sources are usually employed as FWHM Estimator.
        #                 NOTE We may lossen PointSource_MINELLIP if psf itself is significantly asymmetric (e.g. tracking problem).
        #
        #                 WARNING: At the faint end, this subgroup would be contaminated by round extended sources 
        #                          (meet PointSource_MINELLIP) which also lie in the cross region. Especially when the field 
        #                          is galaxy-dominated, the contamination can be considerable. In light of the fact that 
        #                          cross region is located at the faint end in the belt, we calculate the median FWHM 
        #                          weighted by source flux, to suppress the over-estimate trend driven by the extended sources.
        #
        #   C) Additional WARNINGS
        #      a. This extracor is ONLY designed for sparse field (isolated sources dominated cases).
        #         We generally only take these isloated & non-saturated sources (FLAGS = 0) into account.
        #         But one may also loosen the constrain on FLAGS to be [0,2].
        #
        #      b. We employ Hough Transform to detect the Stright-Line feature in the image, 
        #         naturally sampled from the raw scatter diagram. But note such diagram makes sense 
        #         only if we could detect enough sources (typically > 200) in the given image.
        #         NOTE Reversed axes employed --- MAG_AUTO [X-axis] - FLUX_RADIUS [Y-axis].
        #
        """

        pass

    def MakeCatalog(FITS_obj, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', BACK_TYPE='AUTO', BACK_VALUE=0.0, \
        BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, ANALYSIS_THRESH=1.5, DETECT_MINAREA=5, \
        DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', CHECKIMAGE_TYPE='NONE', \
        AddRD=False, ONLY_FLAGS=[0], BoundarySIZE=30, AddSNR=True, VERBOSE_LEVEL=2):

        """
        # * Remarks on the SExtractor Catalog for Hough Transform
        #   [1] Although XYWIN can be more accurate than XY for point sources, here 
        #       we adopt XY rather than XYWIN for both point and extended sources as a trade-off.
        #   [2] by default, it only takes isolated & non-saturated sources (FLAGS = 0) into account.
        #   [3] one may need to tune DETECT_THRESH & DETECT_MINAREA for a specific program.
        #
        """

        SExParam = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', \
                     'FLAGS', 'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE']
        
        if AddSNR: SExParam.append('SNR_WIN')
        PYSEX_OP = PY_SEx.PS(FITS_obj=FITS_obj, SExParam=SExParam, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            BACK_TYPE=BACK_TYPE, BACK_VALUE=BACK_VALUE, BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, \
            DETECT_THRESH=DETECT_THRESH, ANALYSIS_THRESH=ANALYSIS_THRESH, DETECT_MINAREA=DETECT_MINAREA, \
            DETECT_MAXAREA=DETECT_MAXAREA, DEBLEND_MINCONT=DEBLEND_MINCONT, BACKPHOTO_TYPE=BACKPHOTO_TYPE, \
            CHECKIMAGE_TYPE=CHECKIMAGE_TYPE, AddRD=AddRD, ONLY_FLAGS=ONLY_FLAGS, XBoundary=BoundarySIZE, \
            YBoundary=BoundarySIZE, MDIR=None, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        return PYSEX_OP
    
    def Classifier(AstSEx, Hough_MINFR=0.1, Hough_MAXFR=10.0, Hough_PeakClip=0.7, \
        BeltHW=0.2, PointSource_MINELLIP=0.3, VERBOSE_LEVEL=2):

        """
        # * calssifier based on hough detection  
        #   We use hough transform to detect the point source belt from the scatter points 
        #   (X: MAG_AUTO, Y: FLUX_RADIUS). The belt is a nearly horizontal straight line.
        #
        #   NOTE: the point source belt has a typical theta of 1.5d +/- 1.0d. on this occasion, 
        #         we should stick with the classic implementation of hough transform in 
        #         scikit-image [0.16.1 to 0.18.3], where the measurement bias on rho
        #         is closer to zero than the new implementation!
        #
        #   NOTE: in general, the point source belt is the strongest line feature. 
        #         by default, -Hough_PeakClip=0.7 is proper in most cases.
        #         however, for galaxy dominated fields (e.g., JWST imagines of galaxy cluster),
        #         the point-source-belt can be not very pronounced, and one may should reduce 
        #         the parameter from default 0.7 to, says, ~ 0.4.
        #
        #   Remarks on the Mask
        #   It can be essential to apply restrictions on FLUX_RADIUS (R) for the scatter points used in hough transform. 
        #   (1) Exclude the sources with unusally large R (e.g., R > 20.0) can speed up the calculations.
        #       One can control the upper limit of R using parameter -Hough_MAXFR
        #
        #   (2) The sources with very small R (e.g., R < 0.5) may become districtions for hough detection.
        #       They are not point sources, likely hot pixels or cosmic rays, sometimes,
        #       wrong line features will be detected by chance without proper rejections.
        #
        #       NOTE: One can control the lower limit of R via parameter -Hough_MINFR,
        #             the recommended values of Hough_MINFR range from 0.1 to 1.0. One should choose 
        #             a proper value according to the instrument configuration and typical seeing conditions.
        #
        """

        skimage_version = parse_version(skimage.__version__)
        assert parse_version('0.16.1') <= skimage_version <= parse_version('0.18.3')
        
        A_IMAGE = np.array(AstSEx['A_IMAGE'])
        B_IMAGE = np.array(AstSEx['B_IMAGE'])
        MA_FR = np.array([AstSEx['MAG_AUTO'], AstSEx['FLUX_RADIUS']]).T
        
        ELLIP = (A_IMAGE - B_IMAGE) / (A_IMAGE + B_IMAGE)
        if PointSource_MINELLIP is not None:
            MASK_ELLIP = ELLIP < PointSource_MINELLIP
        else: MASK_ELLIP = np.ones(len(ELLIP)).astype(bool)

        MA, FR = MA_FR[:, 0], MA_FR[:, 1]
        MA_MID = np.nanmedian(MA)
        
        magnitude_main_range = (-7.0, 7.0)   # emperical
        Hmask = np.logical_and.reduce((FR > Hough_MINFR, FR < Hough_MAXFR, \
            MA > MA_MID + magnitude_main_range[0], MA < MA_MID + magnitude_main_range[1]))

        grid_pixsize, count_thresh = 0.05, 1   # emperical
        _res = Hough_Detection.HD(XY_obj=MA_FR, Hmask=Hmask, grid_pixsize=grid_pixsize, \
            count_thresh=count_thresh, peak_clip=Hough_PeakClip)
        ThetaPeaks, RhoPeaks, ScaLineDIST = _res[2:]

        BeltTheta_thresh = 0.2   # emperical
        NHOR_INDEX = np.where(np.abs(ThetaPeaks) < BeltTheta_thresh)[0]
        
        if len(NHOR_INDEX) == 0: 
            bingo_index = None
            if VERBOSE_LEVEL in [0, 1, 2]:
                _warn_message = '[NO] near-horizon peak as Point-Source-Belt!'
                warnings.warn('MeLOn WARNING: %s' %_warn_message)

        elif len(NHOR_INDEX) == 1:
            bingo_index = NHOR_INDEX[0]
            if VERBOSE_LEVEL in [1, 2]:
                _message = '[UNIQUE] near-horizon peak as Point-Source-Belt!'
                print('MeLOn CheckPoint: %s' %_message)
        
        else:
            bingo_index = np.min(NHOR_INDEX)
            if VERBOSE_LEVEL in [0, 1, 2]:
                _warn_message = '[MULTIPLE] near-horizon peaks, of which [strongest] as Point-Source-Belt!'
                warnings.warn('MeLOn WARNING: %s' %_warn_message)
        
        if bingo_index is not None:
            # * use hough detection method
            BeltTheta = ThetaPeaks[bingo_index]
            BeltRho = RhoPeaks[bingo_index]
            MASK_FRM = ScaLineDIST[:, bingo_index] < BeltHW

            if VERBOSE_LEVEL in [2]:
                _message = 'The Point-Source-Belt detected by Hough Transform '
                _message += 'is characterized by [%f, %f]!' %(BeltTheta, BeltRho)
                print('MeLOn CheckPoint: %s' %_message)

            # *** Note on the FRL region *** 
            # geometrically, we know the scatter points above the near-horizon straight line have
            # x*sin(theta) + y*cos(theta) > rho, where theta is BeltTheta and rho is BeltRho
            
            MASK_FRL = MA_FR[:, 0]*np.sin(BeltTheta) + MA_FR[:, 1]*np.cos(BeltTheta) > BeltRho
            MASK_FRL = np.logical_and(MASK_FRL, ~MASK_FRM)  # exlcude the points in the FR-M region.

        else:
            # * use standby method
            BeltTheta, BeltRho = np.nan, np.nan
            if VERBOSE_LEVEL in [0, 1, 2]:
                _warn_message = '[STANDBY] method to determine FR-S/M/L regions!'
                warnings.warn('MeLOn WARNING: %s' %_warn_message)

            # *** Note on the standby belt determination *** 
            # it is quite tricky to find a generic way to determine the point source belt when 
            # hough detection does not work. we noticed that the bright sources with moderate 
            # Flux Radius are more likely point sources. a standby strategy is to calculate 
            # a median Flux Radius weighted by the source flux, and meanwhile, we modulate 
            # the weights by applying a penalty on the sources with large Flux Radius.

            NTE_4FRM = 30  # NUM_TOP_END for FRM, default value used herein.
            _values, _weights = MA_FR[:, 1], np.array(AstSEx['FLUX_AUTO'])
            _weights /= np.clip(_values, a_min=0.5, a_max=None)**2  
            
            FR_MID = TopFlatten_Weighted_Quantile.TFWQ(values=_values, \
                weights=_weights, quantiles=[0.5], NUM_TOP_END=NTE_4FRM)[0]
            
            MASK_FRM = np.abs(MA_FR[:, 1]  - FR_MID) < BeltHW
            MASK_FRL = MA_FR[:, 1]  - FR_MID > BeltHW

        MASK_FRS = ~np.logical_or(MASK_FRM, MASK_FRL)
        LABEL_FR = np.array(['FR-S'] * len(AstSEx))
        LABEL_FR[MASK_FRM] = 'FR-M'
        LABEL_FR[MASK_FRL] = 'FR-L'

        if VERBOSE_LEVEL in [2]:
            _message = 'HoughMorphClassifier FR-S/M/L counts are '
            _message += '[%d / %d / %d]' %(np.sum(MASK_FRS), np.sum(MASK_FRM), np.sum(MASK_FRL))
            print('MeLOn CheckPoint: %s' %_message)

        # * determine Good Sources
        MASK_GS = ~MASK_FRS

        # * determine Point Sources
        MASK_PS = np.logical_and(MASK_FRM, MASK_ELLIP)
        if VERBOSE_LEVEL in [1, 2]:
            print('MeLOn CheckPoint: [%d] Good-Sources on the Image!' %np.sum(MASK_GS))
            print('MeLOn CheckPoint: [%d] Point-Sources on the Image!' %np.sum(MASK_PS))

        # * estimate FWHM
        NTE_4FWHM = 30  # NUM_TOP_END for FWHM, default value used herein.
        _values, _weights = np.array(AstSEx[MASK_PS]['FWHM_IMAGE']), np.array(AstSEx[MASK_PS]['FLUX_AUTO'])
        FWHM = TopFlatten_Weighted_Quantile.TFWQ(values=_values, weights=_weights, \
            quantiles=[0.5], NUM_TOP_END=NTE_4FWHM)[0]
        
        FWHM = round(FWHM, 6)
        if VERBOSE_LEVEL in [1, 2]:
            print('MeLOn CheckPoint: Estimated [FWHM = %.3f pix] from Point-Sources' %FWHM)

        return BeltTheta, BeltRho, LABEL_FR, MASK_GS, MASK_PS, FWHM
