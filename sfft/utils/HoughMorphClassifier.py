import warnings
import numpy as np
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.HoughDetection import Hough_Detection
from sfft.utils.WeightedQuantile import TopFlatten_Weighted_Quantile
# version: Aug 31, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.3"

class Hough_MorphClassifier:
    
    """
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
    #      METHOD: For simplicity, we only crudely divide the diagram into 3 regions, w.r.t. the vetical line.
    #              they are, Radius-Mid (FR-M), Radius-Large (FR-L) and Radius-Small (FR-S).
    #
    #   B) 3 hierarchic groups                
    #      > Good Sources:
    #        + NOT in FR-S region (union of FR-M & FR-L)
    #          NOTE Good Sources is consist of the vertical & horizontal branches with their cross (not the tail), 
    #               which is roughly equivalent to the set of REAL Point-Sources & Extended Sources 
    #               with rejection the samples in the tail (at faint & small-radius end).
    #          NOTE Good Sources are commonly used as FITTING Candidates in Image Subtraction.
    #               It is acceptable to lose the samples in the tail.
    #         
    #        >> {subgroup} Point Sources:
    #               + Restricted into FR-M Region   |||   Should be located around the Hough-Line
    #               + Basically Circular-Shape      |||   PsfEx-ELLIPTICITY = (A-B) / (A+B) < PS_ELLIPThresh
    #                 NOTE At cross region, this identification criteria mis-include some extended source
    #                      On the flip side, some REAL PointSource samples are missing in the tail.
    #                 NOTE Point Sources are usually employed as FWHM Estimator.
    #                 NOTE We may lossen PS_ELLIPThresh if psf itself is significantly asymmetric (e.g. tracking problem).
    #
    #                 WARNING: At the faint end, this subgroup would be contaminated by round extended sources 
    #                          (meet PS_ELLIPThresh) which also lie in the cross region. 
    #                          When the field is galaxy-dominated, the contamination can be considerable.
    #                          In light of the fact that cross region is located at the faint end in the belt,
    #                          We calculate the median FWHM weighted by source flux, to suppress the over-estimate 
    #                          trend driven by the extended sources.
    #
    #               >>> {sub-subgroup} High-SNR Point Sources
    #                       + SNR_WIN > HPS_SNRThresh, then reject the bright end [typically, 15% (HPS_Reject)] point-sources.
    #                       ++ If remaining sources are less than 30 (HPS_NumLowerLimit),
    #                          Simply Use the point-sources with highest SNR_WIN.
    #                          NOTE Usually, this subset is for Flux-Calibration & Building PSF Model.
    #                          NOTE The defult HPS_SNRThresh = 100 might be too high, you may loosen it to
    #                               ~ 15 to make sure you have enough samples, especially for psf modeling.
    #                          WARNING: For galaxy-dominated field, it would be better to estimate FWHM by this set.
    #                                   As SNR is GAIN-sensitive, please ensure the correctness of the GAIN keyword.
    # 
    #      @ Remarks on the HPS BrightEnd-Cutoff 
    #        Assume SExtractor received precise SATURATE, saturated sources should be fully rejected via FLAG constrain. 
    #        However, in practice, it's hard to fullfill this condition strictly, that is why we design a simple BrightEnd-Cutoff
    #        to prevent the set from such contaminations. Compared with mentioned usage of GS & PS, that of HPS is more 
    #        vulnerable to such situation. FLUX-Calibtation and Building PSF-Model do not require sample completeness, but 
    #        likely to be sensitive to the sources with appreiable non-linear response.
    #
    #   C) Additional WARNINGS
    #      a. This extracor is ONLY designed for sparse field (isolated sources dominated cases).
    #         We usually only take these isloated & non-saturated sources (FLAGS=0) into account in this function.
    #         Sometimes, we may loosen the constrain on FLAGS to be [0,2].
    #
    #      b. We employ Hough Transformation to detect the Stright-Line feature in the image, 
    #         naturally sampled from the raw scatter diagram. But note such diagram makes sense 
    #         only if we could detect enough sources (typically > 200) in the given image.
    #         NOTE Reversed axes employed --- MAG_AUTO [X-axis] - FLUX_RADIUS [Y-axis].
    #
    """

    def MakeCatalog(FITS_obj, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        BACK_TYPE='AUTO', BACK_VALUE=0.0, BACK_SIZE=64, BACK_FILTERSIZE=3, \
        DETECT_THRESH=2.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, \
        BACKPHOTO_TYPE='LOCAL', CHECKIMAGE_TYPE='NONE', AddRD=False, ONLY_FLAGS=[0], \
        BoundarySIZE=30, AddSNR=True):

        # * Trigger SExtractor
        #   NOTE: it is a compromise to adopt XY rather than XYWIN for both point and extended sources.
        #   NOTE: only takes Isolated & Non-Saturated sources (FLAGS = 0) into account.
        #   FIXME: one may need to tune DETECT_THRESH & DETECT_MINAREA for specific program.

        PL = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', \
              'FLAGS', 'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE']
        if AddSNR: PL.append('SNR_WIN')

        PYSEX_OP = PY_SEx.PS(FITS_obj=FITS_obj, PL=PL, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            BACK_TYPE=BACK_TYPE, BACK_VALUE=BACK_VALUE, BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, \
            DETECT_THRESH=DETECT_THRESH, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, \
            DEBLEND_MINCONT=DEBLEND_MINCONT, BACKPHOTO_TYPE=BACKPHOTO_TYPE, CHECKIMAGE_TYPE=CHECKIMAGE_TYPE, \
            AddRD=AddRD, ONLY_FLAGS=ONLY_FLAGS, XBoundary=BoundarySIZE, YBoundary=BoundarySIZE, MDIR=None)
        
        return PYSEX_OP
    
    def Classifier(AstSEx, Hough_FRLowerLimit=0.1, Hough_peak_clip=0.7, BeltHW=0.2, PS_ELLIPThresh=0.3, \
        Return_HPS=False, HPS_SNRThresh=100.0, HPS_SNRTopReject=0.15, HPS_NumLowerLimit=30):
        
        A_IMAGE = np.array(AstSEx['A_IMAGE'])
        B_IMAGE = np.array(AstSEx['B_IMAGE'])
        MA_FR = np.array([AstSEx['MAG_AUTO'], AstSEx['FLUX_RADIUS']]).T
        ELLIP = (A_IMAGE - B_IMAGE) / (A_IMAGE + B_IMAGE)
        MASK_ELLIP = ELLIP < PS_ELLIPThresh

        # * Trigger Hough Dectection 
        #    Use Hough-Transformation detect the Point-Source-Line from the scatter points in 
        #    diagram X [MAG_AUTO] - Y [FLUX_RADIUS], which is a nearly-horizon stright line.
        
        # ** Remarks on the Mask for Hough Transformation
        #    It is s useful to make restriction on FLUX_RADIUS (R) of the scatter points for hough detection.
        #    I. Exclude the sources with unusally large R > 20.0 can speed up the process.
        #    II. The sources with small R (typically ~ 0.5) are likely hot pixels or cosmic rays.
        #        The parameter Hough_FRLowerLimit is the lower bound of FLUX_RATIO for Hough transformation.
        #        Setting a proper lower bound can avoid to detect some line features by chance,
        #        which are not contributed from point sources but resides in the small-FLUX_RATIO region.
        #        NOTE: One need to choose a proper Hough_FRLowerLimit according to the fact if the image is 
        #              under/well/over-sampling (depending on the instrumental configuration and typical seeing conditions)
        #              recommended values of Hough_FRLowerLimit range from 0.1 to 1.0.
        #        NOTE: When the point-source-belt is not very pronounced (e.g., in galaxy dominated fields),
        #              one may consider to reduce the parameter from default 0.7 to, says, ~ 0.4.
        
        MA, FR = MA_FR[:, 0], MA_FR[:, 1]
        MA_MID = np.nanmedian(MA)
        Hmask = np.logical_and.reduce((FR > Hough_FRLowerLimit, FR < 10.0, \
            MA > MA_MID-7.0, MA < MA_MID+7.0))
        
        Hough_grid_pixsize, Hough_count_thresh = 0.05, 1   # Emperical
        HDOP = Hough_Detection.HD(XY_obj=MA_FR, Hmask=Hmask, grid_pixsize=Hough_grid_pixsize, \
            count_thresh=Hough_count_thresh, peak_clip=Hough_peak_clip)
        ThetaPeaks, RhoPeaks, ScaLineDIST = HDOP[2:]
        
        Belt_theta_thresh = 0.2   # Emperical
        Avmask = np.abs(ThetaPeaks) < Belt_theta_thresh
        AvIDX = np.where(Avmask)[0]

        if len(AvIDX) == 0: 
            Horindex = None
            warnings.warn('MeLOn WARNING: [NO] nearly-horizon peak as Point-Source-Belt!')
        if len(AvIDX) == 1:
            Horindex = AvIDX[0]
            print('MeLOn CheckPoint: [UNIQUE] nearly-horizon peak as Point-Source-Belt!')
        if len(AvIDX) > 1:
            Horindex = np.min(AvIDX)
            warnings.warn('MeLOn WARNING: [MULTIPLE] nearly-horizon peaks, use the [STRONGEST] as Point-Source-Belt!')
        
        if Horindex is not None:
            HorThetaPeak = ThetaPeaks[Horindex]
            HorRhoPeak = RhoPeaks[Horindex]
            HorScaLineDIST = ScaLineDIST[:, Horindex]

            _message = 'The Point-Source-Belt detected by Hough Transform '
            _message += 'is characterized by [%.6f, %.6f]!' %(HorThetaPeak, HorRhoPeak)
            print('MeLOn CheckPoint: %s' %_message)

            # *** Note on the FRL region *** 
            # (1) HorThetaPeak is around 0, then cos(HorThetaPeak) around 1 thus >> 0.
            # (2) FRL region (above the line) is x_above * sin(HorThetaPeak) + y_above * cos(HorRhoPeak) > rho.

            MASK_FRM = HorScaLineDIST < BeltHW
            MASK_FRL = MA_FR[:, 0] * np.sin(HorThetaPeak) + MA_FR[:, 1] * np.cos(HorThetaPeak) > HorRhoPeak
            MASK_FRL = np.logical_and(MASK_FRL, ~MASK_FRM)

        else:
            warnings.warn('MeLOn WARNING: [STANDBY] approach is active to determine the FRM regions!')

            # *** Note on the belt determination when Hough transform fails *** 
            # (1) It is quite tricky to find a generic way to determine the point source belt when Hough transform fails.
            #     Typically, the bright sources with moderate Flux Radius are likely point sources.
            # (2) Our strategy is to calculate a median Flux Radius weighted by the source flux, meanwhile,
            #     we modulate the weights by applying a penalty on the sources with large Flux Radius.

            NTE_4FRM = 30  # NUM_TOP_END for FRM, default value used herein.
            _values, _weights = MA_FR[:, 1], np.array(AstSEx['FLUX_AUTO'])
            _weights /= np.clip(_values, a_min=0.5, a_max=None)**2  
            FRM = TopFlatten_Weighted_Quantile.TFWQ(values=_values, weights=_weights, \
                quantiles=[0.5], NUM_TOP_END=NTE_4FRM)[0]
            
            MASK_FRM = np.abs(MA_FR[:, 1]  - FRM) < BeltHW
            MASK_FRL = MA_FR[:, 1]  - FRM > BeltHW
        
        MASK_FRS = ~np.logical_or(MASK_FRM, MASK_FRL)
        LABEL_FR = np.array(['FR-S'] * len(AstSEx))
        LABEL_FR[MASK_FRM] = 'FR-M'
        LABEL_FR[MASK_FRL] = 'FR-L'

        _message = 'Label Counts [FR-S (%s) / FR-M (%s) / FR-L (%s)]!' \
            %(np.sum(MASK_FRS), np.sum(MASK_FRM), np.sum(MASK_FRL))
        print('MeLOn CheckPoint: HoughMorphClassifier --- %s' %_message)

        # * Produce the 3 hierarchic groups
        # ** Good Sources
        MASK_GS = ~MASK_FRS

        # *** Point Sources 
        MASK_PS = np.logical_and(MASK_FRM, MASK_ELLIP)
        print('MeLOn CheckPoint: [%d] Good-Sources on the Image!' %np.sum(MASK_GS))
        print('MeLOn CheckPoint: [%d] Point-Sources on the Image!' %np.sum(MASK_PS))
        
        NTE_4FWHM = 30  # NUM_TOP_END for FWHM, default value used herein.
        _values, _weights = np.array(AstSEx[MASK_PS]['FWHM_IMAGE']), np.array(AstSEx[MASK_PS]['FLUX_AUTO'])
        FWHM = round(TopFlatten_Weighted_Quantile.TFWQ(values=_values, weights=_weights, \
            quantiles=[0.5], NUM_TOP_END=NTE_4FWHM)[0], 6)
        print('MeLOn CheckPoint: Estimated [FWHM = %.3f pix] From Point-Sources' %FWHM)

        # **** High-SNR Point Sources 
        MASK_HPS = None
        if Return_HPS:
            assert 'SNR_WIN' in AstSEx.colnames
            # apply SNR threshold and reject top-SNR sources (security for inappropriate SATURATION)
            MASK_HPS = np.logical_and(MASK_PS, AstSEx['SNR_WIN'] >= HPS_SNRThresh)
            if np.sum(MASK_HPS) > 0 and HPS_SNRTopReject > 0:
                cutoff = np.quantile(AstSEx[MASK_HPS]['SNR_WIN'], 1.0-HPS_SNRTopReject)
                MASK_HPS = np.logical_and(MASK_HPS, AstSEx['SNR_WIN'] < cutoff)
            
            # if there are insufficent number of HPS:
            # abandon the SNR threshold and top-SNR rejection, simply use the point sources with largest SNR instead.
            if np.sum(MASK_HPS) < HPS_NumLowerLimit:
                _warn_message = 'Insufficient Number of High-SNR Point-Sources ---- \n'
                _warn_message += 'Give Up the SNR Threshold and Top-SNR Rejection, ' 
                _warn_message += 'Use Point Sources with Largest SNR instead!'
                warnings.warn('MeLOn WARNING: %s' %_warn_message)

                _PSIDX = np.where(MASK_PS)[0]
                _HPSIDX = _PSIDX[np.argsort(-1.0 * AstSEx[_PSIDX]['SNR_WIN'])[:HPS_NumLowerLimit]]
                MASK_HPS = np.zeros(len(AstSEx)).astype(bool)
                MASK_HPS[_HPSIDX] = True
                if np.sum(MASK_HPS) < HPS_NumLowerLimit:
                    _warn_message = 'Insufficient Number of High-SNR Point-Sources ---- \n'
                    _warn_message += 'Use all Point Sources but STILL Insufficient!'
                    warnings.warn('MeLOn WARNING: %s' %_warn_message)
            print('MeLOn CheckPoint: [%d] High-SNR Point-Sources on the Image!' %np.sum(MASK_HPS))

        return FWHM, LABEL_FR, MASK_GS, MASK_PS, MASK_HPS
