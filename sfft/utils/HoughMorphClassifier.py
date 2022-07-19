import sys
import warnings
import numpy as np
from tempfile import mkdtemp
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.HoughDetection import Hough_Detection
# version: Jul 19, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.2"

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
    #            + NOT in FR-S region (union of FR-M & FR-L)
    #              NOTE Good Sources is consist of the vertical & horizontal branches with their cross (not the tail), 
    #                   which is roughly equivalent to the set of REAL Point-Sources & Extended Sources 
    #                   with rejection the samples in the tail (at faint & small-radius end).
    #              NOTE Good Sources are commonly used as FITTING Candidates in Image Subtraction.
    #                   It is acceptable to lose the samples in the tail.
    #         
    #           >> {subgroup} Point Sources:
    #                  + Restricted into FR-M Region   |||   Should be located around the Hough-Line
    #                  + Basically Circular-Shape      |||   PsfEx-ELLIPTICITY = (A-B) / (A+B) < PS_ELLIPThresh
    #                    NOTE At cross region, this identification criteria mis-include some extended source
    #                         On the flip side, some REAL PointSource samples are missing in the tail.
    #                    NOTE Point Sources are usually employed as FWHM Estimator.
    #                    NOTE We may lossen PS_ELLIPThresh if psf itself is significantly asymmetric (e.g. tracking problem).
    #                    WARNING: At the faint end, this subgroup would be contaminated by round extended sources 
    #                             (meet PS_ELLIPThresh) which also lie in the cross region. When the field is galaxy-dominated, 
    #                             especially, the contamination can be severe and you may over-estimated FWHM!
    #
    #                  >>> {sub-subgroup} High-SNR Point Sources
    #                          + SNR_WIN > HPS_SNRThresh, then reject the bright end [typically, 15% (HPS_Reject)] point-sources.
    #                          ++ If remaining sources are less than 30 (HPS_NumLowerLimit),
    #                             Simply Use the point-sources with highest SNR_WIN.
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
        BACK_TYPE='AUTO', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, \
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
    
    def Classifier(AstSEx, Hough_FRLowerLimit=0.1, Hough_res=0.05, Hough_count_thresh=1, Hough_peakclip=0.7, \
        LineTheta_thresh=0.2, BeltHW=0.2, PS_ELLIPThresh=0.3, Return_HPS=False, \
        HPS_SNRThresh=100.0, HPS_Reject=0.15, HPS_NumLowerLimit=30):
        
        A_IMAGE = np.array(AstSEx['A_IMAGE'])
        B_IMAGE = np.array(AstSEx['B_IMAGE'])
        MA_FR = np.array([AstSEx['MAG_AUTO'], AstSEx['FLUX_RADIUS']]).T
        ELLIP = (A_IMAGE - B_IMAGE)/(A_IMAGE + B_IMAGE)
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
        #              recommended values of Hough_FRLowerLimit range from 0.1 to 1.0
        
        MA, FR = MA_FR[:, 0], MA_FR[:, 1]
        MA_MID = np.nanmedian(MA)
        Hmask = np.logical_and.reduce((FR > Hough_FRLowerLimit, FR < 10.0, MA > MA_MID-7.0, MA < MA_MID+7.0))
        
        # FIXME USER-DEFINED
        HDOP = Hough_Detection.HD(XY_obj=MA_FR, Hmask=Hmask, res=Hough_res, \
            count_thresh=Hough_count_thresh, peakclip=Hough_peakclip)
        ThetaPeaks, RhoPeaks, ScaLineDIS = HDOP[1], HDOP[2], HDOP[4]

        # NOTE: consider the strongest nearly-horizon peak as the one associated with the point source feature.
        Avmask = np.abs(ThetaPeaks) < LineTheta_thresh
        AvIDX = np.where(Avmask)[0]
        if len(AvIDX) == 0: 
            Horindex = None
            warnings.warn('MeLOn WARNING: NO nearly-horizon peak as Point-Source-Line!')
        if len(AvIDX) == 1:
            Horindex = AvIDX[0]
            print('MeLOn CheckPoint: the UNIQUE nearly-horizon peak as Point-Source-Line!')
        if len(AvIDX) > 1:
            Horindex = np.min(AvIDX)
            warnings.warn('MeLOn WARNING: there are MULTIPLE nearly-horizon peaks and use the STRONGEST as Point-Source-Line!')
        
        if Horindex is not None:
            HorThetaPeak = ThetaPeaks[Horindex]
            HorRhoPeak = RhoPeaks[Horindex]
            HorScaLineDIS = ScaLineDIS[:, Horindex]
            print('MeLOn CheckPoint: the Hough-Detected Point-Source-Line is characterized by (%s, %s)' \
                %(HorThetaPeak, HorRhoPeak))

            # NOTE: Note that HorThetaPeak is around 0, thus cos(HorThetaPeak) around 1 then >> 0, 
            #       thus above-line/FRL region is x_above * sin(HorThetaPeak) + y_above * cos(HorRhoPeak) > rho.
            MASK_FRM = HorScaLineDIS < BeltHW
            MASK_FRL = MA_FR[:, 0] * np.sin(HorThetaPeak) + MA_FR[:, 1] * np.cos(HorThetaPeak) > HorRhoPeak
            MASK_FRL = np.logical_and(MASK_FRL, ~MASK_FRM)
        
        else:
            # NOTE: If we have enough samples, using the bright & small-FR subgroup might be 
            #       more appropriate for the estimate. However, it is quite tricky to find a generic
            #       reliable way to find the point sources when the Hough Transformation doesn't work.
            #       Here we give a simple emperical solution.

            BPmask = AstSEx['MAGERR_AUTO'] < np.percentile(AstSEx['MAGERR_AUTO'], 75)
            Rv = np.percentile(MA_FR[BPmask, 1], 25)
            MASK_FRM = np.abs(MA_FR[:, 1]  - Rv) < BeltHW
            MASK_FRL = MA_FR[:, 1]  - Rv >  BeltHW
            warnings.warn('MeLOn WARNING: the STANDBY approach is actived to determine the FRM region!')
        
        MASK_FRS = ~np.logical_or(MASK_FRM, MASK_FRL)
        LABEL_FR = np.array(['FR-S'] * len(AstSEx))
        LABEL_FR[MASK_FRM] = 'FR-M'
        LABEL_FR[MASK_FRL] = 'FR-L'

        print('MeLOn CheckPoint: count Lables from Hough Transformation [FR-S (%s) / FR-M (%s) / FR-L (%s)] !' \
            %(np.sum(MASK_FRS), np.sum(MASK_FRM), np.sum(MASK_FRL)))

        # * Produce the 3 hierarchic groups
        # ** Good Sources
        MASK_GS = ~MASK_FRS

        # *** Point Sources 
        MASK_PS = np.logical_and(MASK_FRM, MASK_ELLIP)
        assert np.sum(MASK_PS) > 0
        FWHM = round(np.median(AstSEx[MASK_PS]['FWHM_IMAGE']), 6)

        print('MeLOn CheckPoint: Good-Sources in the Image [%d] ' %np.sum(MASK_GS))
        print('MeLOn CheckPoint: Point-Sources in the Image [%d] ' %np.sum(MASK_PS))
        print('MeLOn CheckPoint: Estimated [FWHM = %.3f] pixel from Point-Sources' %FWHM)

        # **** High-SNR Point Sources 
        MASK_HPS = None
        if Return_HPS:
            assert 'SNR_WIN' in AstSEx.colnames
            MASK_HPS = np.logical_and(MASK_PS, AstSEx['SNR_WIN'] >= HPS_SNRThresh)
            if np.sum(MASK_HPS) > 0:
                cutoff = np.quantile(AstSEx[MASK_HPS]['SNR_WIN'], 1.0 - HPS_Reject)
                MASK_HPS = np.logical_and(MASK_HPS, AstSEx['SNR_WIN'] < cutoff)
            
            if np.sum(MASK_HPS) < HPS_NumLowerLimit:
                _IDX = np.where(MASK_PS)[0]
                _IDX = _IDX[np.argsort(AstSEx[_IDX]['SNR_WIN'])[::-1][:HPS_NumLowerLimit]]
                MASK_HPS = np.zeros(len(AstSEx)).astype(bool)
                MASK_HPS[_IDX] = True
                if np.sum(MASK_HPS) < HPS_NumLowerLimit:
                    warnings.warn('MeLOn WARNING: The number of High-SNR Point Sources still does not reach the lower limit !')    
            print('MeLOn CheckPoint: High-SNR Point-Sources in the image [%d]' %np.sum(MASK_HPS))

        """
        # * SHOW THE CLASSIFICATION [Just for Check]

        import matplotlib
        import matplotlib.pyplot as plt
        from astroML.plotting import setup_text_plots  # optional
        setup_text_plots(fontsize=12, usetex=True)       # optional
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
        plt.switch_backend('agg')

        AstSEx_GS = AstSEx[MASK_GS]
        AstSEx_PS = AstSEx[MASK_PS]
        AstSEx_HPS = AstSEx[MASK_HPS]
        MA_FR = np.array([AstSEx['MAG_AUTO'], AstSEx['FLUX_RADIUS']]).T

        plt.figure()
        print('MeLOn CheckPoint: Diagram Checking for HoughMorphClassifier (Natural Axes) !')
        plt.scatter(MA_FR[:, 1], MA_FR[:, 0], s=0.3, color='gray', label='Sources FLAG=0')
        plt.scatter(AstSEx_GS['FLUX_RADIUS'], AstSEx_GS['MAG_AUTO'], s=0.3, color='red', label='Good-Sources')
        plt.scatter(AstSEx_PS['FLUX_RADIUS'], AstSEx_PS['MAG_AUTO'], s=0.3, color='blue', label='Point-Sources')
        plt.scatter(AstSEx_HPS['FLUX_RADIUS'], AstSEx_HPS['MAG_AUTO'], s=3, color='green', label='High-SNR Point-Sources')

        MA0, MA1 = np.min(MA_FR[:, 0]), np.max(MA_FR[:, 0])
        if MA1 - MA0 > 50.0: MA0, MA1 = MA_mid - 25.0,  MA_mid + 25.0
        MAG_PSL = np.arange(MA0, MA1, 0.02)
        FLR_PSL = -np.tan(HorThetaPeak)*MAG_PSL + HorRhoPeak
        plt.plot(FLR_PSL, MAG_PSL, linestyle='dashed', color='black', linewidth=1)

        plt.legend()
        plt.xlabel('FLUX RADIUS')
        plt.ylabel('MAG AUTO')
        plt.xlim(0, 10.0)
        plt.ylim(MA1+1, MA0-1)

        plt.savefig('HMC_Check.png', dpi=400)
        plt.close()
        
        """

        return FWHM, LABEL_FR, MASK_GS, MASK_PS, MASK_HPS
