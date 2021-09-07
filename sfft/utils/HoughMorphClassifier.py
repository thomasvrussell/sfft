import sys
import numpy as np
from tempfile import mkdtemp
from astropy.stats import sigma_clipped_stats 
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
from sfft.utils.HoughDetection import Hough_Detection

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

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
#
#                  >>> {sub-subgroup} High-SNR Point Sources
#                          + SNR_WIN > HPS_SNRThresh, then reject the bright end [typically, 15% (HPS_Reject)] point-sources.
#                          ++ If remaining sources are less than 30 (HPS_NumLowerLimit),
#                             Simply Use the point-sources with highest SNR_WIN.
#                          NOTE In Common, this subset is for Flux-Calibration & Building PSF Model.
# 
#      @ Remarks on the HPS BrightEnd-Cutoff 
#        Assume SExtractor received precise SATURATE, saturated sources should be fully rejected via FLAG constrain. 
#        However, in practice, it's hard to fullfill this condition strictly, that is why we design a simple BrightEnd-Cutoff
#        to prevent the set from such contaminations. Compared with mentioned usage of GS & PS, that of HPS is more 
#        vulnerable to such situation. FLUX-Calibtation and Building PSF-Model do not require sample completeness, but 
#        likely to be sensitive to the sources with appreiable non-linear response.
#
#   C) Additional WARNINGS
#      a. This extracor is ONLY designed for sparse field (isolated sources dominated case).
#         We just take these isloated & non-saturated sources (FLAGS=0) into account in this function.
#
#      b. We employ Hough Transformation to detect the Stright-Line feature in the image, 
#         naturally sampled from the raw scatter diagram. But note such diagram makes sense 
#         only if we could detect enough sources (typically > 200) in the given image.
#         NOTE Reversed axes employed --- MAG_AUTO [X-axis] - FLUX_RADIUS [Y-axis].

"""

class Hough_MorphClassifier:

    def MakeCatalog(FITS_obj, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', \
        BACK_TYPE='AUTO', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, \
        DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
        BACKPHOTO_TYPE='LOCAL', CHECKIMAGE_TYPE='NONE', \
        AddRD=False, BoundarySIZE=30, AddSNR=True):

        # * Trigger SExtractor (do not use XYWIN here as a compromise for the extended sources)
        #   NOTE ONLY take Isolated & Non-Saturated Sources (FLAGS = 0) into account

        PL = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO', \
              'FLAGS', 'FLUX_RADIUS', 'FWHM_IMAGE', 'A_IMAGE', 'B_IMAGE']
        if AddSNR: PL.append('SNR_WIN')

        PYSEX_OP = PY_SEx.PS(FITS_obj=FITS_obj, PL=PL, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            BACK_TYPE=BACK_TYPE, BACK_VALUE=BACK_VALUE, BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, \
            DETECT_THRESH=DETECT_THRESH, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, \
            BACKPHOTO_TYPE=BACKPHOTO_TYPE, CHECKIMAGE_TYPE=CHECKIMAGE_TYPE, AddRD=AddRD, ONLY_FLAG0=True, \
            XBoundary=BoundarySIZE, YBoundary=BoundarySIZE, MDIR=None)
        
        return PYSEX_OP
    
    def Classifier(AstSEx, Hough_res=0.05, Hough_count_thresh=1, Hough_peakclip=0.7, \
        LineTheta_thresh=0.2, BeltHW=0.2, PS_ELLIPThresh=0.3, Return_HPS=False, \
        HPS_SNRThresh=100.0, HPS_Reject=0.15, HPS_NumLowerLimit=30):

        # * Trigger Hough Dectection 
        #    Use Hough-Transformation detect the Point-Source-Line from the scatter points in 
        #    diagram X [MAG_AUTO] - Y [FLUX_RADIUS], which is a roughly horizon stright line.
        #    @ Check top-5 peaks in hough space to find the roughly-horizon stright line, which
        #         is characterized by (HorThetaPeak, HorRhoPeak) in rho = x*sin(theta) + y*cos(theta)
        #         and further calculate the distance from the scatter points to the detected line.
        #    @ Note HorThetaPeak is around 0, thus cos(HorThetaPeak) around 1 then >> 0, 
        #         thus above-line region is x_above * sin(HorThetaPeak) + y_above * cos(HorRhoPeak) > rho.
        #
        #    > Trick: It's useful to make restriction on FLUX_RADIUS (R) of the scatter points [JUST] for hough detection.
        #      NOTE The sources with R value ~ 0.5 are just likely to be hot pixels or cosmic rays 
        #           (typically only 1/2/3 lighted pixels), which are irrelevant for determing the stright line.
        #      NOTE For computability of hough-transformation, it's usful to exclude rare sources with unusal huge R > 20.0
        
        MA_FR = np.array([AstSEx['MAG_AUTO'], AstSEx['FLUX_RADIUS']]).T
        MA, FR = MA_FR[:, 0], MA_FR[:, 1]
        MA_mid = np.nanmedian(MA)
        Hmask = np.logical_and.reduce((FR > 0.1, FR < 10.0, MA > MA_mid-7.0, MA < MA_mid+7.0))  # FIXME USER-DEFINED
        HDOP = Hough_Detection.HD(XY_obj=MA_FR, Hmask=Hmask, res=Hough_res, \
            count_thresh=Hough_count_thresh, peakclip=Hough_peakclip)

        CheckMaxP = 5
        ThetaPeaks, RhoPeaks, ScaLineDIS = HDOP[1][:CheckMaxP], HDOP[2][:CheckMaxP], HDOP[4][:, 0:CheckMaxP]
        Avmask = np.abs(ThetaPeaks) < LineTheta_thresh
        if np.sum(Avmask) > 0:
            Horindex = np.where(Avmask)[0][0]
        else: 
            ThetaPeaks, RhoPeaks, ScaLineDIS = HDOP[1], HDOP[2], HDOP[4]
            Avmask = np.abs(ThetaPeaks) < LineTheta_thresh
            if np.sum(Avmask) > 0:
                Horindex = np.where(Avmask)[0][0]
                print('MeLOn WARNING: Point-Source-Line detected from numerous peaks !')
            else:
                Horindex = None
                print('MeLOn WARNING: Hough Transformation Fails to detect Point-Source-Line !')

        if Horindex is not None:
            HorThetaPeak = ThetaPeaks[Horindex]
            HorRhoPeak = RhoPeaks[Horindex]
            ScaHorLineDIS = ScaLineDIS[:, Horindex]
            print('MeLOn CheckPoint: The Hough-Detected Point-Source-Line is characterized by (%s, %s)' \
                %(HorThetaPeak, HorRhoPeak))
            MASK_FRM = ScaHorLineDIS < BeltHW
            MASK_FRL = MA_FR[:, 0] * np.sin(HorThetaPeak) + MA_FR[:, 1] * np.cos(HorThetaPeak) > HorRhoPeak
            MASK_FRL = np.logical_and(MASK_FRL, ~MASK_FRM)
        else:
            BPmask = AstSEx['MAGERR_AUTO'] < 0.2  # emperical cut, reject samples with low significance.
            Rmid = sigma_clipped_stats(MA_FR[BPmask, 1], sigma=3.0, maxiters=5)[1]
            MASK_FRM = np.abs(MA_FR[:, 1]  - Rmid) < BeltHW
            MASK_FRL = MA_FR[:, 1]  - Rmid >  BeltHW
            print('MeLOn CheckPoint: Alternative scenario is activated to determine the MIDDLE region !')

        MASK_FRS = ~np.logical_or(MASK_FRM, MASK_FRL)
        LABEL_FR =  np.array(['FR-S'] * len(AstSEx))
        LABEL_FR[MASK_FRM] = 'FR-M'
        LABEL_FR[MASK_FRL] = 'FR-L'
        print('MeLOn CheckPoint: Got Lables by Hough Transformation [FR-S (%s) / FR-M (%s) / FR-L (%s)] !' \
            %(np.sum(MASK_FRS), np.sum(MASK_FRM), np.sum(MASK_FRL)))

        # * Produce the 3 hierarchic groups
        # ** Good Sources
        MASK_GS = ~MASK_FRS

        # *** Point Sources 
        A_IMAGE = np.array(AstSEx['A_IMAGE'])
        B_IMAGE = np.array(AstSEx['B_IMAGE'])
        ELLIP = (A_IMAGE - B_IMAGE)/(A_IMAGE + B_IMAGE)
        MASK_PS = np.logical_and(MASK_FRM, ELLIP < PS_ELLIPThresh)
        
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
                    print('MeLOn WARNING: The number of High-SNR Point Sources still does not reach the lower limit !')    
            print('MeLOn CheckPoint: High-SNR Point-Sources in the image [%d]' %np.sum(MASK_HPS))

        """
        # * SHOW THE CLASSIFICATION [Just for Check]

        DIR_4PLOT = '...'
        PNG_Check = '/'.join([DIR_4PLOT, 'HMC_Check.png'])
        AstSEx_GS = AstSEx[MASK_GS]
        AstSEx_PS = AstSEx[MASK_PS]
        AstSEx_HPS = AstSEx[MASK_HPS]
        
        import matplotlib
        import matplotlib.pyplot as plt
        from astroML.plotting import setup_text_plots  # optional
        setup_text_plots(fontsize=12, usetex=True)       # optional
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
        plt.switch_backend('agg')

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
        plt.savefig(PNG_Check, dpi=400)
        plt.close()
        
        """

        return FWHM, LABEL_FR, MASK_GS, MASK_PS, MASK_HPS
