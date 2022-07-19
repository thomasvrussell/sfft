import re
import os
import sys
import warnings
import subprocess
import numpy as np
import os.path as pa
from astropy.io import fits
from astropy.wcs import WCS
from tempfile import mkdtemp
from astropy.table import Table, Column
from sfft.utils.StampGenerator import Stamp_Generator
from sfft.utils.SymmetricMatch import Symmetric_Match, Sky_Symmetric_Match
from sfft.utils.pyAstroMatic.AMConfigMaker import AMConfig_Maker
# version: Jul 19, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.2"

class PY_SEx:
    @staticmethod
    def PS(FITS_obj=None, PSF_obj=None, FITS_ref=None, PL=None, CATALOG_TYPE='FITS_LDAC', \
        GAIN_KEY='GAIN', SATUR_KEY='SATURATE', PIXEL_SCALE=1.0, SEEING_FWHM=1.2, BACK_TYPE='AUTO', \
        BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, USE_FILT=True, DETECT_THRESH=1.5, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_NTHRESH=32, DEBLEND_MINCONT=0.005, CLEAN='Y', \
        BACKPHOTO_TYPE='LOCAL', PHOT_APERTURES=5.0, NegativeCorr=True, CHECKIMAGE_TYPE='NONE', \
        VIGNET=None, StampImgSize=None, AddRD=False, ONLY_FLAGS=None, XBoundary=0.0, YBoundary=0.0, \
        Coor4Match='XY_', XY_Quest=None, Match_xytol=2.0, RD_Quest=None, Match_rdtol=1.0, \
        MDIR=None, verbose='QUIET'):

        """
        # @Python Version of SExtractor
        # * SExtractor Inputs
        #    ** Array-Inputs:
        #       SEx works on one image for signal detection and another image for photometry 
        #       @Individual Mode (Common): Image4detect and Image4phot are the same image.
        #       @Dual Mode: Image4detect and Image4phot are different images.
        #    ** PSF-Input:
        #       SEx can accept given PSF model for PSF-Photometry.
        #    ** Parameter-Inputs: 
        #       a. Basic keywords in FITS header of Image4detect
        #       b. How to generate Global Background Map
        #       c. Give the criteria for SEx-Detection
        #       d. Clarify SEx-Photometry method
        #       e. Specify output Check-Images & output Table by column-name list.
        #
        # * SExtractor Workflow (Background)
        #    ** Extract Global_Background_Map (GBMap) and its RMS (GBRMap) from Image4detect & Image4phot
        #       Control Parameters: BACK_TYPE, BACK_VALUE, BACK_SIZE and BACK_FILTERSIZE.
        #       @ Produce GBMap 
        #         a. Manual-FLAT (e.g. BACK_TYPE='MANUAL', BACK_VALUE='100.0')
        #            SEx directly define GBMap as a FLAT image with given constant BACK_VALUE.
        #         b. Auto (e.g. BACK_TYPE='AUTO', BACK_SIZE=64, BACK_FILTERSIZE=3)
        #            SEx defines a mesh of a grid that covers the whole frame by [BACK_SIZE].
        #            i. Convergence-based sigma-clipping method on the flux histogram of each tile.
        #               More specificly, the flux histogram is clipped iteratively until convergence at +/- 3sigma around its median.
        #            ii. SEx compute local background estimator from the clipped histogram of each tile.
        #                If sigma is changed by less than 20% during clipping process (the tile is not crowded), 
        #                use the mean of the clipped histogram as estimator
        #                otherwise (the tile is crowded), mode = 2.5*median - 1.5*mean is employed instead.
        #            iii. Once the estimate grid is calculated, a median filter [BACK_FILTERSIZE] can be applied 
        #                 to suppress possible local overestimations.
        #            iv. The resulting background map is them simply a bicubic-spline interpolation between the meshes of the grid.
        #
        #        @ Generate GBRMap
        #            only Auto (e.g, BACK_SIZE=64, BACK_FILTERSIZE=3)
        #            SEx produces the noise map by the same approach of Auto style of GBMap, where the only difference is that
        #            SEx is [probably] use standard deviation as estimator of the clipped flux historgram, other than mean or mode.
        #        
        #        NOTE Abbr. GBMap / GBRMap from Image4detect: GBMap_4d / GBRMap_4d
        #             Abbr. GBMap / GBRMap from Image4phot: GBMap_4p / GBRMap_4p
        #        NOTE WARNING: Dual Mode have to use consistent control parameters for Image4detect & Image4phot, 
        #                      as it is not allowed to set some secondary configuration in SEx software framework, 
        #                      despite that it  is not necessarily reasonable in some cases.
        #
        # * SExtractor Workflow (Detection)
        #    ** a. SEx-Detect on Image4detect: SkySubtraction & Filtering & Thresholding & Deblending & AreaConstrain & Clean
        #          @ SkySubtraction & Filtering Process (e.g. FILTER='Y', FILTER_NAME='default.conv')
        #              SEx Remove GBMap_4d from Image4detect and then perform a convolution to maximizes detectability. 
        #              NOTE The power-spectrum of the noise and that of the superimposed signal can be signiﬁcantly different.
        #              NOTE Although Filtering is a beneﬁt for detection, it distorts proﬁles and correlates the noise.
        #              NOTE Filtering is applied 'on the ﬂy' to the image, and directly affects only the following 
        #                   Thresholding process and Isophotal parameters.
        #
        #          @ Thresholding Process (e.g. DETECT_THRESH=1.5, THRESH_TYPE='RELATIVE')
        #              SEx highlights the pixels in Filtered_Image with Threshold_Map = DETECT_THRESH * GBRMap_4d
        #              We would be bettter to imagine SEx actually make a hovering mask.
        #
        #          @ Deblending Process (e.g. DEBLEND_MINCONT=0.005, DEBLEND_NTHRESH=32)
        #              SEx triggers a deblending process to dentify signal islands from the hovering mask [probably] on Filtered_Image,
        #              which converts the hovering mask to be a hovering label map.
        #
        #          @ Put AreaConstrain (e.g. DETECT_MINAREA=5, DETECT_MAXAREA=0)
        #              MinMax AreaContrain is applied and then iolated cases then lose their hovering labels. 
        #
        #          @ Clean Process (e.g. CLEAN='YES')
        #              SEx will clean the list of objects of artifacts caused by bright objects.
        #              As a correction process, all cleaned objects are subsequently removed from the hovering label map.
        #              NOTE Now the hovering label map is the SEGMENTATION check image. One may refer to such label island as ISOIsland
        #              NOTE These labels are consistent with the indices in the ouput photometry table.
        #              NOTE SEx will report something like this: Objects: detected 514 / sextracted 397 
        #                   SET CLEAN='N', you could find detected == sextracted.
        #
        #    ** b. Generate Positional & BasicShape Paramters from isophotal profile
        #          NOTE In this section 'pixel flux' means values of Filtered_Image.
        #          i. XMIN, XMAX, YMIN, YMAX deﬁne a rectangle which encloses the ISOIsland.
        #          ii. Barycenter X_IMAGE, Y_IMAGE is the first order moments of the profile, where the pixel flux of
        #              ISOIsand is the corresponding weight for cacluatuing Barycenter.
        #          iii. Centered second-order moments X2_IMAGE, Y2_IMAGE, XY2_IMAGE are convenient for
        #               measuring the spatial spread of a source proﬁle. likewise, pixel fluxs of ISOIsand are weights.
        #               (if a parameter can be fully expressed by them, we will use $ to indicate).
        #          iv. Describe ISOIsand as an elliptical shape, centred at Barycenter.
        #               $A_IMAGE, $B_IMAGE are ellipse semi-major and semi-minor axis lengths, respectively.
        #               The ellipse is uniquely determined by $CXX_IMAGE, $CYY_IMAGE, $CXY_IMAGE with KRON_RADIUS, 
        #               where KRON_RADIUS is independently calculated in a routine inspired by Kron's 'first moment' algorithm.
        #               NOTE By-products: $ELONGATION = A / B and $ELLIPTICITY = 1 - B / A
        #
        #    ** c. Generate Positional & BasicShape Paramters from Window
        #          NOTE In this section 'pixel flux' [likely] means values of Image4phot !
        #          NOTE It is designed for Refinement, NOT for Photometry, thereby all of these parameters are irrelevant to Photometry !
        #          METHOD: The computations involved are roughly the same except that the domain is a circular Gaussian window,
        #                  (I don't know what is the window radius) as opposed to the object's isophotal footprint (ISOIsland).
        #          MOTIVATION: Parameters measured within an object's isophotal limit are sensitive to two main factors: 
        #                      + Changes in the detection threshold, which create a variable bias 
        #                      + Irregularities in the object's isophotal boundaries, which act as 
        #                        additional 'noise' in the measurements.
        #
        #          @Positional: This is an iterative process. The computation starts by initializing 
        #                       the windowed centroid coordinates to the Barycenter.
        #                       The process will adjust window and finally its centroid converges 
        #                       at some point: XWIN_IMAGE, YWIN_IMAGE.
        #                       (If the process is failed then XWIN_IMAGE, YWIN_IMAGE = X_IMAGE, Y_IMAGE)
        #                       It has been veriﬁed that for isolated, Gaussian-like PSFs, its accuracy is close to 
        #                       the theoretical limit set by image noise. 
        #                       NOTE: We preferably use it for point sources, like in transient detection. 
        #                             However it may not optimal for extended sources like glaxies.
        #                       NOTE: X_IMAGE & Y_IMAGE seems to be a good compromise choice.
        #
        #          @BasicShape: Windowed second-order moments are computed once the centering process has converged.
        #                       X2WIN_IMAGE, Y2WIN_IMAGE, XY2WIN_IMAGE
        #                       AWIN_IMAGE, BWIN_IMAGE, CXXWIN_IMAGE, CYYWIN_IMAGE, CXYWIN_IMAGE
        #
        #          NOTE: Positional XWIN_IMAGE and YWIN_IMAGE are quite useful to 
        #                provide a refined Object coordinate (of gaussian-like point sources). 
        #                However we seldom use Window version of BasicShape parameters 
        #                to describe the shape of the approaximated ellipse.
        #
        #    ** d. Generate Positional Paramters from PSF Fitting
        #          NOTE In this section 'pixel flux' is [possibly] means values of skysubtracted-Image4detect !
        #          METHOD: XPSF_IMAGE and YPSF_IMAGE are fitted from given PSF model.
        #
        # * SExtractor Workflow (Photometry)
        #    ** Count Flux with various photometry methods
        #          NOTE This process always works on Image4phot.
        #          @ISO: SEx simply count flux according to the hovering label map (ISOIsland).
        #          @APER: SEx count flux on a Circular Aperture with given PHOT_APERTURES (centred at Barycenter X_IMAGE, Y_IMAGE).
        #          @AUTO: SEx count flux on a Elliptic Aperture, which is determined by the hovering isophotal ellipse 
        #                 (centred at Barycenter X_IMAGE, Y_IMAGE). The leaked light fraction is typically less than 10%.
        #          @PSF: SEx count flux according to the PSF fitting results (centred at XPSF_IMAGE and YPSF_IMAGE).
        #                This is optimal for pointsource but fairly wrong for extended objects.
        #
        #    ** Peel background contribution from Counted Flux
        #          @BACKPHOTO_TYPE='LOCAL': background will take a rectangular annulus into account, 
        #                                   which has a donnut shape around the object, measured on Image4phot.
        #          @BACKPHOTO_TYPE='GLOBAL': background will directly use GBMap_4p,
        #                                    which can be Manual-Flat or Auto.
        #
        #    ** Noise Estimation
        #          METHOD: SEx estimate the photometric noise contributed from photon possion distribution and sky background.
        #                  that is, FLUXERR^2 = FLUX / GAIN + Area * skysig^2
        #          NOTE: Area means the pixel area of flux-count domain for various photometry methods.
        #          NOTE: skysig^2 is [probably] the sum of GBRMap_4p within the flux-count domain.
        #          NOTE: Bright Sources are Photon Noise Dominated cases, while Faint Sources are Sky Noise Dominated cases.
        #
        # * SExtractor Workflow (Check-Image)
        #    @BACKGROUND : GBMap_4p
        #    @BACKGROUND_RMS : GBRMap_4d
        #    @-BACKGROUND : Image4phot - GBMap_4p
        #    @FILTERED: Filtered_Image 
        #    @SEGMENTATION: ISOIsland Label Map
        #    @OBJECTS: ISOIslands use flux in -BACKGROUND, zero otherwise
        #    @APERTURES: -BACKGROUND with brighten apertures to show the isophotal ellipses. 

        # * SExtractor Workflow (Formula)
        #    a. MAG = -2.5 * log10(FLUX) + MAG_ZEROPOINT
        #    b. MAGERR = 1.0857 * FLUXERR / FLUX = 1.0857 / SNR 
        #       NOTE: a common crude approxmation SNR ~ 1/MAGERR.
        #       NOTE: It is [probably] derived from MAGERR = 2.5*np.log10(1.0+1.0/SNR) 
        #                   Ref: http://www.ucolick.org/~bolte/AY257/s_n.pdf
        #                   ZTF use a more precise factor 1.085736, whatever, not a big deal.
        #
        # * Additional Clarification
        #    a. SExtractor can read GAIN & SATURATION & MAGZERO from FITS header of Image4phot by their keys.
        #    b. If SExtractor is configured with FITS_LDAC, the FITS header of Image4phot will be delivered 
        #       into output FITS file saved at output-FITS[1].data in table format (only one element: a long integrated header text). 
        #    c. If some ISOIsland has saturated pixel value on Image4phot, you can still find it 
        #       on SEGMENTATION / OBJECTS, and it will be marked by FLAGS=4 in output catalog.
        #    d. Windowed Coordinate has higher priority in the function
        #       i. Make Stamps ii. Convert to RD iii. Symmetric-Match
        #       First use XWIN_IMAGE & YWIN_IMAGE (if exist), 
        #       otherwise, employ X_IMAGE & Y_IMAGE instead.
        #    e. SExtractor allows to submit request for multi-check images 
        #       e.g. CHECKIMAGE_TYPE = "BACKGROUND,SEGMENTATION,..."
        #    f. Although SExtractor can directly provide sky coordinates in output table,
        #       We always independently to get them by convert XY using astropy. 
        #    g. If VIGNET is called in SExtractor, stamps will be extracted around the targets, 
        #       centred at their Barycenter X_IMAGE, Y_IMAGE, from Image4phot.
        #
        # * Additional Tips
        #    a. SNR_WIN: Window-based Gaussian-weighted SNR estimate
        #       Although SNR_WIN is empirically tend to slightly underestimate noise, 
        #       this useful parameter is calculated independently from the employed phot-method.
        #
        #    b. If you got a long runtime, it may caused by 
        #       i. Low DETECT_THRESH 
        #       ii. Request XWIN_IMAGE, YWIN_IMAGE, SNR_WIN
        #       iii. BACKPHOTO_TYPE == 'LOCAL'
        #
        #    c. WARNINGS
        #       i. You may encounter bug if sethead after fits.getdata 
        #       ii. SExtractor do not support string > 256 as argument in command line.
        #           Use 'cd dir && sex ...' to avoid segmentation fault.
        #       iii. FITS_LDAC --- TABLE-HDU 2    |   FITS_1.0 ---  TABLE-HDU 1
        #       iv. In Debian operation system, please correct 'sex' as 'sextractor'`
        #
        # README
        # * 2 models for input image*
        #    @ Dual-Image Mode: FITS_ref is Image4detect, FITS_obj is Image4phot.
        #      ** When this mode is called, please read above comments very carefully.
        #    @ Single-Image Mode: FITS_obj is Image4detect & Image4phot.
        #      ** When sky background has been well subtracted. 
        #         Typically Set: BACK_TYPE='MANUAL', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, BACKPHOTO_TYPE='LOCAL'
        #         a. BACK_TYPE & BACK_VALUE: Use Flat-Zero as Global_Background_Map
        #         b. BACK_SIZE & BACK_FILTERSIZE: Produce RMS of Global_Background_Map [AUTO]
        #         c. BACKPHOTO_TYPE: Use LOCAL / GLOBAL (zero) background to count sky flux contribution in photometry.
        #
        # * Additional Remarks on Background
        #   @ well-subtracted sky just means it probably outperforms SEx GLOBAL-Sky, 
        #       we should not presume the underlying true sky is really subtracted, therefore, 
        #       BACKPHOTO_TYPE = 'LOCAL' is in general necessary !
        #   @ Just like sky subtraction, the sky term in image subtraction can only handle the low-spatial-frequency trend of background.
        #       The image has flat zero background as an ideal expectation, we still need set BACKPHOTO_TYPE = 'LOCAL' on difference photometry.
        #   @ 'LOCAL' method itself can be biased too. We get the conclusion when we try to perform aperture photometry on two psf-homogenized DECam images.
        #       Recall any error on matched kernel is some linear effect for aperture photometry, however, we found a bias in FLUX_APER which is independent 
        #       with the target birghtness. E.g. FLUX_APER_SCI is always smaller than FLUX_APER_REF with a nearly constant value, say 10.0 ADU, 
        #       no matter the target is 19.0 mag or 22.0 mag, as if there is some constant leaked light when we measure on SCI. 
        #       Equivalently, DMAG = MAG_APER_SCI - MAG_APER_REF deviate zero-baseline, and it becomes increasingly serious (towards to the faint end).
        #       ------------------------------------ 
        #       We guess the problem is caused by the fact: the background value calculated from the annulus around target might be a biased 
        #       (over/under-) estimation. One observation supports our argument: the flux bias is much more evident when we increase the aperture size.
        #       NOTE:  As we have found the flux bias is basically a constant, we can do relative-calibration by calculate the compensation 
        #              offset FLUX_APER_REF - FLUX_APER_SCI for a collection of sationary stars. It is 'relative' since we have just assumed 
        #              background estimation of REF is correct, which is proper if we are going to derive the variability (light curve).
        #   @ In which cases, Re-Run SExtractor can get the same coordinate list ?
        #       a. Same configurations but only change photometric method, e.g. from AUTO to APER 
        #       b. Same configurations but from Single-Image Mode to Dual-Image Mode
        #       NOTE: FLAGS is correlated to object image, if we add constraint FLAG=0 then we fail to get the same coordinate list.
        #
        """
        
        # * sex or sextractor ?
        for cmd in ['sex', 'sextractor']:
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                del p
                break
            except OSError:
                continue
        else:
            raise FileNotFoundError('Source Extractor command not found.')
        SEx_KEY = cmd
        del cmd

        # * Make Directory as workplace
        TDIR = mkdtemp(suffix=None, prefix='PYSEx_', dir=MDIR)
        print('\nMeLOn: Run Python Version of SExtractor on FITS file: %s' %FITS_obj)

        # * Keyword Configurations 
        phr_obj = fits.getheader(FITS_obj, ext=0)       
        if GAIN_KEY in phr_obj:
            GAIN = phr_obj[GAIN_KEY]
            print('MeLOn CheckPoint: SEx use GAIN=%s [READ from FITS header]' %GAIN)
        else: 
            GAIN = 0.0 
            warnings.warn('MeLOn WARNING: SEx use GAIN=%s [SEx DEFAULT]' %GAIN)
        
        if SATUR_KEY in phr_obj:
            SATURATION = phr_obj[SATUR_KEY]
            print('MeLOn CheckPoint: SEx use SATURATION=%s [READ from FITS header]' %SATURATION)
        else: 
            SATURATION = 50000.0
            warnings.warn('MeLOn WARNING: SEx use SATURATION=%s [SEx DEFAULT]' %SATURATION)
        
        """
        # some additional remarks
        # [1] MAGERR/FLUXERR/SNR are very sensitive to GAIN value.
        # [2] PIXEL_SCALE (unit: arcsec) only works for surface brightness parameters, 
        #     FWHM (FWHM_WORLD) and star/galaxy separation. PIXEL_SCALE=0 uses FITS WCS info.
        # [3] SEEING_FWHM (unit: arcsec) is only for star/galaxy separation.
        #     VIP: You'd better to give a good estimate if star/galaxy separation is needed.
        # [4] A common criteria for isolated point sources:
        #     CLASS_STAR (output parameter) > 0.95 & FLAG = 0
        #
        """

        # * Main Configurations
        ConfigDict = {}
        ConfigDict['CATALOG_TYPE'] = CATALOG_TYPE
        ConfigDict['VERBOSE_TYPE'] = '%s' %verbose

        ConfigDict['GAIN_KEY'] = '%s' %GAIN_KEY
        ConfigDict['SATUR_KEY'] = '%s' %SATUR_KEY
        ConfigDict['MAG_ZEROPOINT'] = '0.0'        
        ConfigDict['PIXEL_SCALE'] = '%s' %PIXEL_SCALE
        ConfigDict['SEEING_FWHM'] = '%s' %SEEING_FWHM
        
        ConfigDict['BACK_TYPE'] = '%s' %BACK_TYPE
        ConfigDict['BACK_VALUE'] = '%s' %BACK_VALUE
        ConfigDict['BACK_SIZE'] = '%s' %BACK_SIZE
        ConfigDict['BACK_FILTERSIZE'] = '%s' %BACK_FILTERSIZE

        ConfigDict['DETECT_THRESH'] = '%s' %DETECT_THRESH
        ConfigDict['DETECT_MINAREA'] = '%s' %DETECT_MINAREA
        ConfigDict['DETECT_MAXAREA'] = '%s' %DETECT_MAXAREA
        ConfigDict['DEBLEND_NTHRESH'] = '%s' %DEBLEND_NTHRESH
        ConfigDict['DEBLEND_MINCONT'] = '%s' %DEBLEND_MINCONT
        ConfigDict['CLEAN'] = '%s' %CLEAN
        
        ConfigDict['CHECKIMAGE_TYPE'] = '%s' %CHECKIMAGE_TYPE
        ConfigDict['BACKPHOTO_TYPE'] = '%s' %BACKPHOTO_TYPE
        if not isinstance(PHOT_APERTURES, (int, float)):
            ConfigDict['PHOT_APERTURES'] = '%s' %(','.join(np.array(PHOT_APERTURES).astype(str)))
        else: ConfigDict['PHOT_APERTURES'] = '%s' %PHOT_APERTURES
        if PSF_obj is not None: ConfigDict['PSF_NAME'] = '%s' %PSF_obj
        
        # create configuration file .conv
        if USE_FILT:
            # see https://github.com/astromatic/sextractor/blob/master/config/default.conv
            conv_text = "CONV NORM\n"
            conv_text += "# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.\n"
            conv_text += "1 2 1\n"
            conv_text += "2 4 2\n"
            conv_text += "1 2 1"
            conv_path = ''.join([TDIR, "/PYSEx.conv"])
            _cfile = open(conv_path, 'w')
            _cfile.write(conv_text)
            _cfile.close()
        
        USE_NNW = False
        if 'CLASS_STAR' in PL: 
            USE_NNW = True
        
        if USE_NNW:
            # see https://github.com/astromatic/sextractor/blob/master/config/default.nnw
            nnw_text = r"""
            NNW
            # Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
            # inputs:	9 for profile parameters + 1 for seeing.
            # outputs:	``Stellarity index'' (0.0 to 1.0)
            # Seeing FWHM range: from 0.025 to 5.5'' (images must have 1.5 < FWHM < 5 pixels)
            # Optimized for Moffat profiles with 2<= beta <= 4.
            
             3 10 10  1
            
            -1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02  8.31957e-01  2.15505e+00  2.64769e-01
             3.03477e+00  2.69561e+00  3.16188e+00  3.34497e+00  3.51885e+00  3.65570e+00  3.74856e+00  3.84541e+00  4.22811e+00  3.27734e+00
            
            -3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01  5.52395e-01 -4.36564e-01 -5.30052e+00
             4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01  1.29492e-01  1.42290e+00  2.90741e+00  2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
            -2.57424e+00  8.96469e-01  8.34775e-01  2.18845e+00  2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02  9.30403e-02  1.64942e+00 -1.01231e+00
             4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00  5.59549e-01  8.08468e-01 -1.01592e-02 -7.54052e+00
             1.01933e+01 -2.09484e+01 -1.07426e+00  9.87912e-01  6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
            -5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00  1.45967e+00  1.62946e+00 -1.01301e+00  1.13514e-01
             2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01  1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00  8.23195e+00 -1.71561e-02 -1.13749e+01
             3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01  1.29122e+00 -7.49380e-01  6.71712e-01
            -8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01  1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
            -3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01  1.27910e-01  1.91906e+00  1.42119e+00 -4.35502e+00
            
            -1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00  2.42038e+00  1.22919e+00 -9.22250e-01 -2.32028e+00
            
            
             0.00000e+00 
             1.00000e+00 """

            nintent = len(re.split('NNW', re.split('\n', nnw_text)[1])[0])
            nnw_text = '\n'.join([line[nintent: ] for line in re.split('\n', nnw_text)[1:]])
            nnw_path = ''.join([TDIR, "/PYSEx.nnw"])
            _cfile = open(nnw_path, 'w')
            _cfile.write(nnw_text)
            _cfile.close()

        # create configuration file .param
        Param_lst = PL.copy()
        if PL is None: Param_lst = []
        if 'X_IMAGE' not in Param_lst: 
            Param_lst.append('X_IMAGE')
        if 'Y_IMAGE' not in Param_lst: 
            Param_lst.append('Y_IMAGE')

        Param_path = ''.join([TDIR, "/PYSEx.param"])
        if VIGNET is not None: Param_lst += ['VIGNET(%d, %d)' %(VIGNET[0], VIGNET[1])]
        Param_text = '\n'.join(Param_lst)
        pfile = open(Param_path, 'w')
        pfile.write(Param_text)
        pfile.close()

        # create configuration file .sex
        if not USE_FILT: ConfigDict['FILTER'] = 'N'
        if USE_FILT: ConfigDict['FILTER_NAME'] = "%s" %conv_path
        if USE_NNW: ConfigDict['STARNNW_NAME'] = "%s" %nnw_path
        ConfigDict['PARAMETERS_NAME'] = "%s" %Param_path

        FNAME = pa.basename(FITS_obj)
        FITS_SExCat = ''.join([TDIR, '/%s_PYSEx_CAT.fits' %FNAME[:-5]])
        _cklst = re.split(',', CHECKIMAGE_TYPE)
        FITS_SExCheckLst = [''.join([TDIR, '/%s_PYSEx_CHECK_%s.fits' %(FNAME[:-5], ck)]) for ck in _cklst]

        sexconfig_path = AMConfig_Maker.AMCM(MDIR=TDIR, AstroMatic_KEY=SEx_KEY, \
            ConfigDict=ConfigDict, tag='PYSEx')
            
        # * Trigger SExtractor
        if FITS_ref is None:
            os.system("cd %s && %s %s -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s"\
                %(pa.dirname(FITS_obj), SEx_KEY, FNAME, sexconfig_path, FITS_SExCat, ','.join(FITS_SExCheckLst)))
        
        if FITS_ref is not None:
            # WARNING: more risky to fail due to the too long string.
            os.system("%s %s,%s -c %s -CATALOG_NAME %s -CHECKIMAGE_NAME %s"\
                %(SEx_KEY, FITS_ref, FITS_obj, sexconfig_path, FITS_SExCat, ','.join(FITS_SExCheckLst)))    
        
        """
        # deprecated as it requires WCSTools, and seems not very useful
        def record(FITS):
            os.system('sethead %s SOURCE=%s' %(FITS, FITS_obj))
            os.system('sethead %s RSOURCE=%s' %(FITS, FITS_ref))
            for key in ConfigDict:
                value = ConfigDict[key]
                pack = ' : '.join(['Sex Parameters', key, value])
                os.system('sethead %s HISTORY="%s"' %(FITS, pack))
            return None
        """

        if CATALOG_TYPE == 'FITS_LDAC': tbhdu = 2
        if CATALOG_TYPE == 'FITS_1.0': tbhdu = 1
        if pa.exists(FITS_SExCat):
            #if MDIR is not None: record(FITS_SExCat)
            AstSEx = Table.read(FITS_SExCat, hdu=tbhdu)
        else: FITS_SExCat, AstSEx = None, None
        
        PixA_SExCheckLst = []
        FtmpLst = FITS_SExCheckLst.copy()
        for k, FITS_SExCheck in enumerate(FITS_SExCheckLst):
            if pa.exists(FITS_SExCheck):
                #if MDIR is not None: record(FITS_SExCheck)
                PixA_SExCheck = fits.getdata(FITS_SExCheck, ext=0).T
            else: PixA_SExCheck, FtmpLst[k]  = None, None
            PixA_SExCheckLst.append(PixA_SExCheck)
        FITS_SExCheckLst = FtmpLst

        # * Optional functions
        if AstSEx is not None:
            Modify_AstSEx = False

            # ** a. CORRECT the SExtractor bug on MAG & MAGERR due to negative flux count.
            #       For such cases, corresponding MAG & MAGERR will turn to be a trivial 99
            #       PYSEx will re-calculate them from FLUX & FLUXERR 
            #       if MAG & MAGERR in PL and NegativeCorr=True.
                        
            MAG_TYPES = [p for p in Param_lst if p[:4] == 'MAG_']
            if len(MAG_TYPES) > 0:
                if NegativeCorr:
                    MAGERR_TYPES = ['MAGERR_' + MAGT[4:] for MAGT in MAG_TYPES]
                    FLUX_TYPES = ['FLUX_' + MAGT[4:] for MAGT in MAG_TYPES]
                    FLUXERR_TYPES = ['FLUXERR_' + MAGT[4:] for MAGT in MAG_TYPES]
                    
                    for i in range(len(MAG_TYPES)):
                        pcomplete = (MAGERR_TYPES[i] in Param_lst) & \
                                    (FLUX_TYPES[i] in Param_lst) & \
                                    (FLUXERR_TYPES[i] in Param_lst)
                        
                        if not pcomplete:
                            sys.exit('MeLOn ERROR: NEGATIVE FLUX Correction requires ' + \
                                     'complete FLUX FLUXERR MAG MAGERR columns')

                        FLUX = np.array(AstSEx[FLUX_TYPES[i]])
                        FLUXERR = np.array(AstSEx[FLUXERR_TYPES[i]])
                        Mask_NC = FLUX < 0.0
                        MAG_NC = -2.5*np.log10(np.abs(FLUX[Mask_NC]))
                        MAGERR_NC = 1.0857 * np.abs(FLUXERR[Mask_NC] / FLUX[Mask_NC])
                        AstSEx[MAG_TYPES[i]][Mask_NC] = MAG_NC
                        AstSEx[MAGERR_TYPES[i]][Mask_NC] = MAGERR_NC
                    Modify_AstSEx = True

            # ** b. ADD-COLUMN SEGLABEL if SEGMENTATION requested.
            #       If some lines are discarded later, corresponding 
            #       SEGLABEL will be lost in the output table.

            if 'SEGMENTATION' in CHECKIMAGE_TYPE:
                SEGLABEL = 1 + np.arange(len(AstSEx))   # label 0 is background
                AstSEx.add_column(Column(SEGLABEL, name='SEGLABEL'), index=0)
                Modify_AstSEx = True

            # ** c. ADD-COLUMN of Sky-Coordinates by Astropy
            #       (X_IMAGE, Y_IMAGE) to (X_WORLD, Y_WORLD)
            #       (XWIN_IMAGE, YWIN_IMAGE) to (XWIN_WORLD, YWIN_WORLD)

            if AddRD:
                def read_wcs(FITS_obj):
                    phr = fits.getheader(FITS_obj, ext=0)
                    if phr['CTYPE1'] == 'RA---TAN' and 'PV1_0' in phr:
                        _hdr = phr.copy()
                        _hdr['CTYPE1'] = 'RA---TPV'
                        _hdr['CTYPE2'] = 'DEC--TPV'
                        w = WCS(_hdr)
                    else: w = WCS(phr)
                    return w
                
                w_obj = read_wcs(FITS_obj)
                _XY = np.array([AstSEx['X_IMAGE'], AstSEx['Y_IMAGE']]).T
                _RD = w_obj.all_pix2world(_XY, 1)
                AstSEx.add_column(Column(_RD[:, 0], name='X_WORLD'))          
                AstSEx.add_column(Column(_RD[:, 1], name='Y_WORLD'))

                if 'XWIN_IMAGE' in Param_lst:
                    if 'YWIN_IMAGE' in Param_lst:
                        _XY = np.array([AstSEx['XWIN_IMAGE'], AstSEx['YWIN_IMAGE']]).T
                        _RD = w_obj.all_pix2world(_XY, 1)
                        AstSEx.add_column(Column(_RD[:, 0], name='XWIN_WORLD'))          
                        AstSEx.add_column(Column(_RD[:, 1], name='YWIN_WORLD'))
                Modify_AstSEx = True
            
            # ** d. Restriction on FLAGS
            if ONLY_FLAGS is not None:
                if 'FLAGS' not in PL:
                    sys.exit('MeLOn ERROR: Restriction of FLAGS required column FLAGS !')
                else: 
                    """
                    FLAGS description
                    1	aperture photometry is likely to be biased by neighboring sources or by more than 10% of bad pixels in any aperture
                    2	the object has been deblended
                    4	at least one object pixel is saturated
                    8	the isophotal footprint of the detected object is truncated (too close to an image boundary)
                    16	at least one photometric aperture is incomplete or corrupted (hitting buffer or memory limits)
                    32	the isophotal footprint is incomplete or corrupted (hitting buffer or memory limits)
                    64	a memory overflow occurred during deblending
                    128	a memory overflow occurred during extraction
                    # NOTE FLAGS == 0 is a typical constrain for getting isolated & non-saturated sources.
                    """
                    AstSEx = AstSEx[np.in1d(AstSEx['FLAGS'], ONLY_FLAGS)]    
                    Modify_AstSEx = True
            
            # ** e. Remove Boundary Sources
            if XBoundary != 0.0 or XBoundary != 0.0:
                NX, NY = int(phr_obj['NAXIS1']), int(phr_obj['NAXIS2'])
                _XY = np.array([AstSEx['X_IMAGE'], AstSEx['Y_IMAGE']]).T
                InnerMask = np.logical_and.reduce((_XY[:, 0] > XBoundary + 0.5, \
                                                   _XY[:, 0] < NX - XBoundary + 0.5, \
                                                   _XY[:, 1] > YBoundary + 0.5, \
                                                   _XY[:, 1] < NY - YBoundary + 0.5))
                AstSEx = AstSEx[InnerMask]
                Modify_AstSEx = True
            
            # ** f. Only preserve the sources matched to the quest coordinates
            if Coor4Match == 'XY_':
                Xcoln_4Match, Ycoln_4Match = 'X_IMAGE', 'Y_IMAGE'
                RAcoln_4Match, DECcoln_4Match = 'X_WORLD', 'Y_WORLD'

            if Coor4Match == 'XYWIN_':
                assert 'XWIN_IMAGE' in Param_lst
                assert 'YWIN_IMAGE' in Param_lst
                Xcoln_4Match, Ycoln_4Match = 'XWIN_IMAGE', 'YWIN_IMAGE'
                RAcoln_4Match, DECcoln_4Match = 'XWIN_WORLD', 'YWIN_WORLD'

            if XY_Quest is not None:
                if RD_Quest is not None:
                    sys.exit('MeLOn ERROR: Please feed only one type XY_Quest / RD_Quest !')
            if RD_Quest is not None:
                if not AddRD:
                    sys.exit('MeLOn ERROR: RD_Quest requires AddRD !')

            Symm = None
            if XY_Quest is not None:
                _XY = np.array([AstSEx[Xcoln_4Match], AstSEx[Ycoln_4Match]]).T
                Symm = Symmetric_Match.SM(POA=XY_Quest, POB=_XY, tol=Match_xytol)
            if RD_Quest is not None:
                _RD = np.array([AstSEx[RAcoln_4Match], AstSEx[DECcoln_4Match]]).T
                Symm = Sky_Symmetric_Match.SSM(RD_A=RD_Quest, RD_B=_RD, tol=Match_rdtol)
            
            if Symm is not None:
                AstSEx = AstSEx[Symm[:, 1]]
                AstSEx.add_column(Column(Symm[:, 0], name='QuestIndex'))
                Modify_AstSEx = True
            
            # ** g. ADD-COLUMN Stamp
            if StampImgSize is not None:
                _XY = np.array([AstSEx['X_IMAGE'], AstSEx['Y_IMAGE']]).T
                PixA_StpLst = Stamp_Generator.SG(FITS_obj=FITS_obj, StampImgSize=StampImgSize, \
                    Coordinates=_XY, CoorType='Image', AutoFill='Nan', MDIR=None)[0]
                AstSEx.add_column(Column(PixA_StpLst, name='Stamp'))
                Modify_AstSEx = True
            
            # ** UPDATE the file FITS_SExCat
            if MDIR is not None and Modify_AstSEx:
                hdl = fits.open(FITS_SExCat)
                FITS_tmp = ''.join([TDIR, '/TMP_SEX_%s' %FNAME])
                AstSEx.write(FITS_tmp, overwrite=True)
                hdl_tmp = fits.open(FITS_tmp)
                if tbhdu == 2: 
                    fits.HDUList([hdl[0], hdl[1], hdl_tmp[1]]).writeto(FITS_SExCat, overwrite=True)
                if tbhdu == 1: 
                    fits.HDUList([hdl[0], hdl_tmp[1]]).writeto(FITS_SExCat, overwrite=True)
                hdl.close()
                hdl_tmp.close()
                os.system('rm %s' %FITS_tmp)

        # ** REMOVE temporary directory
        if MDIR is None: 
            os.system('rm -rf %s' %TDIR)

        return AstSEx, PixA_SExCheckLst, FITS_SExCat, FITS_SExCheckLst
