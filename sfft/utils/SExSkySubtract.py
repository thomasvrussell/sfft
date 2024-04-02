import sep
import warnings
import numpy as np
from astropy.io import fits
from scipy.stats import iqr
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
# version: Feb 4, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class SEx_SkySubtract:
    @staticmethod
    def SSS(FITS_obj, FITS_skysub=None, FITS_sky=None, FITS_skyrms=None, PIXEL_SCALE=1.0, SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', \
        SATUR_DEFAULT=100000., BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, VERBOSE_LEVEL=2,
        MDIR=None):

        """
        # Inputs & Outputs:

        -FITS_obj []                    # FITS file path of the input image 
 
        -FITS_skysub [None]             # FITS file path of the output sky-subtracted image

        -FITS_sky [None]                # FITS file path of the output sky image
        
        -FITS_skyrms [None]             # FITS file path of the output sky RMS image

        -PIXEL_SCALE [1.0]              # Pixel scale of image in arcsec/px. 

        -ESATUR_KEY ['ESATUR']          # Keyword for the effective saturation level of sky-subtracted image
                                        # P.S. the value will be saved in the primary header of -FITS_skysub
                            
        -SATUR_DEFAULT [100000]         # Default saturation value if keyword not available in FITS header. 

        # Configurations for SExtractor:
        
        -SATUR_KEY ['SATURATE']         # SExtractor Parameter SATUR_KEY
                                        # i.e., keyword of the saturation level in the input image header

        -BACK_SIZE [64]                 # SExtractor Parameter BACK_SIZE

        -BACK_FILTERSIZE [3]            # SExtractor Parameter BACK_FILTERSIZE

        -DETECT_THRESH [1.5]            # SExtractor Parameter DETECT_THRESH

        -DETECT_MINAREA [5]             # SExtractor Parameter DETECT_MINAREA
        
        -DETECT_MAXAREA [0]             # SExtractor Parameter DETECT_MAXAREA

        # Miscellaneous
        
        -VERBOSE_LEVEL [2]              # The level of verbosity, can be [0, 1, 2]
                                        # 0/1/2: QUIET/NORMAL/FULL mode
        
        -MDIR [None]                    # Parent Directory for output files
                                        # PYSEx will generate a child directory with a random name under the paraent directory 
                                        # all output files are stored in the child directory

        # Returns:

            SKYDIP                      # The flux peak of the sky image (outliers rejected)

            SKYPEAK                     # The flux dip of the sky image (outliers rejected)
            
            PixA_skysub                 # Pixel Array of the sky-subtracted image 
            
            PixA_sky                    # Pixel Array of the sky image 
            
            PixA_skyrms                 # Pixel Array of the sky RMS image 

        """

        # * Generate SExtractor OBJECT-MASK
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # NOTE: GAIN, SATURATE, ANALYSIS_THRESH, DEBLEND_MINCONT, BACKPHOTO_TYPE do not affect the detection mask.
            DETECT_MASK = PY_SEx.PS(FITS_obj=FITS_obj, SExParam=['X_IMAGE', 'Y_IMAGE'], GAIN_KEY='PHGAIN', SATUR_KEY=SATUR_KEY, \
                BACK_TYPE='AUTO', BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, DETECT_THRESH=DETECT_THRESH, \
                ANALYSIS_THRESH=1.5, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, DEBLEND_MINCONT=0.005, \
                BACKPHOTO_TYPE='GLOBAL', CHECKIMAGE_TYPE='OBJECTS', MDIR=MDIR, VERBOSE_LEVEL=VERBOSE_LEVEL)[1][0].astype(bool)
        
        # * Extract SExtractor SKY-MAP from the Unmasked Image
        PixA_obj = fits.getdata(FITS_obj, ext=0).T
        _PixA = PixA_obj.astype(np.float64, copy=True)    # default copy=True, just to emphasize
        _PixA[DETECT_MASK] = np.nan
        if not _PixA.flags['C_CONTIGUOUS']: _PixA = np.ascontiguousarray(_PixA)

        # NOTE: here we use faster sep package instead of SExtractor.
        sepbkg = sep.Background(_PixA, bw=BACK_SIZE, bh=BACK_SIZE, fw=BACK_FILTERSIZE, fh=BACK_FILTERSIZE)
        PixA_sky, PixA_skyrms = sepbkg.back(), sepbkg.rms()
        PixA_skysub = PixA_obj - PixA_sky

        # * Make simple statistics for the SKY-MAP
        Q1 = np.percentile(PixA_sky, 25)
        Q3 = np.percentile(PixA_sky, 75)
        IQR = iqr(PixA_sky)
        SKYDIP = Q1 - 1.5*IQR    # outlier rejected dip
        SKYPEAK = Q3 + 1.5*IQR   # outlier rejected peak
        
        if FITS_skysub is not None:
            with fits.open(FITS_obj) as hdl:
                hdl[0].header['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
                hdl[0].header['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
                try:
                    ESATUR = float(hdl[0].header[SATUR_KEY]) - SKYPEAK    # use a conservative value
                except KeyError: 
                    ESATUR = float(SATUR_DEFAULT - SKYPEAK)
                hdl[0].header[ESATUR_KEY] = (ESATUR, 'MeLOn: Effective SATURATE after SEx-SKY-SUB')
                hdl[0].data[:, :] = PixA_skysub.T
                hdl.writeto(FITS_skysub, overwrite=True)
        
        if FITS_sky is not None:
            with fits.open(FITS_obj) as hdl:
                hdl[0].header['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
                hdl[0].header['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
                hdl[0].data[:, :] = PixA_sky.T
                hdl.writeto(FITS_sky, overwrite=True)
        
        if FITS_skyrms is not None:
            with fits.open(FITS_obj) as hdl:
                hdl[0].header['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
                hdl[0].header['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
                hdl[0].data[:, :] = PixA_skyrms.T
                hdl.writeto(FITS_skyrms, overwrite=True)

        return SKYDIP, SKYPEAK, PixA_skysub, PixA_sky, PixA_skyrms
