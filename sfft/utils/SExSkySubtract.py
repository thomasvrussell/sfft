import sep
import warnings
import numpy as np
from astropy.io import fits
from scipy.stats import iqr
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
# version: Apr 22, 2024

import sys
import logging
import multiprocessing
_logger = logging.getLogger(f'sfft')
if not _logger.hasHandlers():
    log_out = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter(f'[%(asctime)s - sfft - %(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    log_out.setFormatter(formatter)
    _logger.addHandler(log_out)
    _logger.setLevel(logging.DEBUG) # ERROR, WARNING, INFO, or DEBUG (in that order by increasing detail)

# improved by Lauren Aldoroty (Duke Univ.)
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class SEx_SkySubtract:
    @staticmethod
    def SSS(FITS_obj, FITS_skysub=None, FITS_sky=None, FITS_skyrms=None, FITS_detmask=None,
            SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', \
            BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
            VERBOSE_LEVEL=2, MDIR=None):

        """
        # Inputs & Outputs:

        -FITS_obj []                    # FITS file path of the input image

        -FITS_skysub [None]             # FITS file path of the output sky-subtracted image

        -FITS_sky [None]                # FITS file path of the output sky image

        -FITS_skyrms [None]             # FITS file path of the output sky RMS image

        -ESATUR_KEY ['ESATUR']          # Keyword for the effective saturation level of sky-subtracted image
                                        # P.S. the value will be saved in the primary header of -FITS_skysub

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

        procid = multiprocessing.current_process().pid

        # * Generate SExtractor OBJECT-MASK
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _logger.debug( f"Process {procid} running PY_SEx.PS on {FITS_obj}..." )
            # NOTE: GAIN, SATURATE, ANALYSIS_THRESH, DEBLEND_MINCONT, BACKPHOTO_TYPE do not affect the detection mask.
            DETECT_MASK = PY_SEx.PS(FITS_obj=FITS_obj, SExParam=['X_IMAGE', 'Y_IMAGE'], GAIN_KEY='PHGAIN', SATUR_KEY=SATUR_KEY,
                                    BACK_TYPE='AUTO', BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE,
                                    DETECT_THRESH=DETECT_THRESH, ANALYSIS_THRESH=1.5, DETECT_MINAREA=DETECT_MINAREA,
                                    DETECT_MAXAREA=DETECT_MAXAREA, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='GLOBAL',
                                    CHECKIMAGE_TYPE='OBJECTS', MDIR=MDIR, VERBOSE_LEVEL=VERBOSE_LEVEL,
                                    logger=_logger
                                    )[1][0].astype(bool)
            _logger.debug( f"...process {procid} done running PY_SEx.PS on {FITS_obj}..." )

        # * Extract SExtractor SKY-MAP from the Unmasked Image
        _logger.debug( f"Running fits.getdata({FITS_obj}, ext=0)..." )
        PixA_obj = fits.getdata(FITS_obj, ext=0).T
        _logger.debug( f"...done running fits.getdata({FITS_obj}, ext=0)." )
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

        with fits.open(FITS_obj) as hdl:
            FITS_obj_hdr = hdl[0].header
            FITS_obj_hdr['SKYRMS'] = ( np.median( PixA_skyrms ), 'MeLOn: Median of sky RMS' )

        if FITS_skysub is not None:
            hdr = FITS_obj_hdr.copy()
            hdr['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
            hdr['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
            if SATUR_KEY in hdr:
                ESATUR = float(hdr[SATUR_KEY]) - SKYPEAK    # use a conservative value
                hdr[ESATUR_KEY] = (ESATUR, 'MeLOn: Effective SATURATE after SEx-SKY-SUB')
            fits.writeto( FITS_skysub, PixA_skysub.T, hdr, overwrite=True )

        if FITS_sky is not None:
            hdr = FITS_obj_hdr.copy()
            hdr['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
            hdr['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
            fits.writeto(FITS_sky, PixA_sky.T, hdr, overwrite=True)

        if FITS_skyrms is not None:
            hdr = FITS_obj_hdr.copy()
            hdr['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
            hdr['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
            fits.writeto(FITS_skyrms, PixA_skyrms.T, hdr, overwrite=True)

        if FITS_detmask is not None:
            fits.writeto( FITS_detmask, DETECT_MASK.T.astype( np.uint8 ), overwrite=True )

        return SKYDIP, SKYPEAK, PixA_skysub, PixA_sky, PixA_skyrms
