import sep
import numpy as np
from astropy.io import fits
from scipy.stats import iqr
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class SEx_SkySubtract:
    @staticmethod
    def SSS(FITS_obj, FITS_skysub=None, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', \
        BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0):

        # * Generate SExtractor OBJECT-MASK
        PL = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'MAG_AUTO', 'MAGERR_AUTO']
        Mask_DET = PY_SEx.PS(FITS_obj=FITS_obj, PL=PL, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            BACK_TYPE='AUTO', BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, \
            DETECT_THRESH=DETECT_THRESH, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, \
            BACKPHOTO_TYPE='GLOBAL', CHECKIMAGE_TYPE='OBJECTS')[1][0].astype(bool)

        # * Extract SExtractor SKY-MAP from the Unmasked Image
        #   NOTE: here we use faster sep package other than SExtractor. 
        PixA_obj = fits.getdata(FITS_obj, ext=0).T
        _PixA = PixA_obj.astype(np.float64, copy=True)    # default copy=True, just to emphasize
        _PixA[Mask_DET] = np.nan
        if not _PixA.flags['C_CONTIGUOUS']: _PixA = np.ascontiguousarray(_PixA)
        PixA_sky = sep.Background(_PixA, bw=BACK_SIZE, bh=BACK_SIZE, \
            fw=BACK_FILTERSIZE, fh=BACK_FILTERSIZE).back()
        PixA_skysub = PixA_obj - PixA_sky

        # * Make simple statistics for the SKY-MAP
        Q1 = np.percentile(PixA_sky, 25)
        Q3 = np.percentile(PixA_sky, 55)
        IQR = iqr(PixA_sky)
        SKYDIP = Q1 - 1.5*IQR    # NOTE outlier rejected dip
        SKYPEAK = Q3 + 1.5*IQR   # NOTE outlier rejected peak

        if FITS_skysub is not None:
            with fits.open(FITS_obj) as hdl:
                hdl[0].header['SKYDIP'] = (SKYDIP, 'MeLOn: IQR-MINIMUM of SEx-SKY-MAP')
                hdl[0].header['SKYPEAK'] = (SKYPEAK, 'MeLOn: IQR-MAXIMUM of SEx-SKY-MAP')
                ESATUR = float(hdl[0].header['SATURATE']) - SKYPEAK  # NOTE a conservative value
                hdl[0].header['ESATUR'] = (ESATUR, 'MeLOn: Effective SATURATE after SEx-SKY-SUB')
                hdl[0].data[:, :] = PixA_skysub.T
                hdl.writeto(FITS_skysub, overwrite=True)

        return SKYDIP, SKYPEAK, PixA_sky, PixA_skysub
