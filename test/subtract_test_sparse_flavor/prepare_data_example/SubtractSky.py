import os.path as pa
from sfft.utils.SExSkySubtract import SEx_SkySubtract
# sfft version: 1.3.4+

CDIR = pa.dirname(pa.abspath(__file__))
FITS_oTEMP = CDIR + '/inputs/c4d_180806_052738_ooi_i_v1.S20.fits'  # (not sky-subtracted) reference image before image resampling
FITS_oSCI = CDIR + '/inputs/c4d_180802_062754_ooi_i_v1.S20.fits'   # (not sky-subtracted) science image

for FITS_obj in [FITS_oTEMP, FITS_oSCI]:
    FITS_skysub = CDIR + '/outputs/%s.skysub.fits' %(pa.basename(FITS_obj)[:-5])
    SEx_SkySubtract.SSS(FITS_obj=FITS_obj, FITS_skysub=FITS_skysub, FITS_sky=None, FITS_skyrms=None, \
        SATUR_KEY='SATURATE', ESATUR_KEY='ESATUR', BACK_SIZE=64, BACK_FILTERSIZE=3, \
        DETECT_THRESH=1.5, DETECT_MINAREA=5, DETECT_MAXAREA=0)
print('\nMeLOn CheckPoint: SKY SUBTRACTION WITH SEXTRACTOR DONE!\n')
