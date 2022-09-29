import numpy as np
import os.path as pa
from sfft.utils.pyAstroMatic.PYSWarp import PY_SWarp
# sfft version: 1.3.4+

# NOTE: please run SubtractSky.py first.
CDIR = pa.dirname(pa.abspath(__file__))
FITS_TEMP = CDIR + '/outputs/c4d_180806_052738_ooi_i_v1.S20.skysub.fits'  # (sky-subtracted) reference image before image resampling
FITS_SCI = CDIR + '/outputs/c4d_180802_062754_ooi_i_v1.S20.skysub.fits'   # (sky-subtracted) science image
FITS_REF = FITS_TEMP[:-5] + '.resamp.fits'

PY_SWarp.PS(FITS_obj=FITS_TEMP, FITS_ref=FITS_SCI, FITS_resamp=FITS_REF, FITS_resamp_weight=None, \
    GAIN_KEY='GAIN', SATUR_KEY='ESATUR', OVERSAMPLING=1, RESAMPLING_TYPE='LANCZOS3', SUBTRACT_BACK='N', \
    FILL_VALUE=np.nan, WEIGHT_SUFFIX='.weight.fits', WRITE_XML='N', VERBOSE_TYPE='NORMAL')
print('\nMeLOn CheckPoint: IMAGE ALIGNMENT WITH SWARP DONE!\n')
