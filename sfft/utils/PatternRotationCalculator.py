import numpy as np
from sfft.utils.ReadWCS import Read_WCS

__last_update__ = "2024-09-22"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.6.1"

class PatternRotation_Calculator:
    @staticmethod
    def PRC(hdr_obj, hdr_targ):
        """calculate the rotation angle for a pattern from the object frame to target frame

        Parmaeters
        ----------
          hdr_obj: astropy.io.fits.header.Header
            The WCS of the object frame is read from this header.  (The
            header doesn't need anything else.)  Doesn't have to
            actually be a astropy Header; it needs to be able to have
            its elements accessed like a dictionary, it needs to suppor
            the copy() method for a shallow copy, and it needs to be
            something that the astropy.wcs.WCS() constructor can take.

          hdr_targ:  astropy.io.fits.header.Header
            Like hdr_obj, but for the target.

        """
        def calculate_skyN_vector(wcshdr, x_start, y_start, shift_dec=1.0):
            w = Read_WCS.RW(wcshdr, VERBOSE_LEVEL=1)
            ra_start, dec_start = w.all_pix2world(np.array([[x_start, y_start]]), 1)[0]
            ra_end, dec_end = ra_start, dec_start + shift_dec/3600.0
            x_end, y_end = w.all_world2pix(np.array([[ra_end, dec_end]]), 1)[0]
            skyN_vector = np.array([x_end - x_start, y_end - y_start])
            return skyN_vector

        # Note: rotate_angle is counterclockwise, in degree
        def calculate_rotate_angle(vector_ref, vector_obj):
            rad = np.arctan2(np.cross(vector_ref, vector_obj), np.dot(vector_ref, vector_obj))
            rotate_angle = np.rad2deg(rad)
            if rotate_angle < 0.0: rotate_angle += 360.0
            return rotate_angle

        _phdr = hdr_obj
        _w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
        x0, y0 = 0.5 + int(_phdr['NAXIS1'])/2.0, 0.5 + int(_phdr['NAXIS2'])/2.0
        ra0, dec0 = _w.all_pix2world(np.array([[x0, y0]]), 1)[0]
        skyN_vector = calculate_skyN_vector(wcshdr=_phdr, x_start=x0, y_start=y0)

        _phdr = hdr_targ
        _w = Read_WCS.RW(_phdr, VERBOSE_LEVEL=1)
        x1, y1 = _w.all_world2pix(np.array([[ra0, dec0]]), 1)[0]
        skyN_vectorp = calculate_skyN_vector(wcshdr=_phdr, x_start=x1, y_start=y1)
        PATTERN_ROTATE_ANGLE = calculate_rotate_angle(vector_ref=skyN_vector, vector_obj=skyN_vectorp)

        return PATTERN_ROTATE_ANGLE
