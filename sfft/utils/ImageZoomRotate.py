import os
import math
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from tempfile import mkdtemp
from CudaResampling import Cuda_Resampling
#from sfft.utils.pyAstroMatic.PYSWarp import PY_SWarp

__last_update__ = "2024-09-11"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"

class Image_ZoomRotate:
    @staticmethod
    def IZR(PixA_obj, ZOOM_SCALE_X=1.0, ZOOM_SCALE_Y=1.0, OUTSIZE_PARIRY_X='UNCHANGED', OUTSIZE_PARIRY_Y='UNCHANGED', \
        PATTERN_ROTATE_ANGLE=0.0, RESAMPLING_TYPE='LANCZOS3', FILL_VALUE=0.0, VERBOSE_LEVEL=2):
        
        """
        # Remarks on Image Zoom & Rotate
        # [1] In this function, 'Zoom' means to change the spatial resolution (pixel size) of given image.
        #     That is to say, we would like to resample the original image on a new grid with different resolution.
        #     The arguments -ZOOM_SCALE_X and -ZOOM_SCALE_Y control the resacle factor of pixel size.
        #     E.g., ZOOM_SCALE_X = 1.2 means new pixel is 1.2 times larger than the original pixel along the X axis.
        #
        # [2] Throughout the image Zoom, the position of image center is always pinned.
        #     The Zoomed image frame has the smallest area which can fully cover the original frame.
        #     NOTE: the image Zoom would not alert the even/odd property of image size.
        # 
        # [3] This function also allows for an additional pattern rotation about the pinned image center.
        #     The argument -PATTERN_ROTATE_ANGLE is the rotate angle (counterclockwise, in deg).
        # 
        # [4] Given that the function is commonly used to manipulate PSF model, 
        #     here we set LANCZOS4 as default resampling method following PSFEx.
        #
        """

        # check inputs
        assert ZOOM_SCALE_X > 0.0
        assert ZOOM_SCALE_Y > 0.0
        assert 0.0 <= PATTERN_ROTATE_ANGLE < 360.0

        # create temporary directory
        TDIR = mkdtemp(suffix=None, prefix=None, dir=None)

        # * create a header with initialized wcs for the original image
        NAXIS1_ORI, NAXIS2_ORI = PixA_obj.shape                           # image original size
        CRPIX1_ORI, CRPIX2_ORI = 0.5+NAXIS1_ORI/2.0, 0.5+NAXIS2_ORI/2.0   # reference point at image center 
        CRVAL1_ORI, CRVAL2_ORI = 180.0, 0.0                               # trivial skycoord of the reference point
        CD1_1_ORI, CD1_2_ORI = 1.0/3600, 0.0                              # trivial pixel scale of the original image
        CD2_1_ORI, CD2_2_ORI = 0.0, 1.0/3600                              # trivial pixel scale of the original image

        wcsinfo_ORI = {'NAXIS': 2,
                       'NAXIS1': NAXIS1_ORI, 'NAXIS2': NAXIS2_ORI,
                       'CRPIX1': CRPIX1_ORI, 'CRPIX2': CRPIX2_ORI,
                       'CRVAL1': CRVAL1_ORI, 'CRVAL2': CRVAL2_ORI,
                       'CD1_1': CD1_1_ORI, 'CD1_2': CD1_2_ORI,
                       'CD2_1': CD2_1_ORI, 'CD2_2': CD2_2_ORI,
                       'CUNIT1': 'deg', 'CUNIT2': 'deg',
                       'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN'}

        # * save the original image with initialized wcs
        header_ORI = WCS(wcsinfo_ORI).to_header(relax=True)
        hdl_ORI = fits.HDUList([fits.PrimaryHDU(data=PixA_obj.T, header=header_ORI)])
        
        hdl_ORI[0].header['GAIN'] = (1.0, 'MeLOn: PlaceHolder')    
        hdl_ORI[0].header['SATURATE'] = (50000.0, 'MeLOn: PlaceHolder')
        hdl_ORI[0].header['EXPTIME'] = (60.0, 'MeLOn: PlaceHolder')
        hdl_ORI[0].header['MJD-OBS'] = (58849.0, 'MeLOn: PlaceHolder')
        
        FITS_ORI = TDIR + '/original_image.fits'
        hdl_ORI.writeto(FITS_ORI, overwrite=True)

        # * define the target wcs
        wcsinfo_TARG = wcsinfo_ORI.copy()

        # * determine the number of pixels of the zoomed image
        def NPIX_ORI2ZOOMED(NPIX_ORI, ZOOM_SCALE, OUTSIZE_PARIRY):
            if OUTSIZE_PARIRY == 'UNCHANGED':
                if NPIX_ORI % 2 == 0: OUTSIZE_UPARIRY = 'EVEN'
                if NPIX_ORI % 2 == 1: OUTSIZE_UPARIRY = 'ODD'
            if OUTSIZE_PARIRY == 'ODD': OUTSIZE_UPARIRY = 'ODD'
            if OUTSIZE_PARIRY == 'EVEN': OUTSIZE_UPARIRY = 'EVEN'
            HALFWIDTH_ORI = NPIX_ORI/2.0  # unit of original pixel
            if OUTSIZE_UPARIRY == 'EVEN':
                NPIX_ZOOMED = 2 * math.ceil(HALFWIDTH_ORI/ZOOM_SCALE)
            if OUTSIZE_UPARIRY == 'ODD':
                NPIX_ZOOMED = 2 * math.ceil((HALFWIDTH_ORI - ZOOM_SCALE/2.0)/ZOOM_SCALE) + 1
            return NPIX_ZOOMED

        NAXIS1_ZOOMED = NPIX_ORI2ZOOMED(NAXIS1_ORI, ZOOM_SCALE_X, OUTSIZE_PARIRY_X)
        NAXIS2_ZOOMED = NPIX_ORI2ZOOMED(NAXIS2_ORI, ZOOM_SCALE_Y, OUTSIZE_PARIRY_Y)
        
        # * give the coordinate change due to the frame transform from original to zoomed
        def COORD_ORI2ZOOMED(X_ORI, Y_ORI):
            # NOTE: X_ORI/Y_ORI can be a number or a 1d-array
            OFFSET_X = ZOOM_SCALE_X * (0.5 + NAXIS1_ZOOMED/2.0) - (0.5 + NAXIS1_ORI/2.0)   # unit of original pixel
            OFFSET_Y = ZOOM_SCALE_Y * (0.5 + NAXIS2_ZOOMED/2.0) - (0.5 + NAXIS2_ORI/2.0)   # unit of original pixel
            X_ZOOMED = (X_ORI + OFFSET_X) / ZOOM_SCALE_X
            Y_ZOOMED = (Y_ORI + OFFSET_Y) / ZOOM_SCALE_Y
            return X_ZOOMED, Y_ZOOMED

        # * modify wcs to adapt to the zoomed frame
        CRPIX1_ZOOMED, CRPIX2_ZOOMED = COORD_ORI2ZOOMED(CRPIX1_ORI, CRPIX2_ORI)
        CD1_1_ZOOMED, CD1_2_ZOOMED = ZOOM_SCALE_X * CD1_1_ORI, 0.0
        CD2_1_ZOOMED, CD2_2_ZOOMED = 0.0, ZOOM_SCALE_Y * CD2_2_ORI

        if ZOOM_SCALE_X != 1.0 or ZOOM_SCALE_Y != 1.0:
            if VERBOSE_LEVEL in [1, 2]: 
                print('MeLOn CheckPoint: Modify WCS to adapt to the Zoomed Frame.')
            wcsinfo_TARG['NAXIS1'], wcsinfo_TARG['NAXIS2'] = NAXIS1_ZOOMED, NAXIS2_ZOOMED
            wcsinfo_TARG['CRPIX1'], wcsinfo_TARG['CRPIX2'] = CRPIX1_ZOOMED, CRPIX2_ZOOMED
            wcsinfo_TARG['CD1_1'], wcsinfo_TARG['CD1_2'] = CD1_1_ZOOMED, CD1_2_ZOOMED
            wcsinfo_TARG['CD2_1'], wcsinfo_TARG['CD2_2'] = CD2_1_ZOOMED, CD2_2_ZOOMED
        else:
            if VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: NO IMAGE ZOOM!')
            assert np.allclose([wcsinfo_TARG['NAXIS1'], wcsinfo_TARG['NAXIS2']], [NAXIS1_ZOOMED, NAXIS2_ZOOMED])
            assert np.allclose([wcsinfo_TARG['CRPIX1'], wcsinfo_TARG['CRPIX2']], [CRPIX1_ZOOMED, CRPIX2_ZOOMED])
            assert np.allclose([wcsinfo_TARG['CD1_1'], wcsinfo_TARG['CD1_2']], [CD1_1_ZOOMED, CD1_2_ZOOMED])
            assert np.allclose([wcsinfo_TARG['CD2_1'], wcsinfo_TARG['CD2_2']], [CD2_1_ZOOMED, CD2_2_ZOOMED])

        alpha = np.deg2rad(PATTERN_ROTATE_ANGLE)
        RotMAT = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        RotMAT_inv = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])

        # * give the coordinate change due to pattern rotate on zoomed
        def COORD_ZOOMED2ROTATED(X_ZOOMED, Y_ZOOMED):
            # NOTE: X_ZOOMED/Y_ZOOMED can be a number or a 1d-array
            cX_ZOOMED = X_ZOOMED - CRPIX1_ZOOMED
            cY_ZOOMED = Y_ZOOMED - CRPIX2_ZOOMED
            cCOORD_ZOOMED = np.array([cX_ZOOMED, cY_ZOOMED]).reshape((2, -1))
            cCOORD_ROTATED = np.dot(RotMAT, cCOORD_ZOOMED)
            if cCOORD_ROTATED.shape[1] == 1: cX_ROTATED, cY_ROTATED = cCOORD_ROTATED[:, 0]
            else: cX_ROTATED, cY_ROTATED = cCOORD_ROTATED
            X_ROTATED = cX_ROTATED + CRPIX1_ZOOMED
            Y_ROTATED = cY_ROTATED + CRPIX2_ZOOMED
            return X_ROTATED, Y_ROTATED

        # * further modify wcs to adapt to the rotated pattern
        CDMAT_ZOOMED = np.array([[CD1_1_ZOOMED, CD1_2_ZOOMED], [CD2_1_ZOOMED, CD2_2_ZOOMED]])
        CDMAT_ROTATED = np.dot(CDMAT_ZOOMED, RotMAT_inv)
        CD1_1_ROTATED, CD1_2_ROTATED = CDMAT_ROTATED[0, :]
        CD2_1_ROTATED, CD2_2_ROTATED = CDMAT_ROTATED[1, :]

        if PATTERN_ROTATE_ANGLE != 0.0:
            if VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: Modify WCS to adapt to the Rotated Frame.')
            wcsinfo_TARG['CD1_1'], wcsinfo_TARG['CD1_2'] = CD1_1_ROTATED, CD1_2_ROTATED
            wcsinfo_TARG['CD2_1'], wcsinfo_TARG['CD2_2'] = CD2_1_ROTATED, CD2_2_ROTATED
        else:
            if VERBOSE_LEVEL in [1, 2]:
                print('MeLOn CheckPoint: NO IMAGE ROTATION!')
            assert np.allclose([wcsinfo_TARG['CD1_1'], wcsinfo_TARG['CD1_2']], [CD1_1_ROTATED, CD1_2_ROTATED])
            assert np.allclose([wcsinfo_TARG['CD2_1'], wcsinfo_TARG['CD2_2']], [CD2_1_ROTATED, CD2_2_ROTATED])

        # * give the throughout coordinate change from original to rotated.
        def func_coordtrans(X_ORI, Y_ORI):
            X_ZOOMED, Y_ZOOMED = COORD_ORI2ZOOMED(X_ORI, Y_ORI)
            X_ROTATED, Y_ROTATED = COORD_ZOOMED2ROTATED(X_ZOOMED, Y_ZOOMED)
            return X_ROTATED, Y_ROTATED

        # * save an empty image with target wcs
        header_TARG = WCS(wcsinfo_TARG).to_header(relax=True)
        PixA_empty = np.zeros((wcsinfo_TARG['NAXIS1'], wcsinfo_TARG['NAXIS2'])).astype(float)
        hdl_TARG = fits.HDUList([fits.PrimaryHDU(data=PixA_empty.T, header=header_TARG)])
        
        hdl_TARG[0].header['GAIN'] = (1.0, 'MeLOn: PlaceHolder')    
        hdl_TARG[0].header['SATURATE'] = (50000.0, 'MeLOn: PlaceHolder')   
        hdl_TARG[0].header['EXPTIME'] = (60.0, 'MeLOn: PlaceHolder')
        hdl_TARG[0].header['MJD-OBS'] = (58849.0, 'MeLOn: PlaceHolder')

        FITS_TARG = TDIR + '/target_empty_image.fits'
        hdl_TARG.writeto(FITS_TARG, overwrite=True)

        # * run SWarp to perform the image resampling
        FITS_resamp = TDIR + '/resampled_image.fits'
        PixA_resamp = Cuda_Resampling.CR(FITS_obj=FITS_ORI, FITS_targ=FITS_TARG, FITS_resamp=FITS_resamp, 
            METHOD=RESAMPLING_TYPE, FILL_ZEROPIX=False, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        PixA_resamp[np.isnan(PixA_resamp)] = FILL_VALUE
        with fits.open(FITS_TARG) as hdl:
            hdl[0].data[:, :] = PixA_resamp.T
            hdl.writeto(FITS_resamp, overwrite=True)
        
        # PY_SWarp.PS(FITS_obj=FITS_ORI, FITS_ref=FITS_TARG, FITS_resamp=FITS_resamp, \
        #     GAIN_KEY='GAIN', SATUR_KEY='SATURATE', OVERSAMPLING=1, RESAMPLING_TYPE=RESAMPLING_TYPE, \
        #     SUBTRACT_BACK='N', FILL_VALUE=FILL_VALUE, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        PixA_resamp = fits.getdata(FITS_resamp, ext=0).T
        os.system('rm -rf %s' %TDIR)
        
        return PixA_resamp, func_coordtrans
