import sys
import numpy as np
import os.path as pa
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from sfft.utils.ReadWCS import Read_WCS
# version: Feb 6, 2023

# improved by Lauren Aldoroty (Duke Univ.)
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class Stamp_Generator:
    @staticmethod
    def SG(FITS_obj=None, EXTINDEX=0, EXTNAME=None, COORD=None, COORD_TYPE='IMAGE', \
        STAMP_IMGSIZE=None, FILL_VALUE=np.nan, FITS_StpLst=None, VERBOSE_LEVEL=2):

        """
        # MeLOn Notes
        # @ Stamp Generator
        # * Remarks on the 2 basic coordinate systems
        #   A. Fortran & FITS ( ds9 & SEx & WCS ) version
        #      @ Matrix Index start from 1. Thus the first pixel of a matrix is < r_F, c_F > = < 1, 1 >
        #      @ [Basic Consistence Rule in Fortran]
        #        Corresponding xy-Coordinate System: Pixel < r_F, c_F > has Center with xy-coordinate ( x_F, y_F ) = ( r_F, c_F )
        #        NOTE the origin point this system is located at the center point of the pixel < r_F, c_F > = < 0, 0 >, 
        #             which is not physical existed.
        #      @ [Basic inclusion Rule in Fortran]
        #        Given Coordinate is enclosed in which pixel ?
        #        < r_F, c_F > = < int(x_F + 0.5), int(y_F + 0.5) >
        #
        #   B. Numpy & C version
        #      @ Matrix Index start from 0. Thus the first pixel of a matrix is < r_C, c_C > = < 0, 0 >
        #      @ [Basic Consistence Rule in C]
        #        Corresponding xy-Coordinate System: Pixel < r_C, c_C > has Center with xy-coordinate ( x_C, y_C ) = ( r_C, c_C )
        #        NOTE the origin point this system is located at the center point of the pixel < r_C, c_C > = < 0, 0 >, 
        #             which is just the first pixel.
        #      @ [Basic inclusion Rule in C]
        #        Given Coordinate is enclosed in which pixel ?
        #        Hold Pixel < r_C, c_C > = < int(x_C + 0.5), int(y_C + 0.5) >
        #        (Hold Pixel has solid boundary at bottom and left side, and dashed boundary at upper and right side.)
        #   
        #   P.S. < r_F, c_F > = < r_C + 1, c_C + 1 > and < x_F, y_F > = < x_C + 1, y_C + 1 >
        #
        # * Our MeLOn Convention
        #   In our context, we always use C Matrix Index System with Fortran Coordinate System!
        #   henceforth, Let r, c means r_C, c_C and x, y indicate x_F, y_F.
        #
        #   @ [Basic Consistence Rule in Our Convention]
        #     Pixel < r, c > has Center with xy-coordinate (x, y) = (r + 1.0, c + 1.0)      
        #   @ [Basic Inclusion Rule in Our Convention]
        #     Hold Pixel < r, c > = < int(x - 0.5), int(y - 0.5) >
        #   @ Image Center has xy-coordinate (x, y) = (NX/2 + 0.5, NY/2 + 0.5)
        #   @ Image valid domain can be defined as,
        #     Without boundary: 0 <= r, c < NX, NY
        #                       0.5 <= x, y < NX + 0.5, NY + 0.5 
        #     With boundary: NBX, NBY <= r, c < NX - NBX, NY - NBY  
        #                    NBX + 0.5, NBY + 0.5 <= x, y < NX - NBX + 0.5, NY - NBY + 0.5
        #
        # * The convertion function in Astropy
        #   Both w.all_pix2world and w.all_world2pix has an argument called origin 
        #   origin = 1, Fortran xy-coordinate   |   This is what we always use!
        #   origin = 0, C xy-coordinate
        #
        # * Stamp-Procedure for digital image
        #   Given (x, y) ----- Enclosed in Pixel < r, c > = < ⌊x - 0.5⌋,  ⌊y - 0.5⌋ >
        #   Stamp(PixA, x, y, P, Q) = PixA[r - ⌊P/2⌋: r + ⌈P/2⌉, c - ⌊Q/2⌋: c + ⌈Q/2⌉]
        #
        # * Equivalent Stamp-Procedure for given row & column range
        #   Given RowRange = (r0, r1) and ColRange = (c0, c1)
        #   Let P = r1 - r0 and Q = c1 - c0, then 
        #   PixA[r0: r1, c0: c1] = Stamp(PixA, x, y, P, Q) 
        #   where x = r0 + ⌊P/2⌋ + 1.0 and y = c0 + ⌊Q/2⌋ + 1.0
        #
        # * Might be right, but haven't been Checked yet.
        #   PixA[E: -E, F: -F] = Stamp(PixA, U/2, V/2, U-2*E, V-2*F), where U, V = PixA.shape
        #   Stamp(PixA, x, y, P, Q)[E: -E, F: -F] = Stamp(PixA, x, y, P-E, Q-F)
        #
        # * Additional Remarks 
        #   a. Only Support Single-Extension FITS file currently.
        #   b. The Boundary Pixels could be filled with np.nan | median vaule | given value 
        #   c. Here we just care the essential cases with COORD_TYPE='IMAGE' since other 'WORLD' cases could be transformed.
        #   d. Cutout2D size parameter has the form of (ny, nx), however, the required stamp with STAMP_IMGSIZE 
        #      has a form of (NSX, NSY). We require the input of Cutout2D should only be < r, c >, even this 
        #      function actually also support float values as inputs.
        #   e. We should note NAXIS in the header would be automatic updated to adjust the image size.
        #   f. For general purpose, We do use the astropy generated wcs for the stamps.
        #      Alternatively, a more naive method is employed with correction of CRPIX1 & CRPIX2
        #      which is reasonable at least for TPV and SIP transformation, where the equations 
        #      are only dependent on the relative pixel coordinate, that it, the pixel-vector.   
        #      Ref https://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html  
        #      Ref https://fits.gsfc.nasa.gov/registry/sip/SIP_distortion_v1_0.pdf
        #
        """

        NSX, NSY = STAMP_IMGSIZE
        assert len(COORD.shape) == 2 and COORD.shape[1] == 2
        NPOS = COORD.shape[0]
        
        if EXTNAME is not None:
            hdr_obj = fits.getheader(FITS_obj, extname=EXTNAME)
            data_obj = fits.getdata(FITS_obj, extname=EXTNAME)
        else: 
            hdr_obj = fits.getheader(FITS_obj, ext=EXTINDEX)
            data_obj = fits.getdata(FITS_obj, ext=EXTINDEX)
        dtype_obj = data_obj.dtype
        
        # * read positions
        if np.issubdtype(COORD.dtype, np.floating):
            UCOORD = COORD
        elif np.issubdtype(COORD.dtype, np.integer):
            UCOORD = COORD.astype(float)
            if VERBOSE_LEVEL in [2]:
                _warn_message = 'COORD has integer data type, covert to float automatically!'
                print('MeLOn WARNING: %s' %_warn_message)
        else: 
            _error_message = 'WRONG data type for COORD!'
            sys.exit('MeLOn ERROR: %s' %_error_message)
        
        assert COORD_TYPE in ['IMAGE', 'WORLD']
        if COORD_TYPE == 'IMAGE':
            positions = (UCOORD - 0.5).astype(int)
        if COORD_TYPE == 'WORLD':
            w_obj = Read_WCS.RW(hdr_obj, VERBOSE_LEVEL=VERBOSE_LEVEL)
            positions = (w_obj.all_world2pix(UCOORD, 1) - 0.5).astype(int)

        # * read fill_value, as same type with data
        if type(FILL_VALUE) == float:
            fv = FILL_VALUE
            if np.issubdtype(dtype_obj, np.integer):
                fv = np.array([fv], dtype=dtype_obj)[0]
                if VERBOSE_LEVEL in [1, 2]:
                    _warn_message = 'float FILL_VALUE is coverted to integer automatically!'
                    print('MeLOn WARNING: %s' %_warn_message)
        
        elif type(FILL_VALUE) == int:
            fv = FILL_VALUE
            if np.issubdtype(dtype_obj, np.floating):
                fv = np.array([fv], dtype=dtype_obj)[0]
                if VERBOSE_LEVEL in [2]:
                    _warn_message = 'integer FILL_VALUE is coverted to float automatically!'
                    print('MeLOn WARNING: %s' %_warn_message)
        
        elif FILL_VALUE in ['MEDIAN', 'MEAN']:
            if FILL_VALUE == 'MEDIAN': fv = np.nanmedian(data_obj)
            if FILL_VALUE == 'MEAN': fv = np.nanmean(data_obj)

            if np.issubdtype(dtype_obj, np.integer) and type(fv) == float:
                fv = np.array([fv], dtype=dtype_obj)[0]
                if VERBOSE_LEVEL in [1, 2]:
                    _warn_message = 'float FILL_VALUE is coverted to integer automatically!'
                    print('MeLOn WARNING: %s' %_warn_message)
            
            if np.issubdtype(dtype_obj, np.floating) and type(fv) == int:
                fv = np.array([fv], dtype=dtype_obj)[0]
                if VERBOSE_LEVEL in [2]:
                    _warn_message = 'integer FILL_VALUE is coverted to float automatically!'
                    print('MeLOn WARNING: %s' %_warn_message)

        else:
            _error_message = 'FILL_VALUE is not recognized!'
            sys.exit('MeLOn ERROR: %s' %_error_message)
        
        # * create stamps with astropy function Cutout2D
        PixA_StpLst = []
        for i in range(NPOS):
            try:
                PixA_Stp = Cutout2D(data_obj, position=positions[i], size=(NSY, NSX), \
                    mode='partial', fill_value=fv).data.T
            except: PixA_Stp = fv * np.array(np.ones(STAMP_IMGSIZE), dtype=dtype_obj)
            PixA_StpLst.append(PixA_Stp)

        # * save stamps to FITS files with WCS updated
        if FITS_StpLst is not None:
            hdr_StpLst = [] 
            FNAME = pa.basename(FITS_obj)

            for i in range(NPOS):
                hdr_Stp = hdr_obj.copy()
                row_StpCent, col_StpCent = positions[i]
                row_StpO, col_StpO = row_StpCent - int(NSX/2), col_StpCent - int(NSY/2)
                try:
                    hdr_Stp['CRPIX1'] = float(hdr_Stp['CRPIX1']) - row_StpO
                    hdr_Stp['CRPIX2'] = float(hdr_Stp['CRPIX2']) - col_StpO
                except: pass
                hdr_Stp['COMMENT'] = 'Stamp created from %s ' %FNAME + \
                    'centred at row-colum [%d, %d] ' %(row_StpCent, col_StpCent) + \
                    'with stamp image size (%d %d)' %(NSX, NSY)
                hdr_StpLst.append(hdr_Stp)
            
            for i in range(NPOS):
                PixA_Stp, hdr_Stp = PixA_StpLst[i], hdr_StpLst[i]
                if isinstance(FITS_StpLst, list):
                    FITS_Stp = FITS_StpLst[i]
                elif isinstance(FITS_StpLst, str):
                    FITS_Stp = FITS_StpLst
                fits.HDUList([fits.PrimaryHDU(PixA_Stp.T, header=hdr_Stp)]).writeto(FITS_Stp, overwrite=True)

        return PixA_StpLst
