import re
import os
import numpy as np
import os.path as pa
from astropy.io import fits
from astropy.wcs import WCS
from tempfile import mkdtemp
from astropy.nddata.utils import Cutout2D

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

"""
# MeLOn Notes
# @Stamp Generator
# * Remarks on the 2 basic Coordinate System
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
# NOTE < r_F, c_F > = < r_C + 1, c_C + 1 >
# NOTE < x_F, y_F > = < x_C + 1, y_C + 1 >
#
# * Our Convention
#   We generally use C Matrix Index System with Fortran Coordinate System !
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
#   origin = 1, Fortran xy-coordinate   |   This is what we generally use !
#   origin = 0, C xy-coordinate
#
# * Stamp-Procedure for digital image
#   Given (x, y) ----- Enclosed in Pixel < r, c > = < ⌊x - 0.5⌋,  ⌊y - 0.5⌋ >
#   Stamp(PixA, x, y, P, Q) = PixA[r - ⌊P/2⌋: r + ⌈P/2⌉, c - ⌊Q/2⌋: c + ⌈Q/2⌉]
#
# * Equivalent Stamp-Procedure for given row & column range
#   Given RowRange = (r0, r1) and ColRange = (c0, c1)
#   Let  P = r1 - r0 and Q = c1 - c0, then 
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
#   c. Here we just care the essential cases with CoorType='Image' since other 'World' cases could be transformed.
#   d. Cutout2D size parameter has the form of (ny, nx), However the required stamp with StampImgSize has a form of (NSX, NSY)
#      We require the input of Cutout2D should only be < r, c >, even this function actually support float values as input.
#   e. We should note NAXIS in the header would be automatic updated to adjust the image size.
#   f. For general purpose, We do use the astropy generated wcs for the stamps.
#      Alternatively, a more naive method is employed with correction of CRPIX1 & CRPIX2
#      which is reasonable at least for TPV and SIP transformation, where the equations 
#      are only dependent on the relative pixel coordinate, that it, the pixel-vector.   
#      Ref https://fits.gsfc.nasa.gov/registry/tpvwcs/tpv.html  
#      Ref https://fits.gsfc.nasa.gov/registry/sip/SIP_distortion_v1_0.pdf

"""

class Stamp_Generator:
    @staticmethod
    def SG(FITS_obj=None, StampImgSize=None, Coordinates=None, \
        CoorType='Image',  AutoFill='Nan', MDIR=None):

        NSX, NSY = StampImgSize
        NCoors = Coordinates.shape[0]
        Stamptags = np.arange(NCoors)
        hdl = fits.open(FITS_obj)
        hdr = hdl[0].header
        data = hdl[0].data

        # * Covert Coordinates to Image-Type 
        positions = (Coordinates - 0.5).astype(int)
        if CoorType == 'World':
            w_obj = WCS(hdr, hdl)
            positions = (w_obj.all_world2pix(Coordinates, 1) - 0.5).astype(int)
        hdl.close()

        # * Determin the auto fill value for boundary case
        fill_value = AutoFill
        if AutoFill is 'Median':
            fill_value = np.nanmedian(data)
        if AutoFill == 'Nan':
            fill_value = -65536    # FIXME In most cases, It works
        
        # * Make stamps with function Cutout2D
        #   FIXME collection of stamp arrays can be memory-consuming 
        PixA_StpLst = []
        Rsize = (NSY, NSX)
        for i in range(NCoors):
            try: PixA_Stp = Cutout2D(data, positions[i], Rsize, mode='partial', fill_value=fill_value).data.T.astype(float)
            except: PixA_Stp = fill_value * np.ones((StampImgSize[0], StampImgSize[1])).astype(float)
            if AutoFill == 'Nan': PixA_Stp[PixA_Stp == -65536] = np.nan
            PixA_StpLst.append(PixA_Stp)
        
        # * Make header for the stamps with modification of CRPIX1 CRPIX2 offset then save them
        #    NOTE I have checked the correctness of this operation.
        FITS_StpLst = []
        if MDIR is not None:
            FNAME = pa.basename(FITS_obj)
            TDIR = mkdtemp(suffix=None, prefix='SG_', dir=MDIR)
            hdr_StpLst = []
            for i in range(NCoors):
                hdr_Stp = hdr.copy()
                row_stpcent, col_stpcent = positions[i]
                row_stpo, col_stpo = row_stpcent - int(NSX/2), col_stpcent - int(NSY/2)
                try: 
                    hdr_Stp['CRPIX1'] = float(hdr_Stp['CRPIX1']) - row_stpo
                    hdr_Stp['CRPIX2'] = float(hdr_Stp['CRPIX2']) - col_stpo
                except: pass
                hdr_Stp['COMMENT'] = 'Make Stamp from %s ' %FNAME + \
                                     'with row_stpcent %d col_stpcent %d ' %(row_stpcent, col_stpcent) + \
                                     'with stamp image size %d %d' %(NSX, NSY)
                hdr_StpLst.append(hdr_Stp)
            
            for i in range(NCoors):
                PixA_Stp, hdr_Stp = PixA_StpLst[i], hdr_StpLst[i]
                FITS_Stp = pa.join(TDIR, '%s.Stp%d.fits' %(FNAME[:-5], i))
                fits.HDUList([fits.PrimaryHDU(PixA_Stp.T, header=hdr_Stp)]). writeto(FITS_Stp, overwrite=True)
                FITS_StpLst.append(FITS_Stp)
        
        return PixA_StpLst, FITS_StpLst

