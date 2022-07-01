import re
import os
import sys
import warnings
import numpy as np
import os.path as pa
from astropy.io import fits
from astropy.wcs import WCS
from tempfile import mkdtemp
from sfft.utils.pyAstroMatic.AMConfigMaker import AMConfig_Maker

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.1"

"""
# MeLOn Notes
# @ Python Version of SWarp
#  a) PYSWarp Purpose
#     a. The current python version only implement Resampling of single SEF-FITS image.
#     b. Note SWarp can also do the simple combination procedure after Resampling.
#
#  b) SWarp destination-frame
#     @ Default-NE Mode
#       The output WCS-frame simply use canonical N-E orientation, i.e., NE-WCS-Frame. 
#       NOTE this mode is not supported in our function.
#     @ Given-REF Mode
#       The output WCS-frame (including image size) is determined by the given FITS_ref, 
#       we ganna to prepare a .head file, coincident name with FITS_Re, to store REF-WCS-Frame.
# 
#  c) SWarp resulting header 
#     a. SWarp will automatically calculate Maximum equivalent Gain (to GAIN) & Exposure (to EXPTIME)
#        with additional Saturation (SATURATE) value, into resulting product's header. 
#        However, they are trivial for our non-combination purpose.
#     b. In our function, the resulting product's header is constructed by removing 
#        wcs keywords in FITS_obj and adding those in FITS_ref. In additional, 
#        we will record the SWarp parameters used in the function.
#     c. SWarp will automatically record the processing time in resulting header.
#
#  d) Tips on resampling
#     NOTE: LANCZOS4 is relatively time-consuming
#     NOTE: oversampling may cause higher pixel correlation

"""

class PY_SWarp:
    @staticmethod
    def PS(FITS_obj, FITS_ref, FITS_Re=None, FITS_WRe=None, \
        GAIN_KEY='GAIN', SATUR_KEY='SATURATE', EXPTIME_KEY='EXPTIME', \
        oversampling=1, method='LANCZOS3', SUBTRACT_BACK='N', Fillvalue=None, \
        WEIGHT_SUFFIX='.weight.fits', WRITE_XML='N', VERBOSE_TYPE='NORMAL'):

        # * Check Process for FITS_obj header
        phr_obj = fits.getheader(FITS_obj, ext=0)
        BASICKeys = [GAIN_KEY, SATUR_KEY, EXPTIME_KEY]
        for key in BASICKeys:
            if key not in phr_obj: 
                sys.exit('MeLOn ERROR: Some SWarp required keywords are NOT FOUND !')

        # * Make Directory as workplace of PYSwarp operation
        TDIR = mkdtemp(suffix=None, prefix=None, dir=None)

        # * Make Swarp configuration file in TDIR.
        ConfigDict = {}
        ConfigDict['GAIN_KEYWORD'] = '%s' %GAIN_KEY
        ConfigDict['SATLEV_KEYWORD'] = '%s' %SATUR_KEY

        ConfigDict['OVERSAMPLING'] = '%d' %oversampling
        ConfigDict['RESAMPLING_TYPE'] = '%s' %method
        ConfigDict['SUBTRACT_BACK'] = '%s' %SUBTRACT_BACK
        ConfigDict['COMBINE_TYPE'] = 'MEDIAN'  # FIXME

        ConfigDict['WEIGHT_SUFFIX'] = '%s' %WEIGHT_SUFFIX
        ConfigDict['WRITE_XML'] = '%s' %WRITE_XML
        ConfigDict['VERBOSE_TYPE'] = '%s' %VERBOSE_TYPE
        
        swarpconfig_path = AMConfig_Maker.AMCM(MDIR=TDIR, \
            AstroMatic_KEY='swarp', ConfigDict=ConfigDict, tag='PYSWarp')

        # * Make temporary FITS_tRe & FIT_tWRe in DIR
        FITS_tRe = TDIR + '/%s.tmp_resamp.fits' %pa.basename(FITS_obj)[:-5]
        FITS_tWRe = TDIR + '/%s.tmp_wresamp.fits' %pa.basename(FITS_obj)[:-5]

        # * Build .head file from FITS_ref
        #    NOTE: SIP distorsion terms require w.to_header(relax=True)

        phr_ref = fits.getheader(FITS_ref, ext=0)
        w_ref = WCS(phr_ref)            
        _headfile = FITS_tRe[:-5] + '.head'
        wcshdr_ref = w_ref.to_header(relax=True)
        wcshdr_ref['BITPIX'] = phr_ref['BITPIX']
        wcshdr_ref['NAXIS'] = phr_ref['NAXIS']            
        wcshdr_ref['NAXIS1'] = phr_ref['NAXIS1']
        wcshdr_ref['NAXIS2'] = phr_ref['NAXIS2']
        wcshdr_ref.tofile(_headfile)

        # * Trigger SWarp 
        os.system('cd %s && swarp %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -c %s' \
            %(TDIR, FITS_obj, FITS_tRe, FITS_tWRe, swarpconfig_path))

        # * Make a combined header for resulting products
        def combine_header(hdr_base, hdr_wcs):
            
            """
            # Remarks on the wcs convention discrepancy between FITS header and Astropy
            # Case A:
            #    FITS HEADER: 
            #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
            #        CD1_1 = ...           CD1_2 = ...
            #        CD2_1 = ...           CD2_2 = ... 
            #        NOTE: NO DISTORTION TERMS!
            #    
            #    Astropy:        
            #        # print(WCS(phr).to_header())
            #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
            #        PC1_1 = ...           PC1_2 = ...
            #        PC2_1 = ...           PC2_2 = ... 
            #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
            
            # Case B:
            #    FITS HEADER: 
            #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
            #        CD1_1 = ...           CD1_2 = ...
            #        CD2_1 = ...           CD2_2 = ... 
            #        PV1_0 = ...           PV1_1 = ...
            #        ...
            #    
            #    Astropy:        
            #        # print(WCS(phr).to_header())
            #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
            #        PC1_1 = ...           PC1_2 = ...
            #        PC2_1 = ...           PC2_2 = ... 
            #        PV1_0 = ...           PV1_1 = ...
            #        ...
            #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
            #
            # Case B': a possible common situation that causes Astropy.WCS incompatibility.
            #    FITS HEADER:
            #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
            #        CD1_1 = ...           CD1_2 = ...
            #        CD2_1 = ...           CD2_2 = ... 
            #        PV1_0 = ...           PV1_1 = ...
            #        ...
            #     
            #    NOTE: One need to correct as follows to avoid the error
            #          phr['CTYPE1'] = 'RA---TPV'
            #          phr['CTYPE1'] = 'DEC--TPV'
            #          w = WCS(phr)
            #
            #    Astropy:        
            #        # print(w.to_header())
            #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
            #        PC1_1 = ...           PC1_2 = ...
            #        PC2_1 = ...           PC2_2 = ... 
            #        PV1_0 = ...           PV1_1 = ...
            #        ...
            #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
            
            # Case C:
            #    FITS HEADER: 
            #        CTYPE1 = 'RA---TAN-SIP'   CTYPE2 = 'DEC--TAN-SIP'
            #        CD1_1 = ...           CD1_2 = ...
            #        CD2_1 = ...           CD2_2 = ... 
            #        A_ORDER = 2           
            #        A_0_0 = ...           A_0_1 = ...
            #        ...
            #        B_ORDER = 2           
            #        B_0_0 = ...           B_0_1 = ...
            #        ...
            #        AP_ORDER = 2           
            #        AP_0_0 = ...          AP_0_1 = ...
            #        ...
            #        BP_ORDER = 2           
            #        BP_0_0 = ...          BP_0_1 = ...
            #        ...
            #    
            #    Astropy:        
            #        # print(WCS(phr).to_header(relax=True))
            #        CTYPE1 = 'RA---TAN-SIP'   CTYPE2 = 'DEC--TAN-SIP'
            #        PC1_1 = ...           PC1_2 = ...
            #        PC2_1 = ...           PC2_2 = ... 
            #        A_ORDER = 2           
            #        A_0_0 = ...           A_0_1 = ...
            #        ...
            #        B_ORDER = 2           
            #        B_0_0 = ...           B_0_1 = ...
            #        ...
            #        AP_ORDER = 2           
            #        AP_0_0 = ...          AP_0_1 = ...
            #        ...
            #        BP_ORDER = 2           
            #        BP_0_0 = ...          BP_0_1 = ...
            #        ...
            #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
            #        NOTE: here remember use relax=True!
            
            # NOTE: You may also use PC1_1 in FITS HEADER instead of CD1_1 in above cases.
            #       It is readable, however, we do not recommend this.
            """

            def read_wcs(hdr):
                if hdr['CTYPE1'] == 'RA---TAN' and 'PV1_0' in hdr:
                    _hdr = hdr.copy()
                    _hdr['CTYPE1'] = 'RA---TPV'
                    _hdr['CTYPE2'] = 'DEC--TPV'
                    w = WCS(_hdr)
                else: w = WCS(hdr)
                return w

            # ** Remove wcs entries in hdr_base
            hdr_b = hdr_base.copy()
            delkeys = list(read_wcs(hdr_b).to_header(relax=True).keys())
            if 'CD1_1' in hdr_base: 
                delkeys += ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
            if 'MJD-OBS' in delkeys:
                delkeys.remove('MJD-OBS')
            if 'DATE-OBS' in delkeys:
                delkeys.remove('DATE-OBS')
            for key in delkeys:
                if key in hdr_b:
                    del hdr_b[key]
            
            # ** Extract wcs entries in hdr_wcs
            hdr_w = hdr_wcs.copy()
            wcshdr = read_wcs(hdr_w).to_header(relax=True)
            if 'PC1_2' not in wcshdr: wcshdr['PC1_2'] = 0.0
            if 'PC2_1' not in wcshdr: wcshdr['PC2_1'] = 0.0
            if 'MJD-OBS' in wcshdr: del wcshdr['MJD-OBS']
            if 'DATE-OBS' in wcshdr: del wcshdr['DATE-OBS']
            
            # ** Combine 
            hdr_op = hdr_b.copy()
            hdr_op.update(wcshdr)

            # ** Post corrections
            if hdr_wcs['CTYPE1'] == 'RA---TAN' and 'PV1_0' in hdr_wcs:
                hdr_op['CTYPE1'] = 'RA---TAN'
                hdr_op['CTYPE2'] = 'DEC--TAN'

            if 'CD1_1' in hdr_wcs and 'PC1_1' in hdr_op:
                hdr_op.rename_keyword('PC1_1', 'CD1_1')
                hdr_op.rename_keyword('PC1_2', 'CD1_2')
                hdr_op.rename_keyword('PC2_1', 'CD2_1')
                hdr_op.rename_keyword('PC2_2', 'CD2_2')
            
            return hdr_op

        hdr_base = fits.getheader(FITS_obj, ext=0)
        hdr_wcs = fits.getheader(FITS_ref, ext=0)
        hdr_op = combine_header(hdr_base, hdr_wcs)

        hdr_op['SWARP_O'] = (pa.basename(FITS_obj), 'MeLOn: PYSWarp')
        hdr_op['SWARP_R'] = (pa.basename(FITS_ref), 'MeLOn: PYSWarp')

        for key in ConfigDict:
            value = ConfigDict[key]
            pack = ' : '.join(['SWarp Parameters', key, value])
            hdr_op.add_history(pack)

        # * Fill the missing data in resampled image [SWarp default 0]
        PixA_Re, PixA_WRe = None, None
        MissingMask = None
        try: 
            PixA_Re = fits.getdata(FITS_tRe, ext=0).T
            PixA_WRe = fits.getdata(FITS_tWRe, ext=0).T
            MissingMask = PixA_WRe == 0
            if Fillvalue is not None:
                PixA_Re[MissingMask] = Fillvalue

            # * Save Products
            if FITS_Re is not None:
                fits.HDUList(fits.PrimaryHDU(PixA_Re.T, header=hdr_op)).writeto(FITS_Re, overwrite=True)
            if FITS_WRe is not None:
                fits.HDUList(fits.PrimaryHDU(PixA_WRe.T, header=hdr_op)).writeto(FITS_WRe, overwrite=True)
        except: 
            warnings.warn('MeLOn WARNING: SWarp [FAIL] at ', pa.basename(FITS_obj))
        os.system('rm -rf %s'%TDIR)

        return PixA_Re, PixA_WRe, MissingMask
