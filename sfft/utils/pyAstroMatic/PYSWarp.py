import os
import sys
import warnings
import numpy as np
import os.path as pa
from astropy.io import fits
from astropy.wcs import WCS
from tempfile import mkdtemp
from sfft.utils.CombineHeader import Combine_Header
from sfft.utils.pyAstroMatic.AMConfigMaker import AMConfig_Maker
# version: Sep 22, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.3"

class PY_SWarp:
    @staticmethod
    def PS(FITS_obj, FITS_ref, FITS_resamp=None, FITS_resamp_weight=None, \
        GAIN_KEY='GAIN', SATUR_KEY='SATURATE', EXPTIME_KEY='EXPTIME', \
        OVERSAMPLING=1, RESAMPLING_TYPE='LANCZOS3', SUBTRACT_BACK='N', FILL_VALUE=None, \
        WEIGHT_SUFFIX='.weight.fits', WRITE_XML='N', VERBOSE_TYPE='NORMAL'):

        """
        # @ Python Version of SWarp
        #  a) PYSWarp Purpose
        #     1. The current python version only implement Resampling of single SEF-FITS image.
        #     2. Note SWarp can also do the simple combination procedure after Resampling.
        #
        #  b) SWarp destination-frame
        #     @ Default-NE Mode
        #       The output WCS-frame simply use canonical N-E orientation, i.e., NE-WCS-Frame. 
        #       NOTE this mode is not supported in our function.
        #     @ Given-REF Mode
        #       The output WCS-frame (including image size) is determined by the given FITS_ref, 
        #       we ganna to prepare a .head file, coincident name with FITS_resamp, to store REF-WCS-Frame.
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
        #  d) Tips on resampling method
        #     NOTE: LANCZOS4 is relatively time-consuming
        #     NOTE: large OVERSAMPLING may cause higher pixel correlation
        #
        """

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

        ConfigDict['OVERSAMPLING'] = '%d' %OVERSAMPLING
        ConfigDict['RESAMPLING_TYPE'] = '%s' %RESAMPLING_TYPE
        ConfigDict['SUBTRACT_BACK'] = '%s' %SUBTRACT_BACK
        ConfigDict['COMBINE_TYPE'] = 'MEDIAN'  # trivial here

        ConfigDict['WEIGHT_SUFFIX'] = '%s' %WEIGHT_SUFFIX
        ConfigDict['WRITE_XML'] = '%s' %WRITE_XML
        ConfigDict['VERBOSE_TYPE'] = '%s' %VERBOSE_TYPE
        
        swarpconfig_path = AMConfig_Maker.AMCM(MDIR=TDIR, \
            AstroMatic_KEY='swarp', ConfigDict=ConfigDict, tag='PYSWarp')

        # * Make temporary FITS_resamp & FIT_resamp_weight in TDIR
        tFITS_resamp = TDIR + '/%s.tmp_resamp.fits' %pa.basename(FITS_obj)[:-5]
        tFITS_resamp_weight = TDIR + '/%s.tmp_resamp_weight.fits' %pa.basename(FITS_obj)[:-5]

        # * Build .head file from FITS_ref
        phr_ref = fits.getheader(FITS_ref, ext=0)
        w_ref = WCS(phr_ref)            
        _headfile = tFITS_resamp[:-5] + '.head'
        wcshdr_ref = w_ref.to_header(relax=True)   # SIP distorsion requires relax=True

        wcshdr_ref['BITPIX'] = phr_ref['BITPIX']
        wcshdr_ref['NAXIS'] = phr_ref['NAXIS']            
        wcshdr_ref['NAXIS1'] = phr_ref['NAXIS1']
        wcshdr_ref['NAXIS2'] = phr_ref['NAXIS2']
        wcshdr_ref.tofile(_headfile)

        # * Trigger SWarp 
        os.system('cd %s && swarp %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -c %s' \
            %(TDIR, FITS_obj, tFITS_resamp, tFITS_resamp_weight, swarpconfig_path))

        # * Make a combined header for resulting products
        hdr_base = fits.getheader(FITS_obj, ext=0)
        hdr_wcs = fits.getheader(FITS_ref, ext=0)
        hdr_op = Combine_Header.CH(hdr_base=hdr_base, hdr_wcs=hdr_wcs)

        hdr_op['SWARP_O'] = (pa.basename(FITS_obj), 'MeLOn: PYSWarp')
        hdr_op['SWARP_R'] = (pa.basename(FITS_ref), 'MeLOn: PYSWarp')

        for key in ConfigDict:
            value = ConfigDict[key]
            pack = ' : '.join(['SWarp Parameters', key, value])
            hdr_op.add_history(pack)

        # * Fill the missing data in resampled image [SWarp default 0]
        PixA_resamp, PixA_resamp_weight = None, None
        MISSING_MASK = None
        try: 
            PixA_resamp = fits.getdata(tFITS_resamp, ext=0).T
            PixA_resamp_weight = fits.getdata(tFITS_resamp_weight, ext=0).T
            MISSING_MASK = PixA_resamp_weight == 0

            if FILL_VALUE is not None:
                PixA_resamp[MISSING_MASK] = FILL_VALUE
            if FITS_resamp is not None:
                hdl_op = fits.HDUList(fits.PrimaryHDU(PixA_resamp.T, header=hdr_op))
                hdl_op.writeto(FITS_resamp, overwrite=True)
            if FITS_resamp_weight is not None:
                hdl_op = fits.HDUList(fits.PrimaryHDU(PixA_resamp_weight.T, header=hdr_op))
                hdl_op.writeto(FITS_resamp_weight, overwrite=True)
        
        except: 
            warnings.warn('MeLOn WARNING: SWarp FAILS at [%s]' %(pa.basename(FITS_obj)))
        os.system('rm -rf %s'%TDIR)

        return PixA_resamp, PixA_resamp_weight, MISSING_MASK
