import os
import warnings
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
from sfft.utils.ReadWCS import Read_WCS
from sfft.utils.CombineHeader import Combine_Header
from sfft.utils.pyAstroMatic.AMConfigMaker import AMConfig_Maker
# version: Apr 7, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class PY_SWarp:
    @staticmethod
    def PS(FITS_obj, FITS_ref, FITS_resamp=None,  FITS_DATA_EXT=0, \
        GAIN_KEY='GAIN', SATUR_KEY='SATURATE', GAIN_DEFAULT=1.0, SATLEV_DEFAULT=100000., \
        NAXIS1_VAL=None, NAXIS2_VAL=None, RESAMPLE='Y', IMAGE_SIZE=0, OVERSAMPLING=1, RESAMPLING_TYPE='LANCZOS3', \
        SUBTRACT_BACK='N', FILL_VALUE=None, VERBOSE_TYPE='NORMAL', VERBOSE_LEVEL=2,TMPDIR_ROOT=None):
        
        """
        # Inputs & Outputs:

        -FITS_obj []                        # FITS file path of the input image to be resampled

        -FITS_ref []                        # FITS file path of the input image as resampling reference

        -FITS_resamp [None]                 # FITS file path of the output image of resampled -FITS_obj

        -FITS_DATA_EXT [0]                  # The extension where the image is located in the FITS file, e.g., the 0 in hdu[0].data. 

        # SWarp parameters:

        -GAIN_KEY ['GAIN']                  # SWarp Parameter GAIN_KEYWORD
                                            # i.e., keyword of GAIN in FITS image header

        -SATUR_KEY ['SATURATE']             # SWarp Parameter SATLEV_KEYWORD
                                            # i.e., keyword of the saturation level in the FITS image header

        -GAIN_DEFAULT                       # Gain value if no header keyword available

        -SATLEV_DEFAULT                     # Saturation value if no header keyword available

        -NAXIS1_VAL                         # NAXIS1 value if no header keyword available
        
        -NAXIS2_VAL                         # NAXIS2 value if no header keyword available
        
        -RESAMPLE                           # Toggle resampling. Default 'Y'. Other option 'N'.

        -IMAGE_SIZE                         # Size of output image. Default '0', meaning automatic.



        -OVERSAMPLING [1]                   # SWarp Parameter OVERSAMPLING
                                            # Oversampling in each dimension
                                            # (0 = automatic)
                                            # P.S. Here I changed the default value from 0 to 1
                                            # NOTE: large OVERSAMPLING may cause higher pixel correlation

        -RESAMPLING_TYPE ['LANCZOS3']       # SWarp Parameter RESAMPLING_TYPE
                                            # NEAREST,BILINEAR,LANCZOS2,LANCZOS3
                                            # LANCZOS4 (1 per axis) or FLAGS
                                            # NOTE: LANCZOS4 is relatively time-consuming

        -SUBTRACT_BACK ['N']                # SWarp Parameter SUBTRACT_BACK 
                                            # Subtraction sky background (Y/N)? (all or for each image)
                                            # P.S. Here I changed the default value from 'Y' to 'N'
                                       
        -VERBOSE_TYPE ['NORMAL']            # SWarp Parameter VERBOSE_TYPE
                                            # QUIET,LOG,NORMAL, or FULL

        -VERBOSE_LEVEL [2]                  # The level of verbosity, can be [0, 1, 2]
                                            # 0/1/2: QUIET/NORMAL/FULL mode
                                            # NOTE: it only controls the verbosity out of SWarp.

        # Other parameters:

        -FILL_VALUE [None]                  # How to fill the invalid (boundary) pixels in -FITS_resamp and -FITS_resamp_weight
                                            # e.g., -FILL_VALUE = 0.0, -FILL_VALUE = np.nan

        -TMPDIR_ROOT [None]                 # Specify root directory of temporary working directory. 

        # Returns:

            PixA_resamp                     # Pixel array of the resampled image
                                            # P.S. PixA_resamp = fits.getdata(FITS_resamp, ext=0).T
            
            MissingMask                    # Pixel array of the invalid pixels
                                            # P.S. -FILL_VALUE will fill the pixels where MissingMask is True
        
        # * Remarks on PYSWarp
        #   [1] SWarp itself can perform resampling and subsequent image combination.
        #       However, the current python wrapper PYSWarp only performs image resampling.
        #     
        #   [2] In SWarp, one need provide a target WCS-frame stored in fname.head, with name coincident with FITS_resamp.
        #       If no target WCS-frame is available, the output image will have a canonical N-E orientation.
        #       However, the current python wrapper PYSWarp must read a target WCS-frame from a given FITS_ref.
        #
        #   [3] SWarp allows users to feed a corresponding weight map as input.
        #       However, the current python wrapper PYSWarp does not support this feature.
        #
        #   [4] SWarp will automatically calculate the following keywords and set them into the output image.
        #       > EXPTIME: sum of exposure times in the part of coadd with the most overlaps.
        #       > GAIN: effective gain with flux and weighting applied.
        #         Note that any change in pixel scale only redistribute the flux with total flux conservative,
        #         therefore, the image resampling does not alter the GAIN value.
        #       > SATURATE: Minimum of all input saturation levels with flux scaling applied.
        #         Note that a change in pixel scale can alter the saturation level.
        #       > MJD-OBS: MJD of earliest start of exposures 
        #       
        #       For PYSWarp which only performs image resampling:
        #       > We can safely preserve EXPTIME, GAIN, MJD-OBS, and use SWarp output SATURATE value.
        #       > The resulting header of the PYSWarp output is constructed by combining
        #         FITS_obj header with WCS keywords removed & FITS_ref header only with WCS keywords
        #         In additional, we will update SATURATE value as described above, and 
        #         record the SWarp parameters used in the function.
        #
        """

        # * check FITS_obj header
        objname = pa.basename(FITS_obj)
        phr_obj = fits.getheader(FITS_obj, ext=0)

        # a mild warning
        if VERBOSE_LEVEL in [1, 2]:
            if 'EXPTIME' not in phr_obj:
                _warn_message = 'SWarp cannot find keyword EXPTIME in FITS header of [%s]!' %objname
                warnings.warn('MeLOn WARNING: %s' %_warn_message)
        
        # a mild warning
        if VERBOSE_LEVEL in [1, 2]:
            if 'MJD-OBS' not in phr_obj:
                _warn_message = 'SWarp cannot find keyword MJD-OBS in FITS header of [%s]!' %objname
                warnings.warn('MeLOn WARNING: %s' %_warn_message)

        # * make directory as a workplace
        TDIR = mkdtemp(suffix=None, prefix='PYSWarp_', dir=TMPDIR_ROOT)
        print('TDIR', TDIR)

        # * create SWarp configuration file in TDIR
        ConfigDict = {}
        ConfigDict['GAIN_KEYWORD'] = '%s' %GAIN_KEY
        ConfigDict['SATLEV_KEYWORD'] = '%s' %SATUR_KEY

        ConfigDict['GAIN_DEFAULT'] = '%s' %GAIN_DEFAULT
        ConfigDict['SATLEV_DEFAULT'] = '%s' %SATLEV_DEFAULT

        ConfigDict['RESAMPLE'] = '%s' %RESAMPLE
        ConfigDict['IMAGE_SIZE'] = '%d' %IMAGE_SIZE

        ConfigDict['OVERSAMPLING'] = '%d' %OVERSAMPLING
        ConfigDict['RESAMPLING_TYPE'] = '%s' %RESAMPLING_TYPE
        ConfigDict['SUBTRACT_BACK'] = '%s' %SUBTRACT_BACK

        ConfigDict['COMBINE'] = 'Y'  # trivial here
        ConfigDict['COMBINE_TYPE'] = 'MEDIAN'  # trivial here
        
        ConfigDict['WEIGHT_SUFFIX'] = '.weight.fits'  # trivial here
        ConfigDict['WRITE_XML'] = 'N'
        ConfigDict['VERBOSE_TYPE'] = '%s' %VERBOSE_TYPE
        
        swarp_config_path = AMConfig_Maker.AMCM(MDIR=TDIR, \
            AstroMatic_KEY='swarp', ConfigDict=ConfigDict, tag='PYSWarp')

        # * make temporary FITS_resamp & FIT_resamp_weight in TDIR
        tFITS_resamp = os.path.join(TDIR, '%s.tmp_resamp.fits' %pa.basename(FITS_obj)[:-5])
        tFITS_resamp_weight = os.path.join(TDIR, '%s.tmp_resamp_weight.fits' %pa.basename(FITS_obj)[:-5])

        # * build .head file from FITS_ref
        phr_ref = fits.getheader(FITS_ref, ext=0)
        w_ref = Read_WCS.RW(phr_ref, VERBOSE_LEVEL=VERBOSE_LEVEL)

        _headfile = tFITS_resamp[:-5] + '.head'
        wcshdr_ref = w_ref.to_header(relax=True)   # SIP distorsion requires relax=True

        wcshdr_ref['BITPIX'] = phr_ref['BITPIX']
        wcshdr_ref['NAXIS'] = phr_ref['NAXIS']

        try:
            wcshdr_ref['NAXIS1'] = phr_ref['NAXIS1']
            wcshdr_ref['NAXIS2'] = phr_ref['NAXIS2']
        except KeyError:
            wcshdr_ref['NAXIS1'] = NAXIS1_VAL
            wcshdr_ref['NAXIS2'] = NAXIS2_VAL

        wcshdr_ref.tofile(_headfile)

        # * trigger SWarp 
        os.system('cd %s && swarp %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -c %s' \
            %(TDIR, FITS_obj, tFITS_resamp, tFITS_resamp_weight, swarp_config_path))

        print('cd %s && swarp %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -c %s' \
            %(TDIR, FITS_obj, tFITS_resamp, tFITS_resamp_weight, swarp_config_path))
        
        # * make a combined header for output FITS
        hdr_op = Combine_Header.CH(hdr_base=phr_obj, hdr_wcs=phr_ref, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        # * update header by the SWarp generated saturation level

        # NEW_SATUR = fits.getheader(tFITS_resamp, ext=0)['SATURATE']
        NEW_SATUR = fits.open(tFITS_resamp)
        hdr_op[SATUR_KEY] = (NEW_SATUR[0].header['SATURATE'], 'MeLOn: PYSWarp')

        # * add history
        hdr_op['SWARP_O'] = (pa.basename(FITS_obj), 'MeLOn: PYSWarp')
        hdr_op['SWARP_R'] = (pa.basename(FITS_ref), 'MeLOn: PYSWarp')
        for key in ConfigDict:
            value = ConfigDict[key]
            print('key: ', key)
            print('value:', value)
            pack = ' : '.join(['SWarp Parameter', key, value])
            hdr_op.add_history(pack)

        # * fill the missing data in resampled image [SWarp default 0]
        PixA_resamp, MissingMask = None, None
        try: 
            PixA_resamp = fits.getdata(tFITS_resamp, ext=FITS_DATA_EXT).T
            PixA_resamp_weight = fits.getdata(tFITS_resamp_weight, ext=FITS_DATA_EXT).T
            MissingMask = PixA_resamp_weight == 0
            if FILL_VALUE is not None:
                PixA_resamp[MissingMask] = FILL_VALUE
            if FITS_resamp is not None:
                hdl_op = fits.HDUList(fits.PrimaryHDU(PixA_resamp.T, header=hdr_op))
                hdl_op.writeto(FITS_resamp, overwrite=True)
        except:
            if VERBOSE_LEVEL in [0, 1, 2]:
                _warn_message = 'SWarp FAILED on [%s]!' %objname
                warnings.warn('MeLOn WARNING: %s' %_warn_message)
        
        os.system('rm -rf %s'%TDIR)

        return PixA_resamp, MissingMask
