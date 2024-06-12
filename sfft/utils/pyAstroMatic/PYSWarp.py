import os
import warnings
import os.path as pa
from astropy.io import fits
from tempfile import mkdtemp
from sfft.utils.ReadWCS import Read_WCS
from sfft.utils.CombineHeader import Combine_Header
from sfft.utils.pyAstroMatic.AMConfigMaker import AMConfig_Maker
# version: Apr 22, 2024

# improved by Lauren Aldoroty (Duke Univ.)
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class PY_SWarp:
    @staticmethod
    def Mk_ConfigDict(GAIN_KEY='GAIN', SATUR_KEY='SATURATE', GAIN_DEFAULT=0.0, SATLEV_DEFAULT=50000.0, OVERSAMPLING=1, \
                      RESAMPLING_TYPE='LANCZOS3', SUBTRACT_BACK='N', PROJECTION_TYPE='TAN', COMBINE='Y', \
                      COMBINE_TYPE='MEDIAN', WEIGHT_SUFFIX='.weight.fits', WRITE_XML='N', VERBOSE_TYPE='NORMAL'):

        """
        # SWarp parameters:

        -GAIN_KEY ['GAIN']                  # SWarp Parameter GAIN_KEYWORD
                                            # i.e., keyword of GAIN in FITS image header

        -SATUR_KEY ['SATURATE']             # SWarp Parameter SATLEV_KEYWORD
                                            # i.e., keyword of the saturation level in the FITS image header

        -GAIN_DEFAULT [0.0]                 # Gain value if no header keyword available

        -SATLEV_DEFAULT [50000.0]           # Saturation value if no header keyword available

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

        -PROJECTION_TYPE ['TAN']            # SWarp Parameter PROJECTION_TYPE
                                       
        -VERBOSE_TYPE ['NORMAL']            # SWarp Parameter VERBOSE_TYPE
                                            # QUIET, LOG, NORMAL, or FULL

        """

        ConfigDict = {}
        ConfigDict['GAIN_KEYWORD'] = '%s' %GAIN_KEY
        ConfigDict['SATLEV_KEYWORD'] = '%s' %SATUR_KEY

        ConfigDict['GAIN_DEFAULT'] = '%s' %GAIN_DEFAULT
        ConfigDict['SATLEV_DEFAULT'] = '%s' %SATLEV_DEFAULT

        ConfigDict['OVERSAMPLING'] = '%d' %OVERSAMPLING
        ConfigDict['RESAMPLING_TYPE'] = '%s' %RESAMPLING_TYPE
        ConfigDict['SUBTRACT_BACK'] = '%s' %SUBTRACT_BACK

        ConfigDict['PROJECTION_TYPE'] = '%s' %PROJECTION_TYPE

        ConfigDict['COMBINE'] = '%s' %COMBINE  # trivial here
        ConfigDict['COMBINE_TYPE'] = '%s' %COMBINE_TYPE  # trivial here
        
        ConfigDict['WEIGHT_SUFFIX'] = '%s' %WEIGHT_SUFFIX  # trivial here
        ConfigDict['WRITE_XML'] = 'N'
        ConfigDict['VERBOSE_TYPE'] = '%s' %VERBOSE_TYPE
        
        return ConfigDict

    @staticmethod
    def PS(FITS_obj, FITS_ref, ConfigDict, FITS_resamp=None, \
        FILL_VALUE=None, TMPDIR_ROOT=None, VERBOSE_LEVEL=2):
        
        """
        # Inputs & Outputs:

        -FITS_obj []                        # FITS file path of the input image to be resampled

        -FITS_ref []                        # FITS file path of the input image as resampling reference

        -ConfigDict []                      # Dictionary for config file, input from PY_SWarp.Mk_ConfigDict(). 

        -FITS_resamp [None]                 # FITS file path of the output image of resampled -FITS_obj

        # Other parameters:

        -FILL_VALUE [None]                  # How to fill the invalid (boundary) pixels in -FITS_resamp and -FITS_resamp_weight
                                            # e.g., -FILL_VALUE = 0.0, -FILL_VALUE = np.nan

        -VERBOSE_LEVEL [2]                  # The level of verbosity, can be [0, 1, 2]
                                            # 0/1/2: QUIET/NORMAL/FULL mode
                                            # NOTE: it only controls the verbosity out of SWarp.

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

        # * create SWarp configuration file in TDIR        
        swarp_config_path = AMConfig_Maker.AMCM(MDIR=TDIR, \
            AstroMatic_KEY='swarp', ConfigDict=ConfigDict, tag='PYSWarp')

        # * make temporary FITS_resamp & FIT_resamp_weight in TDIR
        fname_obj = pa.basename(FITS_obj)[:-5]
        tFITS_resamp = os.path.join(TDIR, '%s.tmp_resamp.fits' %fname_obj)
        tFITS_resamp_weight = os.path.join(TDIR, '%s.tmp_resamp_weight.fits' %fname_obj)

        # * build .head file from FITS_ref
        phr_ref = fits.getheader(FITS_ref, ext=0)
        w_ref = Read_WCS.RW(phr_ref, VERBOSE_LEVEL=VERBOSE_LEVEL)

        _headfile = tFITS_resamp[:-5] + '.head'
        wcshdr_ref = w_ref.to_header(relax=True)   # SIP distorsion requires relax=True

        wcshdr_ref['BITPIX'] = phr_ref['BITPIX']
        wcshdr_ref['NAXIS'] = phr_ref['NAXIS']

        wcshdr_ref['NAXIS1'] = phr_ref['NAXIS1']
        wcshdr_ref['NAXIS2'] = phr_ref['NAXIS2']

        wcshdr_ref.tofile(_headfile)

        # * trigger SWarp 
        command = "cd %s && swarp %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -c %s"  \
            %(TDIR, FITS_obj, tFITS_resamp, tFITS_resamp_weight, swarp_config_path)
        print('MeLOn CheckPoint: Run SWarp Command ... \n %s' %command)
        os.system(command)

        # * make a combined header for output FITS
        hdr_op = Combine_Header.CH(hdr_base=phr_obj, hdr_wcs=phr_ref, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        # * update header by the SWarp generated saturation level
        NEW_SATUR = fits.getheader(tFITS_resamp, ext=0)['SATURATE']
        if ConfigDict['SATLEV_KEYWORD'] in hdr_op:
            hdr_op[ConfigDict['SATLEV_KEYWORD']] = (NEW_SATUR, 'MeLOn: PYSWarp')

        # * add history
        hdr_op['SWARP_O'] = (pa.basename(FITS_obj), 'MeLOn: PYSWarp')
        hdr_op['SWARP_R'] = (pa.basename(FITS_ref), 'MeLOn: PYSWarp')
        for key in ConfigDict:
            value = ConfigDict[key]
            pack = ' : '.join(['SWarp Parameter', key, value])
            hdr_op.add_history(pack)

        # * fill the missing data in resampled image [SWarp default 0]
        PixA_resamp, MissingMask = None, None
        try: 
            PixA_resamp = fits.getdata(tFITS_resamp, ext=0).T
            PixA_resamp_weight = fits.getdata(tFITS_resamp_weight, ext=0).T
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

    @staticmethod
    def Coadd(FITS_obj, FITS_ref, ConfigDict, OUT_path, \
        FILL_VALUE=None, TMPDIR_ROOT=None, VERBOSE_LEVEL=2):

        """
        # Inputs & Outputs:

        -FITS_obj []                        # List of FITS file paths of the input images to be resampled

        -FITS_ref []                        # FITS file path of the input image as resampling reference

        -ConfigDict []                      # Dictionary for config file, input from PY_SWarp.Mk_ConfigDict(). 

        -FITS_resamp [None]                 # FITS file path of the output image of resampled, coadded -FITS_obj

        # Other parameters:

        -FILL_VALUE [None]                  # How to fill the invalid (boundary) pixels in -FITS_resamp and -FITS_resamp_weight
                                            # e.g., -FILL_VALUE = 0.0, -FILL_VALUE = np.nan

        -VERBOSE_LEVEL [2]                  # The level of verbosity, can be [0, 1, 2]
                                            # 0/1/2: QUIET/NORMAL/FULL mode
                                            # NOTE: it only controls the verbosity out of SWarp.

        -TMPDIR_ROOT [None]                 # Specify root directory of temporary working directory. 

        # Returns:

            PixA_resamp                     # Pixel array of the resampled, coadded image
                                            # P.S. PixA_resamp = fits.getdata(OUT_path, ext=0).T
            
            PixA_resamp_weight              # Pixel array of the weight map
        
        # * Remarks on PYSWarp
        #       
        #   [1] In SWarp, one need provide a target WCS-frame stored in fname.head, with name coincident with FITS_resamp.
        #       If no target WCS-frame is available, the output image will have a canonical N-E orientation.
        #       However, the current python wrapper PYSWarp must read a target WCS-frame from a given FITS_ref.
        #
        #   [2] SWarp allows users to feed a corresponding weight map as input.
        #       However, the current python wrapper PYSWarp does not support this feature.
        #
        #   [3] SWarp will automatically calculate the following keywords and set them into the output image.
        #       > EXPTIME: sum of exposure times in the part of coadd with the most overlaps.
        #       > GAIN: effective gain with flux and weighting applied.
        #         Note that any change in pixel scale only redistribute the flux with total flux conservative,
        #         therefore, the image resampling does not alter the GAIN value.
        #       > SATURATE: Minimum of all input saturation levels with flux scaling applied.
        #         Note that a change in pixel scale can alter the saturation level.
        #       > MJD-OBS: MJD of earliest start of exposures 
        #
        """

        # * check FITS_obj headers
        for obj in FITS_obj:
            phr_obj = fits.getheader(obj, ext=0)

            # a mild warning
            if VERBOSE_LEVEL in [1, 2]:
                if 'EXPTIME' not in phr_obj:
                    _warn_message = 'SWarp cannot find keyword EXPTIME in FITS header of [%s]!' %obj
                    warnings.warn('MeLOn WARNING: %s' %_warn_message)
            
            # a mild warning
            if VERBOSE_LEVEL in [1, 2]:
                if 'MJD-OBS' not in phr_obj:
                    _warn_message = 'SWarp cannot find keyword MJD-OBS in FITS header of [%s]!' %obj
                    warnings.warn('MeLOn WARNING: %s' %_warn_message)

        # * make directory as a workplace
        TDIR = mkdtemp(suffix=None, prefix='PYSWarp_', dir=TMPDIR_ROOT)

        # * create SWarp configuration file in TDIR        
        swarp_config_path = AMConfig_Maker.AMCM(MDIR=TDIR, \
            AstroMatic_KEY='swarp', ConfigDict=ConfigDict, tag='PYSWarp')

        FITS_obj_str = ' '.join(FITS_obj)
        weightOUT_path = OUT_path[:-5] + '.weight.fits'

        # * trigger SWarp 
        command = "cd %s && swarp %s -IMAGEOUT_NAME %s -WEIGHTOUT_NAME %s -c %s"  \
            %(TDIR, FITS_obj_str, OUT_path, weightOUT_path, swarp_config_path)
        print('MeLOn CheckPoint: Run SWarp Command ... \n %s' %command)
        os.system(command)
        os.system('rm -rf %s'%TDIR)

        PixA_resamp = fits.getdata(OUT_path, ext=0).T
        PixA_resamp_weight = fits.getdata(weightOUT_path, ext=0).T

        return PixA_resamp, PixA_resamp_weight