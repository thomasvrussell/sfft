import re
import os
from tempfile import mkdtemp
# version: Jan 21, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class AMConfig_Maker:
    @staticmethod
    def AMCM(MDIR, AstroMatic_KEY, ConfigDict=None, tag='pyastromatic'):

        assert AstroMatic_KEY in ['sex', 'sextractor', 'psfex', 'swarp', 'scamp', 'sky']
        prefix, suffix = '%s_' %tag, '%s_config' %AstroMatic_KEY
        TDIR = mkdtemp(suffix=suffix, prefix=prefix, dir=MDIR)
        config_path = ''.join([TDIR, "/%s.%s" %(tag, AstroMatic_KEY)])

        """
        # Remarks on missfits (not included here)
        # [1] Use missfits to split Multi-Extension-FITS can be simply done in terminal
        #     e.g., missfits -OUTFILE_TYPE SPLIT -WRITE_XML N FITS_MEF
        #
        # [2] Use missfits remove keyword in header
        #     e.g., missfits X.fits -REMOVE_KEYWORD SKY_BACK -SAVE_TYPE REPLACE        
        #     e.g., missfits X.fits -REMOVE_KEYWORD SKY_BACK -SAVE_TYPE NEW -NEW_SUFFIX .miss
        #
        """

        # * Creat default configuration file and extract keywords

        os.system("%s -dd > %s" %(AstroMatic_KEY, config_path))    # WARNING: may be invalid for other os-platform
        config_lst = open(config_path).read().split('\n')
        Keys = [re.split("\s+", line)[0] for line in config_lst]   # WARNING: invalid lines will get '' or '#'

        # * Modify the configuration 
        for key in ConfigDict:
            value = ConfigDict[key]
            if isinstance(value, tuple):
                value = ','.join([str(v) for v in value])
            else: value = str(value)
            
            index = Keys.index(key)
            line = config_lst[index]
            
            comment = ''
            cindx = re.search('#', line)
            if cindx is not None:
                comment = line[cindx.start():]
            config_lst[index] = '    '.join([key, value, comment])

        string = "\n".join(config_lst)

        sfile = open(config_path, 'w')
        sfile.write(string)
        sfile.close()

        return config_path
