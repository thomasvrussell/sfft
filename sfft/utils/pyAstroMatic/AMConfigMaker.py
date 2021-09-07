import re
import os
from tempfile import mkdtemp

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class AMConfig_Maker:
    @staticmethod
    def AMCM(MDIR, AstroMatic_KEY, ConfigDict=None, tag='pyastromatic'):
                
        if AstroMatic_KEY in ['sex', 'sextractor']:
            print('MeLOn : SExtactor employed : %s -dd' %AstroMatic_KEY)
        if AstroMatic_KEY == 'psfex':
            print('MeLOn : psfex employed : psfex -dd')
        if AstroMatic_KEY == 'swarp':
            print('MeLOn : SWarp employed : swarp -dd')
        if AstroMatic_KEY == 'scamp':
            print('MeLOn : Scamp employed : scamp -dd')
        if AstroMatic_KEY == 'sky':
            print('MeLOn : Skymaker employed : sky -dd')

        prefix = '%s_' %tag
        suffix = '%s_config' %AstroMatic_KEY
        TDIR = mkdtemp(suffix=suffix, prefix=prefix, dir=MDIR)
        config_path = ''.join([TDIR, "/%s.%s" %(tag, AstroMatic_KEY)])

        # * Creat default configuration file
        # * Extract the keywords in the configuration        
        # P.S. > Use missfits to split Multi-Extension-FITS can be simply done in terminal
        #        missfits -OUTFILE_TYPE SPLIT -WRITE_XML N FITS_MEF
        #      > Use missfits remove keyword in header
        #        missfits X.fits -REMOVE_KEYWORD SKY_BACK -SAVE_TYPE REPLACE        
        #        missfits X.fits -REMOVE_KEYWORD SKY_BACK -SAVE_TYPE NEW -NEW_SUFFIX .miss    # generate new *.miss.fits
                
        os.system("%s -dd > %s" %(AstroMatic_KEY, config_path))      # WARNING: It can be invalid for other os-platform
        config_lst = open(config_path).read().split('\n')
        Keys = [re.split("\s+", line)[0] for line in config_lst]     # WARNING: Invalid lines will get '' or '#'
        
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
