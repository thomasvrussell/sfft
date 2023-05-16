from sfft.utils.ReadWCS import Read_WCS
# version: Feb 5, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class Combine_Header:
    @staticmethod
    def CH(hdr_base, hdr_wcs, VERBOSE_LEVEL=2):

        # ** remove wcs entries in hdr_base
        hdr_b = hdr_base.copy()
        w_base = Read_WCS.RW(hdr_b, VERBOSE_LEVEL=VERBOSE_LEVEL)
        delkeys = list(w_base.to_header(relax=True).keys())
        
        if 'CD1_1' in hdr_base: 
            delkeys += ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
        if 'MJD-OBS' in delkeys:
            delkeys.remove('MJD-OBS')
        if 'DATE-OBS' in delkeys:
            delkeys.remove('DATE-OBS')
        
        for key in delkeys:
            if key in hdr_b:
                del hdr_b[key]
        
        # ** extract wcs entries in hdr_wcs
        hdr_w = hdr_wcs.copy()
        w_wcs = Read_WCS.RW(hdr_w, VERBOSE_LEVEL=VERBOSE_LEVEL)
        wcshdr = w_wcs.to_header(relax=True)
        
        if 'PC1_2' not in wcshdr: wcshdr['PC1_2'] = 0.0
        if 'PC2_1' not in wcshdr: wcshdr['PC2_1'] = 0.0
        if 'MJD-OBS' in wcshdr: del wcshdr['MJD-OBS']
        if 'DATE-OBS' in wcshdr: del wcshdr['DATE-OBS']

        # ** combine headers
        hdr_op = hdr_b.copy()
        hdr_op.update(wcshdr)

        # ** post corrections
        if hdr_wcs['CTYPE1'] == 'RA---TAN' and 'PV1_0' in hdr_wcs:
            hdr_op['CTYPE1'] = 'RA---TAN'
            hdr_op['CTYPE2'] = 'DEC--TAN'

        if 'CD1_1' in hdr_wcs and 'PC1_1' in hdr_op:
            hdr_op.rename_keyword('PC1_1', 'CD1_1')
            hdr_op.rename_keyword('PC1_2', 'CD1_2')
            hdr_op.rename_keyword('PC2_1', 'CD2_1')
            hdr_op.rename_keyword('PC2_2', 'CD2_2')
        
        return hdr_op
