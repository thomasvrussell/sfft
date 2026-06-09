import re
from sfft.utils.ReadWCS import Read_WCS
# version: Jun 9, 2026

__author__ = "Lei Hu <leihu@sas.upenn.edu>"
__version__ = "v1.7"

class Combine_Header:
    @staticmethod
    def CH(hdr_base, hdr_wcs, extra_delkeys_base=None, VERBOSE_LEVEL=2):

        # ** remove WCS entries in hdr_base
        w_base = Read_WCS.RW(hdr_base, VERBOSE_LEVEL=VERBOSE_LEVEL)
        
        # WCS.to_header() is used as the seed deletion list.
        # NOTE: it may omit some cards such as zero-valued terms.
        delkeys = list(w_base.to_header(relax=True).keys())

        # exclude exposure-time metadata from the deletion list.
        for tkey in ['MJD-OBS', 'DATE-OBS']:
            if tkey in delkeys:
                delkeys.remove(tkey)

        # add WCS reference time to the deletion list (redundant).
        if 'MJDREF' in hdr_base:
            if 'MJDREF' not in delkeys:
                delkeys.append('MJDREF')
        
        # add WCS-related keys missed by WCS.to_header() to the deletion list.
        for key in hdr_base:
            if key not in delkeys:
                KEY = key.upper()
                if (
                    # Linear/TPV/SIP keywords
                    re.match(r"^CD\d+_\d+$", KEY) is not None or
                    re.match(r"^PC\d+_\d+$", KEY) is not None or
                    re.match(r"^PV\d+_\d+$", KEY) is not None or
                    re.match(r"^(A|B|AP|BP)_ORDER$", KEY) is not None or
                    re.match(r"^(A|B|AP|BP)_\d+_\d+$", KEY) is not None or
                    re.match(r"^(A|B|AP|BP)_DMAX$", KEY) is not None or

                    # IRAF legacy coordinate-system keywords
                    re.match(r"^WAT\d+_\d{3}$", KEY) is not None or
                    re.match(r"^LTV\d+$", KEY) is not None or
                    re.match(r"^LTM\d+_\d+$", KEY) is not None
                ):
                    delkeys.append(key)
        
        # add extra keys to the deletion list.
        if extra_delkeys_base is not None:
            for key in extra_delkeys_base:
                if key not in delkeys:
                    delkeys.append(key)
        
        # delete the keywords in the deletion list from the base header.
        hdr_b = hdr_base.copy()
        for key in delkeys:
            if key in hdr_b:
                del hdr_b[key]

        # ** extract WCS entries in hdr_wcs
        w_wcs = Read_WCS.RW(hdr_wcs, VERBOSE_LEVEL=VERBOSE_LEVEL)
        wcshdr = w_wcs.to_header(relax=True)

        # remove observation-time metadata.
        for tkey in ['MJD-OBS', 'DATE-OBS']:
            if tkey in wcshdr:
                del wcshdr[tkey]

        # remove IRAF legacy coordinate cards.
        for key in list(wcshdr.keys()):
            KEY = key.upper()
            if (
                re.match(r"^WAT\d+_\d{3}$", KEY) is not None or
                re.match(r"^LTV\d+$", KEY) is not None or
                re.match(r"^LTM\d+_\d+$", KEY) is not None
            ):
                del wcshdr[key]

        # restore WCS reference time (redundant).
        if 'MJDREF' in hdr_wcs:
            if 'MJDREF' not in wcshdr:
                wcshdr['MJDREF'] = (hdr_wcs['MJDREF'], hdr_wcs.comments['MJDREF'])

        # restore the WCS-related keywords missed by WCS.to_header().
        for key in hdr_wcs:
            if key not in wcshdr:
                KEY = key.upper()
                if (
                    # TPV/SIP keywords (to_header may omit zero-valued terms)
                    re.match(r"^PV\d+_\d+$", KEY) is not None or
                    re.match(r"^(A|B|AP|BP)_ORDER$", KEY) is not None or
                    re.match(r"^(A|B|AP|BP)_\d+_\d+$", KEY) is not None or
                    re.match(r"^(A|B|AP|BP)_DMAX$", KEY) is not None
                ):
                    wcshdr[key] = (hdr_wcs[key], hdr_wcs.comments[key])

        # restore missing non-diagonal PC entries.
        if 'PC1_1' in wcshdr:
            if 'PC1_2' not in wcshdr:
                wcshdr['PC1_2'] = (0.0, "MeLOn AutoCompletion")
            if 'PC2_1' not in wcshdr:
                wcshdr['PC2_1'] = (0.0, "MeLOn AutoCompletion")

        # ** combine headers
        hdr_op = hdr_b.copy()
        hdr_op.update(wcshdr)

        # ** run a post correction
        if hdr_wcs.get('CTYPE1') == 'RA---TAN' and 'PV1_0' in hdr_wcs:
            hdr_op['CTYPE1'] = 'RA---TAN'
            hdr_op['CTYPE2'] = 'DEC--TAN'

        return hdr_op
