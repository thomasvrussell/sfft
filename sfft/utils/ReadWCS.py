import warnings
import numpy as np
from astropy.wcs import WCS, FITSFixedWarning
# version: Mar 17, 2023

__last_update__ = "2025-09-29"
__author__ = "Lei Hu <leihu@andrew.cmu.edu> & Rob Knop <raknop@lbl.gov>"

class Read_WCS:
    @staticmethod
    def determine_cdkey( hdr ):
        if 'CD1_1' in hdr:
            return 'CD'
        if 'PC1_1' not in hdr:
            raise ValueError( "Cannot figure out if WCS uses CD or PC." )
        if ( hdr['CDELT1'] != 1.0 ) or ( hdr['CDELT2'] != 1.0 ):
            raise ValueError( "CDELT1 and/or CDELT2 is not 1, but this code will assume it is." )
        return 'PC'

    @staticmethod
    def read_linear_part_of_wcs(hdr_wcs):
        """
        # * Note on the CD matrix transformation:
        # The sky coordinate (x, y) relative to reference point, a.k.a, intermediate world coordinate, can be connected with 
        # the image coordinate (u, v) relative to reference point, a.k.a, intermediate pixel coordinate by the CD matrix:
        #     [x]   [CD1_1 CD1_2] [u]
        #     [y] = [CD2_1 CD2_2] [v]
        # where CD1_1, CD1_2, CD2_1, CD2_2 are stored in the FITS header.
        #
        """
        N0 = int(hdr_wcs["NAXIS1"])
        N1 = int(hdr_wcs["NAXIS2"])

        CRPIX1 = float(hdr_wcs["CRPIX1"])
        CRPIX2 = float(hdr_wcs["CRPIX2"])

        CRVAL1 = float(hdr_wcs["CRVAL1"])
        CRVAL2 = float(hdr_wcs["CRVAL2"])

        if "LONPOLE" not in hdr_wcs:
            _warnmsg = "keyword LONPOLE not found in the header, set to default value 180.0"
            # print("MeLOn WARNING: %s" % _warnmsg)
            LONPOLE = 180.0    # default value
        else:
            LONPOLE = float(hdr_wcs["LONPOLE"])

        KEYDICT = {
            "N0": N0, "N1": N1, 
            "CRPIX1": CRPIX1, "CRPIX2": CRPIX2,
            "CRVAL1": CRVAL1, "CRVAL2": CRVAL2,
            "LONPOLE": LONPOLE
        }

        if "CD1_1" in hdr_wcs:
            CD1_1 = hdr_wcs["CD1_1"]
            CD1_2 = hdr_wcs["CD1_2"] if "CD1_2" in hdr_wcs else 0.
            CD2_1 = hdr_wcs["CD2_1"] if "CD2_1" in hdr_wcs else 0.
            CD2_2 = hdr_wcs["CD2_2"]
        
        elif "PC1_1" in hdr_wcs:
            PC1_1 = hdr_wcs["PC1_1"]
            PC1_2 = hdr_wcs["PC1_2"] if "PC1_2" in hdr_wcs else 0.
            PC2_1 = hdr_wcs["PC2_1"] if "PC2_1" in hdr_wcs else 0.
            PC2_2 = hdr_wcs["PC2_2"]

            assert "CDELT1" in hdr_wcs
            assert "CDELT2" in hdr_wcs
            CDELT1 = hdr_wcs["CDELT1"]
            CDELT2 = hdr_wcs["CDELT2"]

            CD1_1 = CDELT1 * PC1_1
            CD1_2 = CDELT1 * PC1_2
            CD2_1 = CDELT2 * PC2_1
            CD2_2 = CDELT2 * PC2_2

        else:
            assert "CDELT1" in hdr_wcs
            assert "CDELT2" in hdr_wcs
            CDELT1 = hdr_wcs["CDELT1"]
            CDELT2 = hdr_wcs["CDELT2"]
            
            CD1_1 = CDELT1
            CD1_2 = 0.
            CD2_1 = 0.
            CD2_2 = CDELT2

        CD = np.array([
            [CD1_1, CD1_2], 
            [CD2_1, CD2_2]
        ], dtype=np.float64)
        
        return KEYDICT, CD


    @staticmethod
    def read_cd_wcs(hdr_wcs):
        if ( hdr_wcs["CTYPE1"] != "RA---TAN" ) or ( hdr_wcs["CTYPE2"] != 'DEC--TAN' ):
            raise ValueError( f"Error, read_cd_wcs expects header CTYPE1='RA---TAN' and CTYPE2='DEC--TAN'; "
                              f"instead, got CTYPE1='{hdr_wcs['CTYPE1']} and CTYPE2='{hdr_wcs['CTYPE2']}" )
        return Read_WCS.read_linear_part_of_wcs( hdr_wcs )


    @staticmethod
    def read_sip_wcs(hdr_wcs):
        """ Read TAN-SIP WCS information from FITS header """
        assert hdr_wcs["CTYPE1"] == "RA---TAN-SIP"
        assert hdr_wcs["CTYPE2"] == "DEC--TAN-SIP"

        KEYDICT, CD = Read_WCS.read_linear_part_of_wcs( hdr_wcs )

        A_SIP = np.zeros((A_ORDER+1, A_ORDER+1), dtype=np.float64)
        B_SIP = np.zeros((B_ORDER+1, B_ORDER+1), dtype=np.float64)

        for p in range(A_ORDER + 1):
            for q in range(0, A_ORDER - p + 1):
                keyword = f"A_{p}_{q}"
                if keyword in hdr_wcs:
                    A_SIP[p, q] = hdr_wcs[keyword]
        
        for p in range(B_ORDER + 1):
            for q in range(0, B_ORDER - p + 1):
                keyword = f"B_{p}_{q}"
                if keyword in hdr_wcs:
                    B_SIP[p, q] = hdr_wcs[keyword]
        
        return KEYDICT, CD, A_SIP, B_SIP

    @staticmethod
    def RW(hdr, VERBOSE_LEVEL=2):

        """
        # Remarks on the WCS in FITS header and astropy.WCS
        # Case A:
        #    FITS header:
        #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
        #        CD1_1 = ...           CD1_2 = ...
        #        CD2_1 = ...           CD2_2 = ...
        #        NOTE: NO DISTORTION TERMS!
        #
        #    astropy.WCS:
        #        # print(WCS(phr).to_header())
        #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
        #        PC1_1 = ...           PC1_2 = ...
        #        PC2_1 = ...           PC2_2 = ...
        #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
        #
        # Case B (Distorted tangential with the TPV FITS convention, added officially to the FITS registry in Aug. 2012):
        #    FITS header:
        #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
        #        CD1_1 = ...           CD1_2 = ...
        #        CD2_1 = ...           CD2_2 = ...
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #
        #    astropy.WCS:
        #        # print(WCS(phr).to_header())
        #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
        #        PC1_1 = ...           PC1_2 = ...
        #        PC2_1 = ...           PC2_2 = ...
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
        #
        # Case B' (Distorted tangential with Greisen & Calabretta's 2000 draft, obsoleted):
        #    FITS header:
        #        CTYPE1 = 'RA---TAN'   CTYPE2 = 'DEC--TAN'
        #        CD1_1 = ...           CD1_2 = ...
        #        CD2_1 = ...           CD2_2 = ...
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #
        #    WARNING: To avoid Astropy.WCS incompatibility, one should make corrections
        #        as follows before reading by Astropy.WCS
        #        phr['CTYPE1'] = 'RA---TPV'
        #        phr['CTYPE1'] = 'DEC--TPV'
        #        w = WCS(phr)
        #
        #    astropy.WCS:
        #        # print(w.to_header())
        #        CTYPE1 = 'RA---TPV'   CTYPE2 = 'DEC--TPV'
        #        PC1_1 = ...           PC1_2 = ...
        #        PC2_1 = ...           PC2_2 = ...
        #        PV1_0 = ...           PV1_1 = ...
        #        ...
        #        NOTE: if CD1_2 = CD2_1 = 0.0, then PC1_2 and PC2_1 will be absent.
        #
        # Case C:
        #    FITS header:
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
        #    astropy.WCS:
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
        #
        # P.S. One may use PC1_1 to replace CD1_1 in FITS header for above cases, not recommended though.
        #
        """

        with warnings.catch_warnings():
            if VERBOSE_LEVEL in [0, 1]: behavior = 'ignore'
            if VERBOSE_LEVEL in [2]: behavior = 'default'
            warnings.filterwarnings(behavior, category=FITSFixedWarning)

            if hdr['CTYPE1'] == 'RA---TAN' and 'PV1_0' in hdr:
                _hdr = hdr.copy()
                _hdr['CTYPE1'] = 'RA---TPV'
                _hdr['CTYPE2'] = 'DEC--TPV'
            else: _hdr = hdr
            w = WCS(_hdr)

        return w
