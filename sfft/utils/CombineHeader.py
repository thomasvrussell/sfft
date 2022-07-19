from astropy.wcs import WCS

class Combine_Header:
    @staticmethod
    def CH(hdr_base, hdr_wcs):

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
