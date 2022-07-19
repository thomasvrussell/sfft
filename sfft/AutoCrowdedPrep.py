import os
import sep
import fastremap
import numpy as np
import os.path as pa
from astropy.io import fits
import scipy.ndimage as ndimage
from astropy.stats import sigma_clipped_stats
from sfft.utils.pyAstroMatic.PYSEx import PY_SEx
# version: Jul 19, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.2"

class Auto_CrowdedPrep:
    def __init__(self, FITS_REF, FITS_SCI, GAIN_KEY='GAIN', SATUR_KEY='SATURATE', BACK_TYPE='AUTO', BACK_VALUE='0.0', \
        BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=5.0, DETECT_MINAREA=5, DETECT_MAXAREA=0, \
        DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', ONLY_FLAGS=None, BoundarySIZE=0.0):

        self.FITS_REF = FITS_REF
        self.FITS_SCI = FITS_SCI

        self.GAIN_KEY = GAIN_KEY
        self.SATUR_KEY = SATUR_KEY

        self.BACK_TYPE = BACK_TYPE
        self.BACK_VALUE = BACK_VALUE
        self.BACK_SIZE = BACK_SIZE
        self.BACK_FILTERSIZE = BACK_FILTERSIZE

        self.DETECT_THRESH = DETECT_THRESH
        self.DETECT_MINAREA = DETECT_MINAREA
        self.DETECT_MAXAREA = DETECT_MAXAREA
        self.DEBLEND_MINCONT = DEBLEND_MINCONT
        self.BACKPHOTO_TYPE = BACKPHOTO_TYPE

        self.ONLY_FLAGS = ONLY_FLAGS
        self.BoundarySIZE = BoundarySIZE
    
    def AutoMask(self, BACK_SIZE_SUPER=128, StarExt_iter=2, PriorBanMask=None):

        # * Generate Super-Background for REF & SCI
        PixA_REF = fits.getdata(self.FITS_REF, ext=0).T
        PixA_SCI = fits.getdata(self.FITS_SCI, ext=0).T

        if not PixA_REF.flags['C_CONTIGUOUS']:
            PixA_REF = np.ascontiguousarray(PixA_REF, np.float64)
        else: PixA_REF = PixA_REF.astype(np.float64)

        if not PixA_SCI.flags['C_CONTIGUOUS']:
            PixA_SCI = np.ascontiguousarray(PixA_SCI, np.float64)
        else: PixA_SCI = PixA_SCI.astype(np.float64)

        # NOTE: sep requires input_array.flags['C_CONTIGUOUS'] is True & input_array.dtype.byteorder is '='
        PixA_SBG_REF = sep.Background(PixA_REF, bw=BACK_SIZE_SUPER, bh=BACK_SIZE_SUPER, fw=3, fh=3).back()        
        PixA_SBG_SCI = sep.Background(PixA_SCI, bw=BACK_SIZE_SUPER, bh=BACK_SIZE_SUPER, fw=3, fh=3).back()

        # * Generate Saturation-Mask for REF & SCI
        def GenSatMask(FITS_obj):

            # ** trigger a very-cold SExtractor
            PL = ['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FLUXERR_AUTO', 'FLUX_MAX', 'FWHM_IMAGE']
            PYSEX_OP = PY_SEx.PS(FITS_obj=FITS_obj, PL=PL, GAIN_KEY=self.GAIN_KEY, SATUR_KEY=self.SATUR_KEY, \
                BACK_TYPE=self.BACK_TYPE, BACK_VALUE=self.BACK_VALUE, BACK_SIZE=self.BACK_SIZE, \
                BACK_FILTERSIZE=self.BACK_FILTERSIZE, DETECT_THRESH=self.DETECT_THRESH, DETECT_MINAREA=self.DETECT_MINAREA, \
                DETECT_MAXAREA=self.DETECT_MAXAREA, DEBLEND_MINCONT=self.DEBLEND_MINCONT, BACKPHOTO_TYPE=self.BACKPHOTO_TYPE, \
                CHECKIMAGE_TYPE='SEGMENTATION', AddRD=False, ONLY_FLAGS=self.ONLY_FLAGS, \
                XBoundary=self.BoundarySIZE, YBoundary=self.BoundarySIZE, MDIR=None)
            AstSEx, PixA_SEG = PYSEX_OP[0], PYSEX_OP[1][0].astype(int)
            FWHM = sigma_clipped_stats(AstSEx['FWHM_IMAGE'], sigma=3.0, maxiters=5)[1]

            # ** determine initial saturation region
            SATLEVEL = fits.getheader(FITS_obj, ext=0)[self.SATUR_KEY]
            AstSEx_SAT = AstSEx[AstSEx['FLUX_MAX'] >= SATLEVEL]
            Mappings = {}
            for label in AstSEx_SAT['SEGLABEL']: 
                Mappings[label] = -label
            fastremap.remap(PixA_SEG, Mappings, preserve_missing_labels=True, in_place=True)
            SatMask = PixA_SEG < 0
            
            # ** refine the saturation region
            #     SExtractor tend to label some outskirt-islands as part of a saturated source.
            XY_SAT = np.array([AstSEx_SAT['X_IMAGE'], AstSEx_SAT['Y_IMAGE']]).T            
            Lmap = ndimage.measurements.label(SatMask)[0]
            SATL = Lmap[((XY_SAT[:, 0] - 0.5).astype(int), (XY_SAT[:, 1] - 0.5).astype(int))]
            SATL = list(set(SATL).difference({0}))          # 0 is background
            SatMask = np.in1d(Lmap.flatten(), SATL).reshape(Lmap.shape)    # WARNING NOTE UPDATE

            struct0 = ndimage.generate_binary_structure(2, 1)
            struct = ndimage.iterate_structure(struct0, StarExt_iter)
            SatMask = ndimage.grey_dilation(SatMask, footprint=struct)     # WARNING NOTE UPDATE
            SatPROP = np.sum(SatMask) / (SatMask.shape[0] * SatMask.shape[1])
            print('MeLOn CheckPoint: FILE %s | NSAT [%d] & SATURATE-Proportion [%s]' \
                   %(pa.basename(FITS_obj), len(AstSEx_SAT), '{:.2%}'.format(SatPROP)))

            return SATLEVEL, FWHM, SatMask

        SATLEVEL_REF, FWHM_REF, SatMask_REF = GenSatMask(self.FITS_REF)
        SATLEVEL_SCI, FWHM_SCI, SatMask_SCI = GenSatMask(self.FITS_SCI)

        # * Define ProhibitedZone
        NaNmask_U = None
        if PriorBanMask is None:
            ProZone = np.logical_or(SatMask_REF, SatMask_SCI)
        else: ProZone = np.logical_or.reduce((PriorBanMask, SatMask_REF, SatMask_SCI))
        
        NaNmask_REF = np.isnan(PixA_REF)
        NaNmask_SCI = np.isnan(PixA_SCI)
        if NaNmask_REF.any() or NaNmask_SCI.any():
            NaNmask_U = np.logical_or(NaNmask_REF, NaNmask_SCI)
            ProZone[NaNmask_U] = True

        # * Make ActiveMask > PixA_mREF & PixA_mSCI
        ActiveMask = ~ProZone
        PixA_mREF = PixA_REF.copy()
        PixA_mSCI = PixA_SCI.copy()
        PixA_mREF[ProZone] = PixA_SBG_REF[ProZone]      # NOTE REF is not sky-subtracted yet
        PixA_mSCI[ProZone] = PixA_SBG_SCI[ProZone]      # NOTE SCI is not sky-subtracted yet

        # NOTE: The preparation can guarantee that mREF & mSCI are NaN-Free !
        ActivePROP = np.sum(ActiveMask) / (ActiveMask.shape[0] * ActiveMask.shape[1])
        print('MeLOn CheckPoint: ActiveMask Pixel Proportion [%s]' %('{:.2%}'.format(ActivePROP)))

        # * Create sfft-prep master dictionary
        SFFTPrepDict = {}
        SFFTPrepDict['PixA_REF'] = PixA_REF
        SFFTPrepDict['PixA_SCI'] = PixA_SCI
        SFFTPrepDict['Union-NaN-Mask'] = NaNmask_U

        SFFTPrepDict['SATLEVEL_REF'] = SATLEVEL_REF
        SFFTPrepDict['SATLEVEL_SCI'] = SATLEVEL_SCI
        SFFTPrepDict['FWHM_REF'] = FWHM_REF
        SFFTPrepDict['FWHM_SCI'] = FWHM_SCI

        SFFTPrepDict['REF-SAT-Mask'] = SatMask_REF
        SFFTPrepDict['SCI-SAT-Mask'] = SatMask_SCI
        SFFTPrepDict['Active-Mask'] = ActiveMask
        SFFTPrepDict['PixA_mREF'] = PixA_mREF
        SFFTPrepDict['PixA_mSCI'] = PixA_mSCI

        return SFFTPrepDict
