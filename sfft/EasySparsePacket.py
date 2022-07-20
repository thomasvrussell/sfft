import time
import warnings
import numpy as np
import os.path as pa
from astropy.io import fits
import scipy.ndimage as ndimage
from astropy.table import Column
from sfft.AutoSparsePrep import Auto_SparsePrep
# version: Jul 20, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.2"

class Easy_SparsePacket:
    @staticmethod
    def ESP(FITS_REF, FITS_SCI, FITS_DIFF=None, FITS_Solution=None, ForceConv='AUTO', \
        GKerHW=None, KerHWRatio=2.0, KerHWLimit=(2, 20), KerPolyOrder=2, BGPolyOrder=2, \
        ConstPhotRatio=True, MaskSatContam=False, GAIN_KEY='GAIN', SATUR_KEY='ESATUR', \
        BACK_TYPE='MANUAL', BACK_VALUE='0.0', BACK_SIZE=64, BACK_FILTERSIZE=3, DETECT_THRESH=2.0, \
        DETECT_MINAREA=5, DETECT_MAXAREA=0, DEBLEND_MINCONT=0.005, BACKPHOTO_TYPE='LOCAL', \
        ONLY_FLAGS=[0], BoundarySIZE=30, XY_PriorSelect=None, Hough_FRLowerLimit=0.1, BeltHW=0.2, \
        PS_ELLIPThresh=0.3, MatchTol=None, MatchTolFactor=3.0, MAGD_THRESH=0.12, StarExt_iter=4, \
        XY_PriorBan=None, CheckPostAnomaly=False, PARATIO_THRESH=3.0, BACKEND_4SUBTRACT='Cupy', \
        CUDA_DEVICE_4SUBTRACT='0', NUM_CPU_THREADS_4SUBTRACT=8):
        
        """
        # NOTE: This function is to Perform Sparse-Flavor SFFT for single task:
        #       Pycuda & Cupy backend: do preprocessing on one CPU thread, and do subtraction on one GPU device.
        #       Numpy backend: do preprocessing on one CPU thread, and do subtraction with pyFFTW and Numba 
        #                      using multiple threads (-NUM_CPU_THREADS_4SUBTRACT).

        * Parameters for Sparse-Flavor SFFT [single task]

        # ----------------------------- Computing Enviornment --------------------------------- #

        -BACKEND_4SUBTRACT ['Cupy']         # can be 'Pycuda', 'Cupy' and 'Numpy'. 
                                            # Pycuda backend and Cupy backend require GPU device(s), 
                                            # while 'Numpy' is a pure CPU-based backend for sfft subtraction.
                                            # Cupy backend is even faster than Pycuda, however, it consume more GPU memory.
                                            # NOTE: Cupy backend can support CUDA 11*, while Pycuda does not (issues from Scikit-Cuda).

        -CUDA_DEVICE_4SUBTRACT ['0']        # it specifies certain GPU device (index) to conduct the subtraction task.
                                            # the GPU devices are usually numbered 0 to N-1 (you may use command nvidia-smi to check).
                                            # this argument becomes trivial for Numpy backend.

        -NUM_CPU_THREADS_4SUBTRACT [8]      # it specifies the number of CPU threads used for sfft subtraction in Numpy backend.
                                            # SFFT in Numpy backend has been implemented with pyFFTW and numba, 
                                            # that allow for parallel computing on CPUs. Of course, the Numpy 
                                            # backend is generally much slower than GPU backends.
        
        # ----------------------------- Preprocessing with Source Selection for Image-Masking --------------------------------- #

        # > Configurations for SExtractor

        -GAIN_KEY ['GAIN']          # SExtractor Parameter GAIN_KEY
                                    # i.e., keyword of GAIN in FITS header (of reference & science)

        -SATUR_KEY ['ESATUR']       # SExtractor Parameter SATUR_KEY
                                    # i.e., keyword of effective saturation in FITS header (of reference & science)
                                    # Remarks: one may think 'SATURATE' is a more common keyword name for saturation level.
                                    #          However, note that Sparse-Flavor SFFT requires sky-subtracted images as inputs, 
                                    #          we need deliever the 'effective' saturation level after the sky-subtraction.
                                    #          e.g., set ESATURA = SATURATE - (SKY + 10*SKYSIG)

        -BACK_TYPE ['MANUAL']       # SExtractor Parameter BACK_TYPE = [AUTO or MANUAL].
         
        -BACK_VALUE [0]             # SExtractor Parameter BACK_VALUE (only work for BACK_TYPE='MANUAL')

        -BACK_SIZE [64]             # SExtractor Parameter BACK_SIZE

        -BACK_FILTERSIZE [3]        # SExtractor Parameter BACK_FILTERSIZE

        -DETECT_THRESH [2.0]        # SExtractor Parameter DETECT_THRESH

        -DETECT_MINAREA [5]         # SExtractor Parameter DETECT_MINAREA
        
        -DETECT_MAXAREA [0]         # SExtractor Parameter DETECT_MAXAREA

        -DEBLEND_MINCONT [0.005]    # SExtractor Parameter DEBLEND_MINCONT (typically, 0.001 - 0.005)

        -BACKPHOTO_TYPE ['LOCAL']   # SExtractor Parameter BACKPHOTO_TYPE

        -ONLY_FLAGS [0]             # Restrict SExtractor Output Photometry Catalog by Source FLAGS
                                    # Common FLAGS (Here None means no restrictions on FLAGS):
                                    # 1: aperture photometry is likely to be biased by neighboring sources 
                                    #    or by more than 10% of bad pixels in any aperture
                                    # 2: the object has been deblended
                                    # 4: at least one object pixel is saturated

        -BoundarySIZE [30]          # Restrict SExtractor Output Photometry Catalog by Dropping Sources at Boundary 
                                    # NOTE: This would help to avoid selecting sources too close to image boundary. 

        # > Configurations for Source-Selction

        -XY_PriorSelect [None]      # a Numpy array of pixels coordinates, with shape (N, 2) (e.g., [[x0, y0], [x1, y1], ...])
                                    # this allows sfft to us a prior source selection to solve the subtraction.
                                    # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE

        -Hough_FRLowerLimit [0.1]   # The lower bound of FLUX_RATIO for line feature detection using Hough transformation.
                                    # Setting a proper lower bound can avoid to detect some line features by chance,
                                    # which are not contributed from point sources but resides in the small-FLUX_RATIO region.
                                    # recommended values of Hough_FRLowerLimit: 0.1 ~ 1.0
                                    # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE
    
        -BeltHW [0.2]               # The half-width of point-source-belt detected by Hough Transformation.
                                    # Remarks: if you want to tune this parameter, it is helpful to draw 
                                    #          a figure of MAG_AUTO against FLUX_RADIUS.
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: determine point-source-belt
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: estimate a crude FWHM

        -PS_ELLIPThresh [0.3]       # An additiona Restriction on ELLIPTICITY (ELLIPTICITY < PS_ELLIPThresh) for point sources.
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: determine point-sources
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: estimate a crude FWHM

        -MatchTol [None]            # Given separation tolerance (pix) for source matching 
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: Cross-Match between REF & SCI
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: Cross-Match between REF & SCI AND
                                    #       Cross-Match between the prior selection and mean image coordinates of matched REF-SCI sources.

        -MatchTolFactor [3.0]       # The separation tolerance (pix) for source matching calculated by FWHM,
                                    # MatchTol = np.sqrt((FWHM_REF/MatchTolFactor)**2 + (FWHM_SCI/MatchTolFactor)**2)
                                    # @ Given precise WCS, one can use a high MatchTolFactor ~3.0
                                    # @ For very sparse fields where WCS can be inaccurate, 
                                    #   one can loosen the tolerance with a low MatchTolFactor ~1.0
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: Cross-Match between REF & SCI
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: Cross-Match between REF & SCI AND
                                    #       Cross-Match between the prior selection and mean image coordinates of matched REF-SCI sources.

        -MAGD_THRESH [0.12]         # kicking out the significant variables if the difference of 
                                    # instrument magnitudes (MAG_AUTO) measured on reference and science 
                                    # highly deviate from the median level of the field stars.
                                    # -MAGD_THRESH is the magnitude threshold to define the outliers 
                                    # of the distribution (MAG_AUTO_SCI - MAG_AUTO_REF).
                                    # NOTE: ONLY WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE

        -StarExt_iter [4]           # the image mask is determined by the SExtractor check image SEGMENTATION 
                                    # of the selected sources. note that some pixels (e.g., outskirt region of a galaxy) 
                                    # harbouring signal may be not SExtractor-detectable due to the nature of 
                                    # the thresholding-based detection method (see NoiseChisel paper for more details). 
                                    # we want to include the missing light at outskirt region to contribute to
                                    # parameter-solving process, then a simple mask dilation is introduced.
                                    # -StarExt_iter means the iteration times of the dilation process.
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: mask dilation
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: mask dilation

        -XY_PriorBan [None]         # a Numpy array of pixels coordinates, with shape (N, 2) (e.g., [[x0, y0], [x1, y1], ...])
                                    # this allows us to feed the prior knowledge about the varibility cross the field.
                                    # if you already get a list of variables (transients) and would like SFFT not to select
                                    # them to solve the subtraction, you can tell SFFT their coordinates through -XY_PriorBan.
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [HOUGH-AUTO] MODE :::: mask refinement
                                    # NOTE: IT WORKS FOR Auto-Sparse-Prep [SEMI-AUTO] MODE :::: mask refinement

        # ----------------------------- SFFT Subtraction --------------------------------- #

        -ForceConv ['AUTO']         # it determines which image will be convolved, can be 'REF', 'SCI' and 'AUTO'.
                                    # -ForceConv = 'AUTO' means SFFT will determine the convolution direction according to 
                                    # FWHM_SCI and FWHM_REF: the image with better seeing will be convolved to avoid deconvolution.

        -GKerHW [None]              # the given kernel half-width, None means the kernel size will be 
                                    # automatically determined by -KerHWRatio (to be seeing-related). 

        -KerHWRatio [2.0]           # the ratio between FWHM and the kernel half-width
                                    # KerHW = int(KerHWRatio * Max(FWHM_REF, FWHM_SCI))

        -KerHWLimit [(2, 20)]       # the lower & upper bounds for kernel half-width 
                                    # KerHW is updated as np.clip(KerHW, KerHWLimit[0], KerHWLimit[1]) 
                                    # Remarks: this is useful for a survey since it can constrain the peak GPU memory usage.

        -KerPolyOrder [2]           # Polynomial degree of kernel spatial variation.

        -BGPolyOrder [2]            # Polynomial degree of background spatial variation.
                                    # This argument is trivial for Sparse-Flavor SFFT as input images have been sky subtracted.

        -ConstPhotRatio [True]      # Constant photometric ratio between images? can be True or False
                                    # ConstPhotRatio = True: the sum of convolution kernel is restricted 
                                    #   to be a constant across the field. 
                                    # ConstPhotRatio = False: the flux scaling between images is modeled
                                    #   by a polynomial with degree -KerPolyOrder.

        -MaskSatContam [False]      # Mask saturation-contaminated regions on difference image ? can be True or False
                                    # NOTE the pixels enclosed in the regions are replaced by NaN.

        # ----------------------------- Post Subtraction --------------------------------- #

        -CheckPostAnomaly [False]   # our automatic source selecting (see sfft.Auto_SparsePrep) does not necessarily
                                    # guarantee a complete removal of the sources which cannot be well modeled by SFFT.
                                    # we could inspect the if such missed sources exist by a simple post check on the difference image. 
                                    # for each selected source, we just count the residual flux on the difference image and 
                                    # see if it can be explained by propagated Photon noise.
                                    # NOTE: one can make use of the Post-Anomalies as the Prior-Ban-Sources in a re-subtraction for refinement.

        -PARATIO_THRESH [3.0]       # the ratio as detection threshold of the post anomalies on the difference image.
                                    # NOTE: PARATIO = residual flux / expected Photon noise 
        
        # ----------------------------- Input & Output --------------------------------- #

        -FITS_REF []                # File path of input reference image

        -FITS_SCI []                # File path of input science image

        -FITS_DIFF [None]           # File path of output difference image

        -FITS_Solution [None]       # File path of the solution of the linear system 
                                    # it is an array of (..., a_ijab, ... b_pq, ...)

        # Important Notice:
        #
        # a): if reference is convolved in SFFT, then DIFF = SCI - Convolved_REF.
        #     [difference image is expected to have PSF & flux zero-point consistent with science image]
        #     e.g., -ForceConv='REF' or -ForceConv='AUTO' when reference has better seeing.
        #
        # b): if science is convolved in SFFT, then DIFF = Convolved_SCI - REF
        #     [difference image is expected to have PSF & flux zero-point consistent with reference image]
        #     e.g., -ForceConv='SCI' or -ForceConv='AUTO' when science has better seeing.
        #
        # Remarks: this convention is to guarantee that transients emerge on science image always 
        #          show a positive signal on difference images.

        """
        
        # * Perform Auto Sparse-Prep
        warnings.warn('MeLOn WARNING: Input images for sparse-flavor sfft should be SKY-SUBTRACTED !!!')
        _ASP = Auto_SparsePrep(FITS_REF=FITS_REF, FITS_SCI=FITS_SCI, GAIN_KEY=GAIN_KEY, SATUR_KEY=SATUR_KEY, \
            BACK_TYPE=BACK_TYPE, BACK_VALUE=BACK_VALUE, BACK_SIZE=BACK_SIZE, BACK_FILTERSIZE=BACK_FILTERSIZE, \
            DETECT_THRESH=DETECT_THRESH, DETECT_MINAREA=DETECT_MINAREA, DETECT_MAXAREA=DETECT_MAXAREA, \
            DEBLEND_MINCONT=DEBLEND_MINCONT, BACKPHOTO_TYPE=BACKPHOTO_TYPE, \
            ONLY_FLAGS=ONLY_FLAGS, BoundarySIZE=BoundarySIZE)

        if XY_PriorSelect is None:
            IMAGE_MASK_METHOD = 'HOUGH-AUTO'
            print('MeLOn CheckPoint: TRIGGER Auto-Sparse-Prep [%s] MODE!' %IMAGE_MASK_METHOD)
            SFFTPrepDict = _ASP.HoughAutoMask(Hough_FRLowerLimit=Hough_FRLowerLimit, BeltHW=BeltHW, \
                PS_ELLIPThresh=PS_ELLIPThresh, MatchTol=MatchTol, MatchTolFactor=MatchTolFactor, \
                MAGD_THRESH=MAGD_THRESH, StarExt_iter=StarExt_iter, XY_PriorBan=XY_PriorBan)
        
        if XY_PriorSelect is not None:
            IMAGE_MASK_METHOD = 'SEMI-AUTO'
            print('MeLOn CheckPoint: TRIGGER Auto-Sparse-Prep [%s] MODE!' %IMAGE_MASK_METHOD)
            SFFTPrepDict = _ASP.SemiAutoMask(XY_PriorSelect=XY_PriorSelect, BeltHW=BeltHW, \
                PS_ELLIPThresh=PS_ELLIPThresh, MatchTol=MatchTol, MatchTolFactor=MatchTolFactor, \
                StarExt_iter=StarExt_iter)

        # * Determine ConvdSide & KerHW
        FWHM_REF = SFFTPrepDict['FWHM_REF']
        FWHM_SCI = SFFTPrepDict['FWHM_SCI']

        assert ForceConv in ['AUTO', 'REF', 'SCI']
        if ForceConv == 'AUTO':
            if FWHM_SCI >= FWHM_REF: ConvdSide = 'REF'
            else: ConvdSide = 'SCI'
        else: ConvdSide = ForceConv

        if GKerHW is None:
            FWHM_La = np.max([FWHM_REF, FWHM_SCI])
            KerHW = int(np.clip(KerHWRatio * FWHM_La, KerHWLimit[0], KerHWLimit[1]))
        else: KerHW = GKerHW

        # * Compile Functions in SFFT Subtraction
        from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure

        PixA_REF = SFFTPrepDict['PixA_REF']
        PixA_SCI = SFFTPrepDict['PixA_SCI']

        Tcomp_start = time.time()
        SFFTConfig = SingleSFFTConfigure.SSC(NX=PixA_REF.shape[0], NY=PixA_REF.shape[1], KerHW=KerHW, \
            KerPolyOrder=KerPolyOrder, BGPolyOrder=BGPolyOrder, ConstPhotRatio=ConstPhotRatio, \
            BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, \
            NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
        print('\nMeLOn Report: Compiling Functions in SFFT Subtraction Takes [%.3f s]' %(time.time() - Tcomp_start))
       
        # * Perform SFFT Subtraction
        from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract

        SatMask_REF = SFFTPrepDict['REF-SAT-Mask']
        SatMask_SCI = SFFTPrepDict['SCI-SAT-Mask']
        NaNmask_U = SFFTPrepDict['Union-NaN-Mask']
        PixA_mREF = SFFTPrepDict['PixA_mREF']
        PixA_mSCI = SFFTPrepDict['PixA_mSCI']

        if ConvdSide == 'REF':
            PixA_mI, PixA_mJ = PixA_mREF, PixA_mSCI
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_REF.copy(), PixA_SCI.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_REF, PixA_SCI
            if MaskSatContam: 
                ContamMask_I = SatMask_REF
                ContamMask_J = SatMask_SCI
            else: ContamMask_I = None

        if ConvdSide == 'SCI':
            PixA_mI, PixA_mJ = PixA_mSCI, PixA_mREF
            if NaNmask_U is not None:
                PixA_I, PixA_J = PixA_SCI.copy(), PixA_REF.copy()
                PixA_I[NaNmask_U] = PixA_mI[NaNmask_U]
                PixA_J[NaNmask_U] = PixA_mJ[NaNmask_U]
            else: PixA_I, PixA_J = PixA_SCI, PixA_REF
            if MaskSatContam: 
                ContamMask_I = SatMask_SCI
                ContamMask_J = SatMask_REF
            else: ContamMask_I = None

        Tsub_start = time.time()
        _tmp = GeneralSFFTSubtract.GSS(PixA_I=PixA_I, PixA_J=PixA_J, PixA_mI=PixA_mI, PixA_mJ=PixA_mJ, \
            SFFTConfig=SFFTConfig, ContamMask_I=ContamMask_I, BACKEND_4SUBTRACT=BACKEND_4SUBTRACT, \
            CUDA_DEVICE_4SUBTRACT=CUDA_DEVICE_4SUBTRACT, NUM_CPU_THREADS_4SUBTRACT=NUM_CPU_THREADS_4SUBTRACT)
        Solution, PixA_DIFF, ContamMask_CI = _tmp
        if MaskSatContam:
            ContamMask_DIFF = np.logical_or(ContamMask_CI, ContamMask_J)
        print('\nMeLOn Report: SFFT Subtraction Takes [%.3f s]' %(time.time() - Tsub_start))

        if CheckPostAnomaly:
            AstSEx_SS = SFFTPrepDict['SExCatalog-SubSource']
            SFFTLmap = SFFTPrepDict['SFFT-LabelMap']
            
            # ** Only consider the valid (= non-PriorBanned) SubSources
            if 'MASK_PriorBan' in AstSEx_SS.colnames:
                nPBMASK_SS = ~np.array(AstSEx_SS['MASK_PriorBan'])
                AstSEx_vSS = AstSEx_SS[nPBMASK_SS]
            else: AstSEx_vSS = AstSEx_SS
            
            # ** Estimate expected variance of valid SubSources on the difference
            #    NOTE assume that all of the valid SubSources are stationary.
            Gr = np.array(AstSEx_vSS['FLUXERR_AUTO_REF'])
            Gs = np.array(AstSEx_vSS['FLUXERR_AUTO_SCI'])
            if ConvdSide == 'REF':
                dm = np.median(AstSEx_vSS['MAG_AUTO_SCI'] - AstSEx_vSS['MAG_AUTO_REF'])
                VARr = (Gr/(10**(dm/2.5)))**2
                VARs = Gs**2
            if ConvdSide == 'SCI':
                dm = np.median(AstSEx_vSS['MAG_AUTO_REF'] - AstSEx_vSS['MAG_AUTO_SCI'])
                VARr = Gr**2
                VARs = (Gs/(10**(dm/2.5)))**2
            ExpDVAR_vSS = VARr + VARs

            # ** Measure the ratios of valid SubSources on the difference for detecting the PostAnomaly SubSources
            SEGL_vSS = np.array(AstSEx_vSS['SEGLABEL']).astype(int)
            DFSUM_vSS = ndimage.labeled_comprehension(PixA_DIFF, SFFTLmap, SEGL_vSS, np.sum, float, 0.0)
            PARATIO_vSS = DFSUM_vSS / np.sqrt(np.clip(ExpDVAR_vSS, a_min=0.1, a_max=None))
            PAMASK_vSS = np.abs(PARATIO_vSS) > PARATIO_THRESH

            pamessage = 'Find [%d] PostAnomaly SubSources (%.2f sigma) ' %(np.sum(PAMASK_vSS), PARATIO_THRESH)
            pamessage += 'out of [%d] Valid (= non-PriorBanned) SubSources!\n' %(len(AstSEx_vSS))
            pamessage += 'P.S. There are [%d] Invalid (= PriorBanned) SubSources!' %(len(AstSEx_SS) - len(AstSEx_vSS))
            print('\nMeLOn CheckPoint: %s' %pamessage)
            
            # ** Record the results (decorate AstSEx_SS in SFFTPrepDict)
            if XY_PriorBan is not None:
                ExpDVAR_SS = np.nan * np.ones(len(AstSEx_SS))       # NOTE Prior-Ban is trivial NaN
                DFSUM_SS = np.nan * np.ones(len(AstSEx_SS))         # NOTE Prior-Ban is trivial NaN
                PARATIO_SS = np.nan * np.ones(len(AstSEx_SS))       # NOTE Prior-Ban is trivial NaN
                PAMASK_SS = np.zeros(len(AstSEx_SS)).astype(bool)   # NOTE Prior-Ban is trivial False
                
                ExpDVAR_SS[nPBMASK_SS] = ExpDVAR_vSS
                DFSUM_SS[nPBMASK_SS] = DFSUM_vSS
                PARATIO_SS[nPBMASK_SS] = PARATIO_vSS
                PAMASK_SS[nPBMASK_SS] = PAMASK_vSS
            else: 
                ExpDVAR_SS = ExpDVAR_vSS
                DFSUM_SS = DFSUM_vSS
                PARATIO_SS = PARATIO_vSS
                PAMASK_SS = PAMASK_vSS

            AstSEx_SS.add_column(Column(ExpDVAR_SS, name='ExpDVAR_PostAnomaly'))
            AstSEx_SS.add_column(Column(DFSUM_SS, name='DFSUM_PostAnomaly'))
            AstSEx_SS.add_column(Column(PARATIO_SS, name='RATIO_PostAnomaly'))
            AstSEx_SS.add_column(Column(PAMASK_SS, name='MASK_PostAnomaly'))

        # * Modifications on the difference image
        #   a) when REF is convolved, DIFF = SCI - Conv(REF)
        #       PSF_DIFF is coincident with SCI, transients on SCI are positive signal in DIFF.
        #   b) when SCI is convolved, DIFF = Conv(SCI) - REF
        #       PSF_DIFF is coincident with REF, transients on SCI are still positive signal in DIFF.

        if NaNmask_U is not None:
            # ** Mask Union-NaN region
            PixA_DIFF[NaNmask_U] = np.nan
        if MaskSatContam:
            # ** Mask Saturate-Contaminate region 
            PixA_DIFF[ContamMask_DIFF] = np.nan
        if ConvdSide == 'SCI': 
            # ** Flip difference when science is convolved
            PixA_DIFF = -PixA_DIFF

        # * Save difference image
        if FITS_DIFF is not None:
            _hdl = fits.open(FITS_SCI)
            _hdl[0].data[:, :] = PixA_DIFF.T
            _hdl[0].header['NAME_REF'] = (pa.basename(FITS_REF), 'MeLOn: SFFT')
            _hdl[0].header['NAME_SCI'] = (pa.basename(FITS_SCI), 'MeLOn: SFFT')
            _hdl[0].header['FWHM_REF'] = (FWHM_REF, 'MeLOn: SFFT')
            _hdl[0].header['FWHM_SCI'] = (FWHM_SCI, 'MeLOn: SFFT')
            _hdl[0].header['KERORDER'] = (KerPolyOrder, 'MeLOn: SFFT')
            _hdl[0].header['BGORDER'] = (BGPolyOrder, 'MeLOn: SFFT')
            _hdl[0].header['CPHOTR'] = (str(ConstPhotRatio), 'MeLOn: SFFT')
            _hdl[0].header['KERHW'] = (KerHW, 'MeLOn: SFFT')
            _hdl[0].header['CONVD'] = (ConvdSide, 'MeLOn: SFFT')
            _hdl.writeto(FITS_DIFF, overwrite=True)
            _hdl.close()
        
        # * Save solution array
        if FITS_Solution is not None:
            phdu = fits.PrimaryHDU()
            phdu.header['N0'] = (SFFTConfig[0]['N0'], 'MeLOn: SFFT')
            phdu.header['N1'] = (SFFTConfig[0]['N1'], 'MeLOn: SFFT')
            phdu.header['DK'] = (SFFTConfig[0]['DK'], 'MeLOn: SFFT')
            phdu.header['DB'] = (SFFTConfig[0]['DB'], 'MeLOn: SFFT')
            phdu.header['L0'] = (SFFTConfig[0]['L0'], 'MeLOn: SFFT')
            phdu.header['L1'] = (SFFTConfig[0]['L1'], 'MeLOn: SFFT')
            
            phdu.header['FIJ'] = (SFFTConfig[0]['Fij'], 'MeLOn: SFFT')
            phdu.header['FAB'] = (SFFTConfig[0]['Fab'], 'MeLOn: SFFT')
            phdu.header['FPQ'] = (SFFTConfig[0]['Fpq'], 'MeLOn: SFFT')
            phdu.header['FIJAB'] = (SFFTConfig[0]['Fijab'], 'MeLOn: SFFT')

            PixA_Solution = Solution.reshape((-1, 1))
            phdu.data = PixA_Solution.T
            fits.HDUList([phdu]).writeto(FITS_Solution, overwrite=True)

        return SFFTPrepDict, Solution, PixA_DIFF
