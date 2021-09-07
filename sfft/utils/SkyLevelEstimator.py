import numpy as np

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

class SkyLevel_Estimator: 
    @staticmethod
    def SLE(PixA_obj):
        
        # D. Jones - 2/13/14 | from PythonPhot package
        # equally idl commend: idl -e "sky,readfits('rd.b160504.000147.fits')"
        """This code is from the IDL Astronomy Users Library"""
        def mmm(sky_vector, highbad=False, debug=False, readnoise=False, \
            nsky=False, integer="discrete", mxiter=50, minsky=20, nan=True):

            """Estimate the sky background in a stellar contaminated field.

            MMM assumes that contaminated sky pixel values overwhelmingly display 
            POSITIVE departures from the true value.  Adapted from DAOPHOT 
            routine of the same name.
            
            CALLING SEQUENCE:
                skymod,sigma,skew = mmm.mmm( sky, highbad= , readnoise=, debug=, 
                                            minsky=, nsky=, integer=)
            
            INPUTS:
                sky - Array or Vector containing sky values.  This version of
                        MMM does not require SKY to be sorted beforehand.
            
            RETURNS:
                skymod - Scalar giving estimated mode of the sky values
                sigma -  Scalar giving standard deviation of the peak in the sky
                        histogram.  If for some reason it is impossible to derive
                        skymod, then SIGMA = -1.0
                skew -   Scalar giving skewness of the peak in the sky histogram
            
                If no output variables are supplied or if "debug" is set
                then the values of skymod, sigma and skew will be printed.
            
            OPTIONAL KEYWORD INPUTS:
                highbad - scalar value of the (lowest) "bad" pixel level (e.g. cosmic 
                            rays or saturated pixels) If not supplied, then there is 
                            assumed to be no high bad pixels.
                minsky - Integer giving mininum number of sky values to be used.   MMM
                            will return an error if fewer sky elements are supplied.
                            Default = 20.
                maxiter - integer giving maximum number of iterations allowed,default=50
                readnoise - Scalar giving the read noise (or minimum noise for any 
                            pixel).  Normally, MMM determines the (robust) median by 
                            averaging the central 20% of the sky values.  In some cases
                            where the noise is low, and pixel values are quantized a
                            larger fraction may be needed.  By supplying the optional
                            read noise parameter, MMM is better able to adjust the
                            fraction of pixels used to determine the median.                
                integer - Set this keyword if the  input SKY vector only contains
                            discrete integer values.  This keyword is only needed if the
                            SKY vector is of type float or double precision, but contains 
                            only discrete integer values.  (Prior to July 2004, the
                            equivalent of /INTEGER was set for all data types)
                debug -   If this keyword is set and non-zero, then additional 
                            information is displayed at the terminal.

            OPTIONAL OUTPUT KEYWORD:
                nsky - Integer scalar giving the number of pixels actually used for the
                        sky computation (after outliers have been removed).

            NOTES:
                (1) Program assumes that low "bad" pixels (e.g. bad CCD columns) have
                    already been deleted from the SKY vector.
                (2) MMM was updated in June 2004 to better match more recent versions
                    of DAOPHOT.
                (3) Does not work well in the limit of low Poisson integer counts
                (4) MMM may fail for strongly skewed distributions.

            METHOD:
                The algorithm used by MMM consists of roughly two parts:
                (1) The average and sigma of the sky pixels is computed.  These values
                        are used to eliminate outliers, i.e. values with a low probability
                        given a Gaussian with specified average and sigma.  The average
                        and sigma are then recomputed and the process repeated up to 20
                        iterations.
                (2) The amount of contamination by stars is estimated by comparing the 
                        mean and median of the remaining sky pixels.  If the mean is larger
                        than the median then the true sky value is estimated by
                        3*median - 2*mean
                    
            REVISION HISTORY:
                Adapted to IDL from 1986 version of DAOPHOT in STSDAS        W. Landsman, STX           Feb,      1987
                Added HIGHBAD keyword                                        W. Landsman                January,  1991
                Fixed occasional problem with integer inputs                 W. Landsman                Feb,      1994
                Avoid possible 16 bit integer overflow                       W. Landsman                November, 2001
                Added READNOISE, NSKY keywords,  new median computation      W. Landsman                June,     2004
                Added INTEGER keyword                                        W. Landsman                July,     2004
                Improve numerical precision                                  W. Landsman                October,  2004
                Fewer aborts on strange input sky histograms                 W. Landsman                October,  2005
                Added /SILENT keyword                                                                   November, 2005
                Fix too many /CON keywords to MESSAGE                        W.L.                       December, 2005
                Fix bug introduced June 2004 removing outliers               N. Cunningham/W. Landsman  January,  2006
                    when READNOISE not set
                Make sure that MESSAGE never aborts                          W. Landsman                January,  2008
                Add mxiter keyword and change default to 50                  W. Landsman                August,   2011
                Added MINSKY keyword                                         W.L.                       December, 2011
                Converted to Python                                          D. Jones                   January,  2014
            """

            if nan: sky_vector = sky_vector[np.where(sky_vector == sky_vector)]
            nsky = len( sky_vector )            #Get number of sky elements 
        
            if nsky < minsky:
                sigma=-1.0 ;  skew = 0.0; skymod = np.nan
                print('ERROR -Input vector must contain at least '+str(minsky)+' elements')
                return(skymod,sigma,skew)
        
            nlast = nsky-1                        #Subscript of last pixel in SKY array
            if debug:
                print('Processing '+str(nsky) + ' element array')
            sz_sky = np.shape(sky_vector)

            sky = np.sort(sky_vector)    #Sort SKY in ascending values

            skymid = 0.5*sky[int((nsky-1)/2)] + 0.5*sky[int(nsky/2)]  #Median value of all sky values
            
            cut1 = np.min( [skymid-sky[0],sky[nsky-1] - skymid] ) 
            if highbad: 
                cut1[np.where(cut1 > highbad - skymid)[0]] = highbad - skymid
            cut2 = skymid + cut1
            cut1 = skymid - cut1

            # Select the pixels between Cut1 and Cut2

            good = np.where( (sky <= cut2) & (sky >= cut1))[0]
            Ngood = len(good)

            if ( Ngood == 0 ):
                sigma=-1.0 ;  skew = 0.0; skymod = 0.0   
                print('ERROR - No sky values fall within ' + str(cut1) + \
            ' and ' + str(cut2))
                return(skymod,sigma,skew)
        
            delta = sky[good] - skymid  #Subtract median to improve arithmetic accuracy
            sum = np.sum(delta.astype('float64'))
            sumsq = np.sum(delta.astype('float64')**2)

            maximm = np.max( good) ; minimm = np.min(good)  # Highest value accepted at upper end of vector
            minimm = minimm -1               #Highest value reject at lower end of vector

            # Compute mean and sigma (from the first pass).

            skymed = 0.5*sky[int((minimm+maximm+1)/2)] + 0.5*sky[int((minimm+maximm)/2 + 1)] #median 
            skymn = sum/(maximm-minimm)                            #mean       
            sigma = np.sqrt(sumsq/(maximm-minimm)-skymn**2)             #sigma          
            skymn = skymn + skymid         #Add median which was subtracted off earlier 

            #    If mean is less than the mode, then the contamination is slight, and the
            #    mean value is what we really want.
            #    skymod =  (skymed < skymn) ? 3.*skymed - 2.*skymn : skymn
            if skymed < skymn:
                skymod = 3.*skymed - 2.*skymn
            else: skymod = skymn

            # Rejection and recomputation loop:

            niter = 0
            clamp = 1
            old = 0
            # START_LOOP:
            redo = True
            while redo:
                niter = niter + 1                     
                if ( niter > mxiter ):
                    sigma=-1.0 ;  skew = 0.0   
                    print('ERROR - Too many ('+str(mxiter) + ') iterations,' + \
                            ' unable to compute sky')
                    #            import pdb; pdb.set_trace()
                    return(skymod,sigma,skew)

                if ( maximm-minimm < minsky ):    #Error? 

                    sigma = -1.0 ;  skew = 0.0   
                    print('ERROR - Too few ('+str(maximm-minimm) +  \
                            ') valid sky elements, unable to compute sky')
                    return(skymod,sigma,skew)

                # Compute Chauvenet rejection criterion.

                r = np.log10( float( maximm-minimm ) )      
                r = np.max( [ 2., ( -0.1042*r + 1.1695)*r + 0.8895 ] )

                # Compute rejection limits (symmetric about the current mode).

                cut = r*sigma + 0.5*np.abs(skymn-skymod)   
                #    if integer: cut = cut > 1.5 
                cut1 = skymod - cut   ;    cut2 = skymod + cut

                # 
                # Recompute mean and sigma by adding and/or subtracting sky values
                # at both ends of the interval of acceptable values.
            
                redo = False
                newmin = minimm             
                if sky[newmin+1] >= cut1: tst_min = 1      #Is minimm+1 above current CUT?
                else: tst_min = 0
                if (newmin == -1) and tst_min: done = 1    #Are we at first pixel of SKY?
                else: done = 0
                if not done:
                    if newmin > 0: skyind = newmin
                    else: skyind = 0
                    if (sky[skyind] < cut1) and tst_min: done = 1
                if not done:
                    istep = 1 - 2*int(tst_min)
                    while not done:
                        newmin = newmin + istep
                        if (newmin == -1) | (newmin == nlast): done = 1
                        if not done:
                            if (sky[newmin] <= cut1) and (sky[newmin+1] >= cut1): done = 1
                
                    if tst_min:  delta = sky[newmin+1:minimm+1] - skymid
                    else: delta = sky[minimm+1:newmin+1] - skymid
                    sum = sum - istep*np.sum(delta)
                    sumsq = sumsq - istep*np.sum(delta**2)
                    redo = True
                    minimm = newmin

                newmax = maximm
                if sky[maximm] <= cut2: tst_max = 1           #Is current maximum below upper cut?
                else: tst_max = 0
                if (maximm == nlast) and tst_max: done = 1
                else: done = 0                  #Are we at last pixel of SKY array?
                if not done:
                    if maximm+1 < nlast: skyind = maximm+1
                    else: skyind = nlast
                    if ( tst_max ) and (sky[skyind] > cut2): done = 1 
                if not done: # keep incrementing newmax
                    istep = -1 + 2*int(tst_max)         #Increment up or down?
                    while not done:
                        newmax = newmax + istep
                        if (newmax == nlast) or (newmax == -1): done = 1
                        if not done:
                            if ( sky[newmax] <= cut2 ) and ( sky[newmax+1] >= cut2 ): done = 1

                    if tst_max: 
                        delta = sky[maximm+1:newmax+1] - skymid
                    else:
                        delta = sky[newmax+1:maximm+1] - skymid
                    sum = sum + istep*np.sum(delta)
                    sumsq = sumsq + istep*np.sum(delta**2)
                    redo = True
                    maximm = newmax

                #       
                # Compute mean and sigma (from this pass).
                #
                nsky = maximm - minimm
                if ( nsky < minsky ): # error?
                    sigma = -1.0 ;  skew = 0.0   
                    print('ERROR - Outlier rejection left too few sky elements')
                    return(skymod,sigma,skew)

                skymn = sum/nsky
                var = sumsq/nsky - skymn**2
                if var < 0: var = 0
                sigma = float( np.sqrt( var ))
                skymn = skymn + skymid 

                #  Determine a more robust median by averaging the central 20% of pixels.
                #  Estimate the median using the mean of the central 20 percent of sky
                #  values.   Be careful to include a perfectly symmetric sample of pixels about
                #  the median, whether the total number is even or odd within the acceptance
                #  interval
            
                center = (minimm + 1 + maximm)/2.
                side = np.round(0.2*(maximm-minimm))/2.  + 0.25
                j = np.round(center-side)
                k = np.round(center+side)

                #  In case  the data has a large number of of the same (quantized) 
                #  intensity, expand the range until both limiting values differ from the 
                #  central value by at least 0.25 times the read noise.

                if readnoise:
                    L = round(center-0.25)
                    M = round(center+0.25)
                    R = 0.25*readnoise
                    while ((j > 0) and (k < nsky-1) and \
                            ( ((sky[int(L)] - sky[int(j)]) < R) or ((sky[int(k)] - sky[int(M)]) < R))):
                        j -= 1
                        k += 1

                skymed = np.sum(sky[int(j):int(k+1)])/(k-j+1)

                #  If the mean is less than the median, then the problem of contamination
                #  is slight, and the mean is what we really want.

                if skymed < skymn : 
                    dmod = 3.*skymed-2.*skymn-skymod 
                else: dmod = skymn - skymod

                # prevent oscillations by clamping down if sky adjustments are changing sign
                if dmod*old < 0: clamp = 0.5*clamp
                skymod = skymod + clamp*dmod 
                old = dmod     

            #   if redo then goto, START_LOOP     
            skew = float( (skymn-skymod)/max([1.,sigma]) )
            nsky = maximm - minimm 

            if debug:
                print('% MMM: Number of unrejected sky elements: ', str(nsky,2), \
                        '    Number of iterations: ',  str(niter))
                print('% MMM: Mode, Sigma, Skew of sky vector:', skymod, sigma, skew   )

            return(skymod,sigma,skew)

        BGLevel, BGSig = mmm(sky_vector=PixA_obj)[:2]
        
        """
        # > Alternative Way to use mmm algorithm through photutils
        from astropy.stats import SigmaClip
        from photutils import MMMBackground, StdBackgroundRMS

        sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
        bkg = MMMBackground(sigma_clip=sigma_clip)
        BGLevel = bkg.calc_background(PixA_obj)
        # NOTE: Here calculate the background RMS in an array as the (sigma-clipped) standard deviation
        bkgrms = StdBackgroundRMS(sigma_clip=sigma_clip)
        BGSig = bkgrms.calc_background_rms(PixA_obj)    

        """

        return BGLevel, BGSig
