import numpy as np
# version: Feb 5, 2023

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.4"

class NeighboringPixel_Covariance:
    @staticmethod
    def NPC(PixA_obj):
        
        """
        # MeLOn Notes
        # @ Convariance Calculator of Neighboring Pixels
        # (I) Consider two random vairbales: a is the pixel value at coordinate (r, c) and 
        #     b is the pixel value at coordinate (r+1, c+1). 
        #
        # (II) For a given astronomical image, exhuasting all possible r, c would produce a large 
        #      set of pairs (a, b), as sampled from the joint probability distribution of a & b.
        #      NOTE: In general, we concern the background covariance so only background pixels used.
        #       
        # (III) Rolling the image can easily produce the large sample set.
        #       Samples of (a, b): {image.flatten(), np.roll(image, 1, 1).flatten()}
        #       NOTE: we can replace (+1, +1) with other neighboring position (p, q) 
        #             here we concern 25 relative positions {(0, 0), (+1, 0), (-1, 0) ...}
        #       NOTE: please reject the rolling samples out of image boundary.
        #             tmp = np.roll(image, p, q)
        #             tmp[:p, :] (p > 0) & tmp[p:, :] (p < 0)  are invalid 
        #             tmp[:, :q] (q > 0) & tmp[:, q:] (q < 0)  are invalid
        #
        """
        
        RVs = ([0, 0], \
               [1, 0], [-1, 0], [0, 1], [0, -1], \
               [1, 1], [1, -1], [-1, 1], [-1, -1], \
               [2, 0], [-2, 0], [0, 2], [0, -2], \
               [3, 0], [-3, 0], [0, 3], [0, -3], \
               [4, 0], [-4, 0], [0, 4], [0, -4], \
               [5, 0], [-5, 0], [0, 5], [0, -5])

        im = PixA_obj / PixA_obj.std()
        shiftedim_lst, rejmask_lst = [], []
        for rv in RVs:
            p, q = rv
            shiftedim = np.roll(np.roll(im, p, axis=0), q, axis=1)
            rejmask = np.zeros(im.shape).astype(bool)
            if p > 0: rejmask[:p, :] = True
            if p < 0: rejmask[p:, :] = True
            if q > 0: rejmask[:, :q] = True
            if q < 0: rejmask[:, q:] = True
            shiftedim_lst.append(shiftedim)
            rejmask_lst.append(rejmask)
        
        RMask = np.logical_or.reduce(tuple(rejmask_lst))
        SampleSet = np.array([shiftedim[~RMask].flatten() for shiftedim in shiftedim_lst])  # (Ndim, Nsamp)
        CovMatrix = np.cov(SampleSet, bias=True)

        tmp_ = CovMatrix.copy()
        np.fill_diagonal(tmp_, np.NaN)
        tmp_ = np.abs(tmp_)
        CovLevel = (np.nansum(tmp_) / np.sum(np.diag(CovMatrix)))
        
        return CovMatrix, CovLevel
