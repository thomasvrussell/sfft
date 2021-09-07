import numpy as np

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.0"

# MeLOn Notes
# @ Calculate Covariance of Neighboring Pixels
# Consider 2 random variables
# a: pixel value at coordinate (r, c)
# b: pixel value at coordinate (r+1, c+1)
# For given astronomical image, it is possible to do sampling from the joint probability distribution of a & b.
# Each sample is a pair of pixels cross through the image, note we should pick such pairs on background 
# if we only concern the Covariance of neighboring background pixels.
# * One of the most efficient ways to get massive samples is rolling the image.
#    Samples of (a, b): {image.flatten(), np.roll(image, 1, 1).flatten()}
#    NOTE we denote a as (0, 0) and b (1, 1), here we concerns 25 random variables {(0, 0), (1, 0), (-1, 0) ...}
#    NOTE image has physical boundary thus some samples above should be rejected.
#               For a random variable denoted as (p, q)
#               tmp = np.roll(image, p, q)
#               tmp[:p, :] (p > 0) & tmp[p:, :] (p < 0)  are invalid 
#               tmp[:, :q] (q > 0) & tmp[:, q:] (q < 0)  are invalid

class NeighboringPixel_Covariance:
    @staticmethod
    def NPC(PixA_obj):

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
        SampleSet = np.array([shiftedim[~RMask].flatten() for shiftedim in shiftedim_lst])  # Shape (Ndim, Nsamp)
        #print('MeLOn CheckPoint: SampleSet has shape (Ndim, Nsamp) = (%d, %d)' \
        #    %(SampleSet.shape[0], SampleSet.shape[1]))

        CovMatrix = np.cov(SampleSet, bias=True)   #TODO: Why do we use bias ?
        tmp_ = CovMatrix.copy()
        np.fill_diagonal(tmp_, np.NaN)
        tmp_ = np.abs(tmp_)
        CovLevel = (np.nansum(tmp_) / np.sum(np.diag(CovMatrix)))
        return CovMatrix, CovLevel

