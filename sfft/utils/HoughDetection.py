import bisect
import numpy as np
from skimage import feature
from skimage.transform import hough_line, hough_line_peaks
# version: Aug 21, 2022

__author__ = "Lei Hu <hulei@pmo.ac.cn>"
__version__ = "v1.3"

"""
# MeLon Notes 
# @Hough Transformation
#
# a) A stright line in ImageSpace could be geometrically charaterized by a certain pair (theta, rho), 
#     that is, a certain point in HoughSpace. where rho means the distance between the origin and the straight line 
#     (positive | negative for distance-array Y-Upward | Y-Downward, or X-Rightward | X-Leftward for special vertical line case)
#     while theta is the angle between the distance-stright-line and X+ axis, so it can be also negative. 
#     thus it's easy to prove the stright line could be represented by rho = x*sin(theta) + y*cos(theta)
#     WARNING: Sometimes theta is mathematically defined respect to X- axis, however, we follow X+ rule.
#
# b) Consider one point A in ImageSpace, imagine all possible stright lines cross the point, there's a corresponded 
#     continuous curve U in Hough Space under such transformation !
#
# c) Assume Two points A B in ImageSpace and corresponding Curves U V in Hough Space, the Hough intersection point of U & V is K. 
#     It's easy to verify that the stright line L determied by points A and B will natually link the hough point K.
#
# d) Imagine more points occur on L, says C, its hough curve will continue to contribute to the intersected point K. 
#      >>> Proof >>> Consider A C in ImageSpace and their hough curves U W, the Hough intersection point of U & W is K'
#      the stright line L'=L is determied by points A and C, thus L' links the hough point K'=K.
#      Thus a stright line feature would correspond to a hough frequent-intersection point. 
#
# e) Sampling the HoughSpace, count the such intersection times as flux for each pixel.
#    Then we could detect the stright lines from the peaks in hough space.
#    More precisely, consider hough peak K = (theta_k, rho_k), then the stright line L is rho_k = x*sin(theta_k) + y*cos(theta_k)
#
# f) Note that the obvious disadvantage of Hough-Transformation is the unknow lenth of the stright line.
#    Probability-Hough-Transformation can handle such case but seems more time-consuming.
#
#    NOTE ImageSpace (x, y) and HoughSpace (theta, rho), where theta in (-pi/2, pi/2].
#    NOTE skimage provides hough_transformation function with Mask as input
#    For scatter x-y points as input, we may just make sampling to covert to be 2D-image
#    Creat Mask with count-threshold or canny edge detection as common preliminary method for Hough Detection.
#
# g) Note that the Hough-Transformation is performed on pixel array. A line feature characterized by 
#    (theta, rho) uses the left-lower point of the left-lower pixel as the coordinate origin!
#    That is to say, we don't need any further coordinate conversion when input data are scatter points.

"""

class Hough_Detection:
    @staticmethod
    def HD(XY_obj=None, PixA_obj=None, Hmask=None, grid_pixsize=None, count_thresh=None, canny_sig=None, peak_clip=0.7):
        
        # * Convert non-masked scatter points to be 2D-image
        #   NOTE: It's useful to restrict on scatter points for finding the optimal stright-line featrues.
        #         That is, only partial scatters will be pixelized for detection, which is less time-consuming.

        if XY_obj is not None:
            XY_Hobj = XY_obj.copy()
            if Hmask is not None: XY_Hobj = XY_obj[Hmask]
            
            xmin, xmax = np.min(XY_Hobj[:, 0]), np.max(XY_Hobj[:, 0])
            xnodes = np.arange(xmin, xmax+2*grid_pixsize, grid_pixsize)
            ymin, ymax = np.min(XY_Hobj[:, 1]), np.max(XY_Hobj[:, 1])
            ynodes = np.arange(ymin, ymax+2*grid_pixsize, grid_pixsize)
            PixA_inp = np.zeros((len(xnodes)-1, len(ynodes)-1))

            for x, y in XY_Hobj:
                r = bisect.bisect_right(xnodes, x)-1
                c = bisect.bisect_right(ynodes, y)-1
                PixA_inp[r, c] += 1
        else:
            assert PixA_obj is not None
            PixA_inp = PixA_obj
        
        # * Make Mask from 2D-image with Count-Threshold or canny egde detection
        assert (count_thresh is not None) or (canny_sig is not None)
        if count_thresh is not None: Mask_inp = PixA_inp >= count_thresh
        if canny_sig is not None: Mask_inp = feature.canny(PixA_inp, sigma=canny_sig)
        
        # * Hough Transformation and Peak searching
        Hspace, Theta, Rho = hough_line(Mask_inp.astype(int))
        ThetaPeaks, RhoPeaks = hough_line_peaks(Hspace, Theta, Rho, threshold=peak_clip*np.max(Hspace))[1:]

        # * Calculate distance from the scatter points to the detected lines.
        #   d = |A*x_0 + B*y_0 + C| / (A**2 + B**2)**(1/2) that is, d = |sin(theta)*x_0 + cos(theta)*y_0 - rho|
        
        ScaLineDIST = None
        if XY_obj is not None:
            ScaLineDIST = []
            for i in range(len(RhoPeaks)):
                RhoPeaks[i] = grid_pixsize*RhoPeaks[i] + xmin*np.sin(ThetaPeaks[i]) + ymin*np.cos(ThetaPeaks[i])
                SLd = np.abs(np.sin(ThetaPeaks[i])*XY_obj[:, 0] + np.cos(ThetaPeaks[i])*XY_obj[:, 1] - RhoPeaks[i])
                ScaLineDIST.append(SLd)
            ScaLineDIST = np.array(ScaLineDIST).T

        return PixA_inp, Hspace, ThetaPeaks, RhoPeaks, ScaLineDIST
