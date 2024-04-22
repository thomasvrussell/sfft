import bisect
import skimage
import warnings
import numpy as np
from skimage import feature
from pkg_resources import parse_version
from skimage.transform import hough_line, hough_line_peaks
# version: Feb 4, 2023

__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.4"

class Hough_Detection:
    @staticmethod
    def HD(XY_obj=None, PixA_obj=None, Hmask=None, grid_pixsize=None, \
        count_thresh=None, canny_sig=None, peak_clip=0.7, VERBOSE_LEVEL=2):
        
        """
        # MeLon Notes 
        # What is Hough Transform?
        # a) A stright line in original-space can be geometrically charaterized by a certain pair (theta, rho) in hough-space. 
        #    The object stright line could be represented by x*cos(theta) + y*sin(theta) = rho
        #    The projection from a straight line in original-space to a point in hough-space is a bijection.
        #    
        #    [-] theta [-90d, +90d) is the rotate angle from X+ to the connecting line.
        #        > connecting line: line crossing the origin and the closest point on the straight line.
        #        > there are two directions of the connection line to measure rotate angle, we always 
        #          use the one yielding smaller angle.
        #        > the sign of theta: rotation towards Y+ (Y-) gets postive (negative) angles. 
        #          special case: if the connection line is orthogonal to X+, we adopt rotate angle of -90d.
        #
        #    [-] rho (-inf, +inf) is the length of connecting arrow (however, negative values allowed).
        #        > connecting arrow: arrow from the origin to the closest point on the straight line.
        #        > the sign of rho: arrow towards X+ (X-) get positive (negative) lengths.
        #          special case (theta = -90d): arrow towards Y- (Y+) get positive (negative) lengths.
        #
        #    [*] we can get the distance between two parallel straight lines using hough representations
        #        says, the hough representations are given as below, then their distance is |rho2 - rho1| geometrically.
        #        (1) x*cos(theta) + y*sin(theta) = rho1 | (2) x*cos(theta) + y*sin(theta) = rho2
        #        the corollary is that the distance of any point (x_t, y_t) to a given straight line (1) is
        #        |x_t*cos(theta) + y_t*sin(theta) - rho1|
        #
        #    Remarks on symmetric definition
        #    > There is a symmetric definition of (theta, rho), for which the equation becomes x*sin(theta) + y*cos(theta) = rho.
        #      For paragraphs [-]x2, only we need to change is replacing X+/X-/Y+/Y- by Y+/Y-/X+/X-.
        #      For paragraph [*], just swap the functions cos and sin.
        #      NOTE: Since scikit-image adopts this convention, we will stick with this kind of definition in our context!
        #
        #    Remarks on disadvantage of hough transform
        #    > one obvious disadvantage of hough transform is we don't know the start-end point of the detected stright line.
        #      probability hough transform can address this issue but more time-consuming.
        #
        # How does Hough Transform work?
        # b1) Consider a point A in orinal-space, all possible stright lines cross the point A will produce 
        #     a continuous curve U in hough-space, accordingly. 
        #  
        # b2) Assume two points A and B in orinal-space that can produce curves H(A) and H(B) in hough-space, respectively.
        #     Let K be the crosspoint of H(A) and H(B), it's easy to verify that K is the hough projection of 
        #     the stright line L determied by points A and B.
        #
        # b3) Imagine an additional point C appears on stright line L, its hough curve has to pass through K.
        #     As a result, more points on stright line L will accumulatively contribute to the crosspoint K. 
        #     P.S. Let K' be the crosspoint of H(A) and H(C), then K' is also the hough projection of the stright line L' 
        #          determined by A and C. Since L' = L then K' = K.
        #
        # c) Says, we have a set of scatter points in original-space, each point can produce a curve in hough-space.
        #    A 'high-density' region in hough-space would indicate that a corrsponding striaight feature exist in original-space.
        #
        # d) In practice, both original-space and hough-space are pixelized. One can detect stright line features 
        #    in original-space by the peaks in hough space. using count-threshold or canny edge detection are 
        #    popular preliminary processing to create binary image for hough detection.
        #    
        #    Note that the coordinate origin of a pixeled image is 
        #    (1) lying view: the lower-left corner of the lower-left (first) pixel. 
        #    (2) standing view: the top-left corner of the top-left (first) pixel. 
        # 
        # More details about scikit-image hough transform
        # e) scikit-image updates their implementation of hough transform since v0.19.0
        #    I tested hough transform to detect a thin stripe with random width 1.0-4.0 pix on 1024 X 1024 image.
        #    each row of the following listed as: input theta, input rho, detected-input rho, detected-input theta
        #
        #    > old versions 0.16.1 - 0.18.3: 
        #      [87.5d, 89.5d), [5, 1000], 0.28 +/- 2.87, 0.00 +/- 0.38        | near-vertical-1
        #      [-89.5d, -87.5d], [5, 1000], -1.25 +/- 5.03, -0.07 +/- 0.42    | near-vertical-2
        #      [-89.5d, -87.5d], [-1000, -5], 0.83 +/- 3.00, -0.00 +/- 0.36   | near-vertical-3
        #
        #      [45d, 55d], [5, 1000], 0.04 +/- 0.55, -0.00 +/- 0.36           | normal-1
        #      [-55d, -45d], [5, 1000], 1.32 +/- 5.18, 0.06 +/- 0.40          | normal-2
        #      [-55d, -45d], [-1000, -5], 0.93 +/- 4.64, 0.03 +/- 0.38        | normal-3
        #
        #      [-2.5d, -0.5d], [5, 1000], 0.19 +/- 3.08, -0.00 +/- 0.37       | near-horizental-1
        #      [-0.5d, 0.5d], [5, 1000], 0.54 +/- 2.99, 0.03 +/- 0.38         | near-horizental-2
        #      [0.5d, 2.5d], [5, 1000], -0.07 +/- 2.90, -0.03 +/- 0.37        | near-horizental-3
        #
        #    > new versions 0.19.0 - 0.19.3 (latest)
        #      [87.5d, 89.5d), [5, 1000], -0.51 +/- 2.85, -0.00 +/- 0.37      | near-vertical-1
        #      [-89.5d, -87.5d], [5, 1000], -0.16 +/- 4.98, 0.03 +/- 0.36     | near-vertical-2
        #      [-89.5d, -87.5d], [-1000, -5], 0.31 +/- 2.87, -0.02 +/- 0.34   | near-vertical-3
        #
        #      [45d, 55d], [5, 1000], -0.64 +/- 0.58, 0.00 +/- 0.37           | normal-1
        #      [-55d, -45d], [5, 1000], 0.48 +/- 4.87, 0.05 +/- 0.41          | normal-2
        #      [-55d, -45d], [-1000, -5], 0.18 +/- 4.78, 0.01 +/- 0.40        | normal-3
        #
        #      [-2.5d, -0.5d], [5, 1000], -0.69 +/- 2.98, -0.02 +/- 0.36      | near-horizental-1
        #      [-0.5d, 0.5d], [5, 1000], -0.55 +/- 3.02, -0.00 +/- 0.38       | near-horizental-2  
        #      [0.5d, 2.5d], [5, 1000], -0.54 +/- 2.84, -0.00 +/- 0.37        | near-horizental-3 
        #
        """
        
        # * check version
        skimage_version = parse_version(skimage.__version__)
        if skimage_version < parse_version('0.16.1'):
            if VERBOSE_LEVEL in [0, 1, 2]:
                _warn_message = 'current scikit-image [%s] is too old and not tested!' %skimage_version
                warnings.warn('MeLOn WARNING: %s' %_warn_message)
        
        if parse_version('0.16.1') <= skimage_version <= parse_version('0.18.3'):
            if VERBOSE_LEVEL in [2]:
                _message = 'current scikit-image [%s] uses classic implementation of hough transform' %skimage_version
                print('MeLOn CheckPoint: %s' %_message)
        
        if skimage_version >= parse_version('0.19.0'):
            if VERBOSE_LEVEL in [2]:
                _message = 'current scikit-image [%s] uses updated implementation of hough transform' %skimage_version
                print('MeLOn CheckPoint: %s' %_message)
        
        # * convert scatter points to be 2D-image
        if XY_obj is not None:
            """
            # Remarks on the pixelization
            # [-] the coordinate origin of the image is at (x_min, y_min), and the image can fully cover all scatter points.
            #     we therefore define the complete nodes (containing all boundaries) of the image grid as 
            #     arange(x_min, x_max + 2Gx, Gx), where Gx is grid pixel size along X axis
            #     arange(y_min, y_max + 2Gy, Gy), where Gx is grid pixel size along Y axis
            #
            # [-] when the scatter points contribute the pixels of the image grid, we will regard the left boundary 
            #     and lower boundary of a pixel as solid lines but leave others dashed lines (lying view).
            #  
            # NOTE: to avoid too large pixeled image, it is useful to reject the outliers of input scatter points.
            #       this function allows users to provide a mask for the rejection.
            #
            """
            XY_Hobj = XY_obj.copy()
            if Hmask is not None: XY_Hobj = XY_obj[Hmask]
            x_min, x_max = np.min(XY_Hobj[:, 0]), np.max(XY_Hobj[:, 0])
            xnodes = np.arange(x_min, x_max+2*grid_pixsize, grid_pixsize)
            y_min, y_max = np.min(XY_Hobj[:, 1]), np.max(XY_Hobj[:, 1])
            ynodes = np.arange(y_min, y_max+2*grid_pixsize, grid_pixsize)
            PixA_inp = np.zeros((len(xnodes)-1, len(ynodes)-1))
            for x, y in XY_Hobj:
                r = bisect.bisect_right(xnodes, x)-1
                c = bisect.bisect_right(ynodes, y)-1
                PixA_inp[r, c] += 1
        else:
            assert PixA_obj is not None
            PixA_inp = PixA_obj
        
        # * create mask from 2D-image by count-threshold or canny egde detection
        assert (count_thresh is not None) or (canny_sig is not None)
        if count_thresh is not None: Mask_inp = PixA_inp >= count_thresh
        if canny_sig is not None: Mask_inp = feature.canny(PixA_inp, sigma=canny_sig)
        
        # * perform hough transform and searching peak in hough space
        Hspace, Theta, Rho = hough_line(Mask_inp.astype(int))
        ThetaPeaks, RhoPeaks = hough_line_peaks(Hspace, Theta, Rho, threshold=peak_clip*np.max(Hspace))[1:]
        
        # * switch to the raw coordinate origin and length unit
        # * calculate the scalar distance from scatter points to the detected lines

        ScaLineDIST = None
        if XY_obj is not None:
            ScaLineDIST = []
            for i in range(len(RhoPeaks)):
                RhoPeaks[i] = grid_pixsize*RhoPeaks[i] + x_min*np.sin(ThetaPeaks[i]) + y_min*np.cos(ThetaPeaks[i])
                dist = np.abs(np.sin(ThetaPeaks[i])*XY_obj[:, 0] + np.cos(ThetaPeaks[i])*XY_obj[:, 1] - RhoPeaks[i])
                ScaLineDIST.append(dist)
            ScaLineDIST = np.array(ScaLineDIST).T

        return PixA_inp, Hspace, ThetaPeaks, RhoPeaks, ScaLineDIST
