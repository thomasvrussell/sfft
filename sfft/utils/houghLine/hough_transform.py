import numpy as np
from sfft.utils.houghLine._hough_transform import _hough_line

# NOTE: This is a copy from skimage.transform v1.8.3
# see https://github.com/scikit-image/scikit-image

def hough_line_peaks(hspace, angles, dists, min_distance=9, min_angle=10,
                     threshold=None, num_peaks=np.inf):
    """Return peaks in a straight line Hough transform.

    Identifies most prominent lines separated by a certain angle and distance
    in a Hough transform. Non-maximum suppression with different sizes is
    applied separately in the first (distances) and second (angles) dimension
    of the Hough space to identify peaks.

    Parameters
    ----------
    hspace : (N, M) array
        Hough space returned by the `hough_line` function.
    angles : (M,) array
        Angles returned by the `hough_line` function. Assumed to be continuous.
        (`angles[-1] - angles[0] == PI`).
    dists : (N, ) array
        Distances returned by the `hough_line` function.
    min_distance : int, optional
        Minimum distance separating lines (maximum filter size for first
        dimension of hough space).
    min_angle : int, optional
        Minimum angle separating lines (maximum filter size for second
        dimension of hough space).
    threshold : float, optional
        Minimum intensity of peaks. Default is `0.5 * max(hspace)`.
    num_peaks : int, optional
        Maximum number of peaks. When the number of peaks exceeds `num_peaks`,
        return `num_peaks` coordinates based on peak intensity.

    Returns
    -------
    accum, angles, dists : tuple of array
        Peak values in Hough space, angles and distances.

    Examples
    --------
    >>> from skimage.transform import hough_line, hough_line_peaks
    >>> from skimage.draw import line
    >>> img = np.zeros((15, 15), dtype=bool)
    >>> rr, cc = line(0, 0, 14, 14)
    >>> img[rr, cc] = 1
    >>> rr, cc = line(0, 14, 14, 0)
    >>> img[cc, rr] = 1
    >>> hspace, angles, dists = hough_line(img)
    >>> hspace, angles, dists = hough_line_peaks(hspace, angles, dists)
    >>> len(angles)
    2

    """
    from .peak import _prominent_peaks
    
    min_angle = min(min_angle, hspace.shape[1])
    h, a, d = _prominent_peaks(hspace, min_xdistance=min_angle,
                               min_ydistance=min_distance,
                               threshold=threshold,
                               num_peaks=num_peaks)
    if a.any():
        return (h, angles[a], dists[d])
    else:
        return (h, np.array([]), np.array([]))

def hough_line(image, theta=None):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    image : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of double, optional
        Angles at which to compute the transform, in radians.
        Defaults to a vector of 180 angles evenly spaced from -pi/2 to pi/2.

    Returns
    -------
    hspace : 2-D ndarray of uint64
        Hough transform accumulator.
    angles : ndarray
        Angles at which the transform is computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.
    The angle accuracy can be improved by decreasing the step size in
    the `theta` array.

    Examples
    --------
    Generate a test image:

    >>> img = np.zeros((100, 150), dtype=bool)
    >>> img[30, :] = 1
    >>> img[:, 65] = 1
    >>> img[35:45, 35:50] = 1
    >>> for i in range(90):
    ...     img[i, i] = 1
    >>> img += np.random.random(img.shape) > 0.95

    Apply the Hough transform:

    >>> out, angles, d = hough_line(img)

    .. plot:: hough_tf.py

    """
    if image.ndim != 2:
        raise ValueError('The input image `image` must be 2D.')

    if theta is None:
        # These values are approximations of pi/2
        theta = np.linspace(-np.pi / 2, np.pi / 2, 180)

    return _hough_line(image, theta=theta)

