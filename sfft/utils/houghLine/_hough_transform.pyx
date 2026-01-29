#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp

from libc.math cimport sqrt, ceil

from .interpolation cimport round

cnp.import_array()

def _hough_line(cnp.ndarray img,
                cnp.ndarray[ndim=1, dtype=cnp.double_t] theta):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    img : (M, N) ndarray
        Input image with nonzero values representing edges.
    theta : 1D ndarray of double
        Angles at which to compute the transform, in radians.

    Returns
    -------
    H : 2-D ndarray of uint64
        Hough transform accumulator.
    theta : ndarray
        Angles at which the transform was computed, in radians.
    distances : ndarray
        Distance values.

    Notes
    -----
    The origin is the top left corner of the original image.
    X and Y axis are horizontal and vertical edges respectively.
    The distance is the minimal algebraic distance from the origin
    to the detected line.

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
    # Compute the array of angles and their sine and cosine
    cdef cnp.ndarray[ndim=1, dtype=cnp.double_t] ctheta
    cdef cnp.ndarray[ndim=1, dtype=cnp.double_t] stheta

    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    # compute the bins and allocate the accumulator array
    cdef cnp.ndarray[ndim=2, dtype=cnp.uint64_t] accum
    cdef cnp.ndarray[ndim=1, dtype=cnp.double_t] bins
    cdef Py_ssize_t max_distance, offset

    max_distance = 2 * <Py_ssize_t>ceil(sqrt(img.shape[0] * img.shape[0] +
                                             img.shape[1] * img.shape[1]))
    accum = np.zeros((max_distance, theta.shape[0]), dtype=np.uint64)
    bins = np.linspace(-max_distance / 2.0, max_distance / 2.0, max_distance)
    offset = max_distance / 2

    # compute the nonzero indexes
    cdef cnp.ndarray[ndim=1, dtype=cnp.npy_intp] x_idxs, y_idxs
    y_idxs, x_idxs = np.nonzero(img)

    # finally, run the transform
    cdef Py_ssize_t nidxs, nthetas, i, j, x, y, accum_idx

    nidxs = y_idxs.shape[0]  # x and y are the same shape
    nthetas = theta.shape[0]
    with nogil:
        for i in range(nidxs):
            x = x_idxs[i]
            y = y_idxs[i]
            for j in range(nthetas):
                accum_idx = round((ctheta[j] * x + stheta[j] * y)) + offset
                accum[accum_idx, j] += 1

    return accum, theta, bins
