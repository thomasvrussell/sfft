"""Utility functions used in the morphology subpackage."""


import numpy as np
from scipy import ndimage as ndi

def _resolve_neighborhood(selem, connectivity, ndim):
    """Validate or create structuring element.

    Depending on the values of `connectivity` and `selem` this function
    either creates a new structuring element (`selem` is None) using
    `connectivity` or validates the given structuring element (`selem` is not
    None).

    Parameters
    ----------
    selem : ndarray
        A structuring element used to determine the neighborhood of each
        evaluated pixel (``True`` denotes a connected pixel). It must be a
        boolean array and have the same number of dimensions as `image`. If
        neither `selem` nor `connectivity` are given, all adjacent pixels are
        considered as part of the neighborhood.
    connectivity : int
        A number used to determine the neighborhood of each evaluated pixel.
        Adjacent pixels whose squared distance from the center is less than or
        equal to `connectivity` are considered neighbors. Ignored if
        `selem` is not None.
    ndim : int
        Number of dimensions `selem` ought to have.

    Returns
    -------
    selem : ndarray
        Validated or new structuring element specifying the neighborhood.

    Examples
    --------
    >>> _resolve_neighborhood(None, 1, 2)
    array([[False,  True, False],
           [ True,  True,  True],
           [False,  True, False]])
    >>> _resolve_neighborhood(None, None, 3).shape
    (3, 3, 3)
    """
    if selem is None:
        if connectivity is None:
            connectivity = ndim
        selem = ndi.generate_binary_structure(ndim, connectivity)
    else:
        # Validate custom structured element
        selem = np.asarray(selem, dtype=bool)
        # Must specify neighbors for all dimensions
        if selem.ndim != ndim:
            raise ValueError(
                "number of dimensions in image and structuring element do not"
                "match"
            )
        # Must only specify direct neighbors
        if any(s != 3 for s in selem.shape):
            raise ValueError("dimension size in structuring element is not 3")

    return selem
