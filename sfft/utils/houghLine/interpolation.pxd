#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
"""
Note: All edge modes implemented here follow the corresponding numpy.pad
conventions.

The table below illustrates the behavior for the array [1, 2, 3, 4], if padded
by 4 values on each side:

                               pad     original    pad
    constant (with c=0) :    0 0 0 0 | 1 2 3 4 | 0 0 0 0
    wrap                :    1 2 3 4 | 1 2 3 4 | 1 2 3 4
    symmetric           :    4 3 2 1 | 1 2 3 4 | 4 3 2 1
    edge                :    1 1 1 1 | 1 2 3 4 | 4 4 4 4
    reflect             :    3 4 3 2 | 1 2 3 4 | 3 2 1 2
"""
from libc.math cimport ceil, floor

import numpy as np
cimport numpy as np
from .fused_numerics cimport np_floats

cdef inline Py_ssize_t round(np_floats r) nogil:
    return <Py_ssize_t>(
        (r + <np_floats>0.5) if (r > <np_floats>0.0) else (r - <np_floats>0.5)
    )
