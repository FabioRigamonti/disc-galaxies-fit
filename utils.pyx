#Standard python imports
# cython: infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from __future__ import division
import  numpy as np
cimport numpy as np
cimport cython
from scipy.special.cython_special cimport i0e, i1e, k0e, k1e
from libc.math cimport M_PI, sin, cos, exp, log

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim = 3, mode ="c"] differenza(double[:,:,::1] r):
    #DIFFERENZA tra le bessel function,
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = r.shape[0]
    cdef Py_ssize_t J = r.shape[1]
    cdef Py_ssize_t K = 100
    cdef np.ndarray[double, ndim = 3, mode ="c"] result = np.zeros((N,J,K))
    cdef double[:,:,::1] result_view = result
    cdef double r_tmp = 0.0
    for i in range(N):
        for j in range(J):
            for k  in range(K):
                r_tmp = r[i,j,k]
                result_view[i,j,k] = i0e(r_tmp)*k0e(r_tmp)-i1e(r_tmp)*k1e(r_tmp)
    return result
