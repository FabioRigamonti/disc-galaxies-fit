#Standard python imports
# cython: infer_types=True
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from __future__ import division
import  numpy as np
cimport numpy as np
cimport cython
from scipy.special.cython_special cimport i0e, i1e, k0e, k1e
from libc.math cimport M_PI, sin, cos, exp, log, sqrt, acos, atan2

'''this is in Km^2 Kpc / (M_sun s^2).'''
cdef double G = 4.299e-6

'''
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
                r_tmp = r[i,j,k]    #questo va diviso per 2Rd, vedi v_tot
                result_view[i,j,k] = i0e(r_tmp)*k0e(r_tmp)-i1e(r_tmp)*k1e(r_tmp)
    return result
'''
cdef inline double scaled_radius(double x, double y, double r0):
    return sqrt(x*x+y*y)/r0

cdef double Xfunction(double s) nogil:
    # see Hernquist 1990
    cdef double one_minus_s2 = 1.0-s*s
    if s < 1.0:
        return log((1 + sqrt(one_minus_s2)) / s) / sqrt(one_minus_s2)
    elif s ==  1.0:
        return 1.0
    else:
        return acos(1/s)/sqrt(-one_minus_s2)

#cdef class herquist:
#    def __cinit__(self, double M, double Rb, np.ndarray[double, ndim = 3, mode ="c"] x0, np.ndarray[double, ndim = 3, mode ="c"] y0)
#
#    def __call__(self)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim = 3, mode ="c"] hernquist_rho(double M, double Rb, double[:,:,::1] x0, double[:,:,::1] y0):
    #function that return density and
    #isotropic velocity dispersion.
    #calcolo tutti insieme
	
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = x0.shape[0]
    cdef Py_ssize_t J = x0.shape[1]
    cdef Py_ssize_t K = 100
    cdef double s = 0
    cdef double sqrt_one_minus_s2 = 0.0
    cdef double X = 0.0
    cdef np.ndarray[double, ndim = 3, mode ="c"] rho = np.zeros((N,J,K))
    cdef double[:,:,::1] rho_view = rho
    cdef double Rb2 = Rb*Rb
    if (M == 0.) or (Rb == 0.):
        return rho

    for i in range(N):
        for j in range(J):
            for k in range(K):
                s = scaled_radius(x0[i,j,k],y0[i,j,k],Rb)
                if s == 0:
                    s = 1e-8
                X = Xfunction(s)
                if s >= 0.98 and s <=1.02:
                    rho_view[i,j,k] = 4*M / (30*M_PI*Rb2)
                else:
                    rho_view[i,j,k] = (M / (2 * M_PI * Rb2 * (1 - s*s)**2))*((2 + s*s) * X - 3.0)
    return rho

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim = 3, mode ="c"] hernquist_sigma(double M, double Rb, double[:,:,::1] rho, double[:,:,::1] x0, double[:,:,::1] y0):
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t N = x0.shape[0]
    cdef Py_ssize_t J = x0.shape[1]
    cdef Py_ssize_t K = 100
    cdef double s = 0.0
    cdef double s2 = 0.0
    cdef double s3 = 0.0
    cdef double s4 = 0.0
    cdef double s6 = 0.0
    cdef double A,B,C,D
    
    cdef np.ndarray[double, ndim = 3, mode ="c"] sigma = np.zeros((N,J,K))
    cdef double[:,:,::1] sigma_view = sigma
    
    for i in range(N):
        for j in range(J):
            for k in range(K):
                s = scaled_radius(x0[i,j,k],y0[i,j,k],Rb)
                s2 = s*s
                s3 = s2*s
                s4 = s2*s2
                s6 = s3*s3
                if s > 6.0:
                    # s > 6
                    A = (G * M * 8) / (15 * s * M_PI * Rb)
                    B = (8/M_PI - 75*M_PI/64) / s
                    C = (64/(M_PI*M_PI) - 297/56) / s2
                    D = (512/(M_PI*M_PI*M_PI) - 1199/(21*M_PI) - 75*M_PI/512) / s3
                    sigma_view[i,j,k] = A * (1 + B + C + D)
                else:
                    if s >= 0.98 and s <=1.02:
                        sigma_view[i,j,k] = G*M*(332 - 105*M_PI)/28*Rb
                    else:
                        A = (G * M**2 ) / (12 * M_PI * Rb**3 * rho[i,j,k])
                        B = 1 /  (2*(1 - s2)**3)
                        C = (-3 * s2) * Xfunction(s) * (8*s6 - 28*s4 + 35*s2 - 20) - 24*s6 + 68*s4 - 65*s2 + 6
                        sigma_view[i,j,k] = A * (B*C - 6*M_PI*s)
                        
    return sigma

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double v_D(double Md,double Rd,double r) nogil:
    #circular v^2 for exp disc
    cdef double v_c2 = 0.
    if Md == 0 :
        return v_c2
    cdef double r_tmp = 0.
    r_tmp = r/(2*Rd)
    v_c2 =  ((G * Md * r) / (Rd*Rd)) * r_tmp * (i0e(r_tmp)*k0e(r_tmp) - i1e(r_tmp)*k1e(r_tmp))
    
    return v_c2 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double v_H(double Mb,double Rb,double r) nogil:
    #circular v^2 for herquinst (actually bulge and halo)
    cdef double v_c2 = 0.
    if Mb == 0.:
        return v_c2
    cdef double r_tmp = 0.
    r_tmp = r/Rb
    v_c2 = G * Mb * r_tmp / (Rb*(1+r_tmp)*(1+r_tmp) )
    
    return v_c2

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray[double, ndim = 3, mode ="c"] v_tot(double Mb,
                                                    double Rb,
                                                    double Md,
                                                    double Rd,
                                                    double Mh,
                                                    double Rh,
                                                    double incl,
                                                    double[:,:,::1] x,
                                                    double[:,:,::1] y):
    cdef Py_ssize_t i,j,k 
    cdef Py_ssize_t N = x.shape[0]
    cdef Py_ssize_t J = y.shape[1]
    cdef Py_ssize_t K = 100
    cdef double r_tmp = 0.
    cdef double x_tmp = 0.
    cdef double y_tmp = 0.
    cdef double phi = 0.
    cdef double sin_i = sin(incl)
    cdef np.ndarray[double, ndim = 3, mode ="c"] v_tot = np.zeros((N,J,K))
    cdef double[:,:,::1] v_tot_view = v_tot 

    for i in range(N):
        for j in range(J):
            for k in range(K):
                x_tmp = x[i,j,k]
                y_tmp = y[i,j,k]
                r_tmp = scaled_radius(x_tmp,y_tmp,1.)
                phi = atan2(y_tmp,x_tmp)
                v_tot_view[i,j,k] = sin_i * cos(phi) * sqrt(v_D(Md,Rd,r_tmp) + v_H(Mb,Rb,r_tmp) + v_H(Mh,Rh,r_tmp)) 
    
    return v_tot 