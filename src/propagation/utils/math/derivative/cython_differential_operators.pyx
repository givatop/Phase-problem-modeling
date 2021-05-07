cimport numpy as np
cimport cython
import numpy as np

"""
Численный расчет частных производных методом конечных сумм
"""

#2D gradient of a real function
@cython.boundscheck(False) # turn of bounds-checking for entire function
def gradient_2d(np.ndarray[double, ndim=2] f, double dx, double dy):
    cdef unsigned int i, j, ni, nj
    cdef double ifactor, jfactor
    ni = f.shape[0]
    nj = f.shape[1]
    cdef np.ndarray[double, ndim=2] grad = np.zeros((ni,nj))

    ifactor = 1/(2*dx)
    jfactor = 1/(2*dy)

    # Аппроксимация по среднему
    for i in xrange(1,ni-1):
        for j in xrange(1, nj-1):
            grad[i, j] = (f[i+1, j] - f[i-1, j])*ifactor + (f[i, j+1] - f[i, j-1])*jfactor

    return grad


#2D inverse Laplacian of a real function
@cython.boundscheck(False) # turn of bounds-checking for entire function
def ilaplacian_2d(np.ndarray[double, ndim=2] f, double dx, double dy):
    cdef unsigned int i, j, ni, nj
    cdef double ifactor, jfactor, ijfactor
    ni = f.shape[0]
    nj = f.shape[1]
    cdef np.ndarray[double, ndim=2] lapf = np.zeros((ni,nj))

    ifactor = 1/dx**2
    jfactor = 1/dy**2
    ijfactor = 2.0 * (ifactor + jfactor)

    for i in xrange(1,ni-1):
        for j in xrange(1, nj-1):
            lapf[i, j] = (f[i, j-1] + f[i, j+1]) * jfactor + (f[i-1, j] + f[i+1, j]) * ifactor - f[i,j] * ijfactor
            lapf[i, j] = 1 / lapf[i, j]  # make Laplacian inverse
            """
            if lapf[i, j] == 0.0:
                pass # lapf[i, j] = 1e-2
            else:
                lapf[i, j] = 1 / lapf[i, j]  # make Laplacian inverse
            """

    return lapf


#3D laplacian of a complex function
@cython.boundscheck(False) # turn of bounds-checking for entire function
def laplacianFD3dcomplex(np.ndarray[double complex, ndim=3] f, double complex dx, double complex dy, double complex dz):
    cdef unsigned int i, j, k, ni, nj, nk
    cdef double complex ifactor, jfactor, kfactor, ijkfactor
    ni = f.shape[0]
    nj = f.shape[1]
    nk = f.shape[2]
    cdef np.ndarray[double complex, ndim=3] lapf = np.zeros((ni,nj,nk)) +0.0J

    ifactor = 1/dx**2
    jfactor = 1/dy**2
    kfactor = 1/dz**2
    ijkfactor = 2.0*(ifactor + jfactor + kfactor)

    for i in xrange(1,ni-1):
        for j in xrange(1, nj-1):
            for k in xrange(1, nk-1):
                lapf[i, j, k] = (f[i, j, k-1] + f[i, j, k+1])*kfactor + (f[i, j-1, k] + f[i, j+1, k])*jfactor + (f[i-1, j, k] + f[i+1, j, k])*ifactor - f[i,j,k]*ijkfactor
    return lapf


#3D laplacian of a real function
@cython.boundscheck(False) # turn of bounds-checking for entire function
def laplacianFD3dreal(np.ndarray[double, ndim=3] f, double dx, double dy, double dz):
    cdef unsigned int i, j, k, ni, nj, nk
    cdef double ifactor, jfactor, kfactor, ijkfactor
    ni = f.shape[0]
    nj = f.shape[1]
    nk = f.shape[2]
    cdef np.ndarray[double, ndim=3] lapf = np.zeros((ni,nj,nk))

    ifactor = 1/dx**2
    jfactor = 1/dy**2
    kfactor = 1/dz**2
    ijkfactor = 2.0*(ifactor + jfactor + kfactor)

    for i in xrange(1,ni-1):
        for j in xrange(1, nj-1):
            for k in xrange(1, nk-1):
                lapf[i, j, k] = (f[i, j, k-1] + f[i, j, k+1])*kfactor + (f[i, j-1, k] + f[i, j+1, k])*jfactor + (f[i-1, j, k] + f[i+1, j, k])*ifactor - f[i,j,k]*ijkfactor
    return lapf