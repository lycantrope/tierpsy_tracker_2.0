# spline_cython.pyx

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

cdef class SplineSheet:
    cdef public np.ndarray controlPoints
    cdef public tuple M, halfSupport

    def __init__(self, tuple M):
        self.M = M
        self.controlPoints = np.zeros(((M[1] + 4) * (M[0] + 4), 3), dtype=np.float64)
        self.halfSupport = (2.0, 2.0)

    cpdef initializeFromPoints(self, np.ndarray samplePoints, np.ndarray sampleParameters):
        cdef int N = len(samplePoints)
        cdef int M0 = self.M[0]
        cdef int M1 = self.M[1]
        cdef int support0 = 4
        cdef int support1 = 4

        phi = np.zeros((N, (M1 + support1) * (M0 + support0)), dtype=np.float64)
        sigmaX = np.zeros(N, dtype=np.float64)
        sigmaY = np.zeros(N, dtype=np.float64)
        sigmaZ = np.zeros(N, dtype=np.float64)

        cdef int n, l, k, kp
        cdef double t, s, tVal, sVal, basisFactor1, basisFactor2
        for n in range(N):
            sigmaX[n] = samplePoints[n, 0]
            sigmaY[n] = samplePoints[n, 1]
            sigmaZ[n] = samplePoints[n, 2]
            t = sampleParameters[n, 1]
            s = sampleParameters[n, 0]
            for l in range(M0 + support0):
                for k in range(M1 + support1):
                    kp = k + (M1 + support1) * l
                    tVal = t - (k - 2.0)
                    if -2.0 < tVal < 2.0:
                        basisFactor1 = 2.0 / 3.0 - (abs(tVal) ** 2) + (abs(tVal) ** 3) / 2.0 if abs(tVal) < 1.0 else ((2.0 - abs(tVal)) ** 3) / 6.0
                    else:
                        basisFactor1 = 0.0
                    sVal = s - (l - 2.0)
                    if -2.0 < sVal < 2.0:
                        basisFactor2 = 2.0 / 3.0 - (abs(sVal) ** 2) + (abs(sVal) ** 3) / 2.0 if abs(sVal) < 1.0 else ((2.0 - abs(sVal)) ** 3) / 6.0
                    else:
                        basisFactor2 = 0.0
                    phi[n, kp] += basisFactor1 * basisFactor2

        cX = np.linalg.lstsq(phi, sigmaX, rcond=None)[0]
        cY = np.linalg.lstsq(phi, sigmaY, rcond=None)[0]
        cZ = np.linalg.lstsq(phi, sigmaZ, rcond=None)[0]

        for l in range(M0 + support0):
            for k in range(M1 + support1):
                kp = k + (M1 + support1) * l
                self.controlPoints[kp] = np.array([cX[kp], cY[kp], cZ[kp]], dtype=np.float64)

    cpdef parametersToWorld(self, tuple params):
        cdef np.ndarray point = np.zeros(3, dtype=np.float64)
        return self.nonPoleContributions(params[1], params[0], point)

    cdef np.ndarray nonPoleContributions(self, double t, double s, np.ndarray point):
        cdef int l, k, kp
        cdef double tVal, sVal, basisFactor1, basisFactor2
        cdef int M0 = self.M[0]
        cdef int M1 = self.M[1]
        cdef int support0 = 4
        cdef int support1 = 4
        for l in range(M0 + support0):
            sVal = s - (l - 2.0)
            if -2.0 < sVal < 2.0:
                for k in range(M1 + support1):
                    tVal = t - (k - 2.0)
                    if -2.0 < tVal < 2.0:
                        basisFactor1 = 2.0 / 3.0 - (abs(tVal) ** 2) + (abs(tVal) ** 3) / 2.0 if abs(tVal) < 1.0 else ((2.0 - abs(tVal)) ** 3) / 6.0
                        basisFactor2 = 2.0 / 3.0 - (abs(sVal) ** 2) + (abs(sVal) ** 3) / 2.0 if abs(sVal) < 1.0 else ((2.0 - abs(sVal)) ** 3) / 6.0
                        kp = k + (M1 + support1) * l
                        point += self.controlPoints[kp] * basisFactor1 * basisFactor2
        return point
