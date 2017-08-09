from NativeInterface import *
import numpy as np

def ComputeL(k, p, qa, qb, pOnElement):
    result = Complex()
    pp = Float2(p[0], p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(pOnElement)
    helmholtz.ComputeL_RAD(c_float(k), pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)

def ComputeM(k, p, qa, qb, pOnElement):
    result = Complex()
    pp = Float2(p[0], p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(pOnElement)
    helmholtz.ComputeM_RAD(c_float(k), pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)

def ComputeMt(k, p, vec_p, qa, qb, pOnElement):
    result = Complex()
    pp = Float2(p[0], p[1])
    vec_pp = Float2(vec_p[0], vec_p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(pOnElement)
    helmholtz.ComputeMt_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result)    )

    return np.complex64(result.re+result.im*1j)

def ComputeN(k, p, vec_p, qa, qb, pOnElement):
    result = Complex()
    pp = Float2(p[0], p[1])
    vec_pp = Float2(vec_p[0], vec_p[1])
    a = Float2(qa[0], qa[1])
    b = Float2(qb[0], qb[1])
    x = c_bool(pOnElement)
    helmholtz.ComputeN_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result))

    return np.complex64(result.re+result.im*1j)
