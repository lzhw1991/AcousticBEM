from InteriorHelmholtzSolverRAD import *
from NativeInterface import *

class InteriorHelmholtzSolverRAD_C(InteriorHelmholtzSolverRAD):
    
    def __init__(self, *args, **kwargs):
        super(InteriorHelmholtzSolverRAD_C, self).__init__(*args, **kwargs)

    @classmethod
    def ComputeL(cls, k, p, qa, qb, pOnElement):
        result = Complex()
        pp = Float2(p[0], p[1])
        a = Float2(qa[0], qa[1])
        b = Float2(qb[0], qb[1])
        x = c_bool(pOnElement)
        helmholtz.ComputeL_RAD(c_float(k), pp, a, b, x, byref(result))

        return np.complex64(result.re+result.im*1j)
        
    @classmethod
    def ComputeM(cls, k, p, qa, qb, pOnElement):
        result = Complex()
        pp = Float2(p[0], p[1])
        a = Float2(qa[0], qa[1])
        b = Float2(qb[0], qb[1])
        x = c_bool(pOnElement)
        helmholtz.ComputeM_RAD(c_float(k), pp, a, b, x, byref(result))

        return np.complex64(result.re+result.im*1j)
        
    @classmethod
    def ComputeMt(cls, k, p, vec_p, qa, qb, pOnElement):
        result = Complex()
        pp = Float2(p[0], p[1])
        vec_pp = Float2(vec_p[0], vec_p[1])
        a = Float2(qa[0], qa[1])
        b = Float2(qb[0], qb[1])
        x = c_bool(pOnElement)
        helmholtz.ComputeMt_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result))

        return np.complex64(result.re+result.im*1j)
        
    @classmethod
    def ComputeN(cls, k, p, vec_p, qa, qb, pOnElement):
        result = Complex()
        pp = Float2(p[0], p[1])
        vec_pp = Float2(vec_p[0], vec_p[1])
        a = Float2(qa[0], qa[1])
        b = Float2(qb[0], qb[1])
        x = c_bool(pOnElement)
        helmholtz.ComputeN_RAD(c_float(k), pp, vec_pp, a, b, x, byref(result))

        return np.complex64(result.re+result.im*1j)
        
