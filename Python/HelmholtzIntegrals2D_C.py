from NativeInterface import *
import numpy as np

def hankel1(order, x):
    z = Complex()
    helmholtz.Hankel1(c_int(order), c_float(x), byref(z))
    return np.complex64(z.re + z.im*1j)

def ComputeL(k, p, qa, qb, pOnElement):                                          
    result = Complex()                                                           
    p = Float2(p[0], p[1])                                                       
    qa = Float2(qa[0], qa[1])                                                    
    qb = Float2(qb[0], qb[1])                                                    
    x = c_bool(pOnElement)                                                       
    helmholtz.ComputeL_2D(c_float(k), p, qa, qb, x, byref(result))               
    return np.complex64(result.re+result.im*1j)                                  
                                                                                 
def ComputeM(k, p, qa, qb, pOnElement):                                          
    result = Complex()                                                           
    p = Float2(p[0], p[1])                                                       
    qa = Float2(qa[0], qa[1])                                                    
    qb = Float2(qb[0], qb[1])                                                    
    x = c_bool(pOnElement)                                                       
    helmholtz.ComputeM_2D(c_float(k), p, qa, qb, x, byref(result))               
    return np.complex64(result.re+result.im*1j)                                  
                                                                                 
def ComputeMt(k, p, normal_p, qa, qb, pOnElement):                               
    result = Complex()                                                           
    p = Float2(p[0], p[1])                                                       
    normal_p = Float2(normal_p[0], normal_p[1])                                  
    qa = Float2(qa[0], qa[1])                                                    
    qb = Float2(qb[0], qb[1])                                                    
    x = c_bool(pOnElement)                                                       
    helmholtz.ComputeMt_2D(c_float(k), p, normal_p, qa, qb, x, byref(result))    
    return np.complex64(result.re+result.im*1j)                                  
                                                                                 
def ComputeN(k, p, normal_p, qa, qb, pOnElement):                                
    result = Complex()                                                           
    p = Float2(p[0], p[1])                                                       
    normal_p = Float2(normal_p[0], normal_p[1])                                  
    qa = Float2(qa[0], qa[1])                                                    
    qb = Float2(qb[0], qb[1])                                                    
    x = c_bool(pOnElement)                                                       
    helmholtz.ComputeN_2D(c_float(k), p, normal_p, qa, qb, x, byref(result))     
    return np.complex64(result.re+result.im*1j)
