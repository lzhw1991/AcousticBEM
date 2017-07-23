from NativeInterface import *
import numpy as np


def ComputeL(k, p, qa, qb, qc, pOnElement):                           
    result = Complex()                                                         
    p = Float3(p[0],   p[1],  p[2])                                            
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(pOnElement)                                                    
    helmholtz.ComputeL_3D(c_float(k), p, a, b, c, on, byref(result))           
    return np.complex64(result.re+result.im*1j)                                
                                                                           
def ComputeM(k, p, qa, qb, qc, pOnElement):                               
    result = Complex()                                                         
    p = Float3(p[0], p[1], p[2])                                               
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(pOnElement)                                                    
    helmholtz.ComputeM_3D(c_float(k), p, a, b, c, on, byref(result))           
    return np.complex64(result.re+result.im*1j)                                
                                                                           
def ComputeMt(k, p, vec_p, qa, qb, qc, pOnElement):                       
    result = Complex()                                                         
    p = Float3(p[0], p[1], p[2])                                               
    vp = Float3(vec_p[0], vec_p[1], vec_p[2])                                  
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(pOnElement)                                                    
    helmholtz.ComputeMt_3D(c_float(k), p, vp, a, b, c, on, byref(result))      
    return np.complex64(result.re+result.im*1j)                                
                                                                           
def ComputeN(k, p, vec_p, qa, qb, qc, pOnElement):                        
    result = Complex()                                                         
    p = Float3(p[0], p[1], p[2])                                               
    vp = Float3(vec_p[0], vec_p[1], vec_p[2])                                  
    a = Float3(qa[0], qa[1], qa[2])                                            
    b = Float3(qb[0], qb[1], qb[2])                                            
    c = Float3(qc[0], qc[1], qc[2])                                            
    on = c_bool(pOnElement)                                                    
    helmholtz.ComputeN_3D(c_float(k), p, vp, a, b, c, on, byref(result))       
    return np.complex64(result.re+result.im*1j)                                
    
