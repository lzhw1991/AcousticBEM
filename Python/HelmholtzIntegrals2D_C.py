# ---------------------------------------------------------------------------
# Copyright (C) 2017 Frank Jargstorff
#
# This file is part of the AcousticBEM library.
# AcousticBEM is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# AcousticBEM is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with AcousticBEM.  If not, see <http://www.gnu.org/licenses/>.
# ---------------------------------------------------------------------------
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
