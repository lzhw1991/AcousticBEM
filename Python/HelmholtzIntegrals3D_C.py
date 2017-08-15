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
    
