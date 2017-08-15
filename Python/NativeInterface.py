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
from ctypes import *

helmholtz = CDLL("/home/fjargsto/AcousticBEM/C/libhelmholtz.so")

helmholtz.Hankel1.argtypes = [c_int, c_float, c_void_p]


class Complex(Structure):
    _fields_ = [('re', c_float), ('im', c_float)]
    
class Float2(Structure):
    _fields_ = [('x', c_float), ('y', c_float)]
    
class Float3(Structure):
    _fields_ = [('x', c_float), ('y', c_float), ('z', c_float)]
    
    
