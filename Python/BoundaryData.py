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
import numpy as np
from AcousticProperties import *

class BoundaryCondition(object):
    def __init__(self, size):
        self.alpha = np.empty(size, dtype = np.complex64)
        self.beta  = np.empty(size, dtype = np.complex64)
        self.f     = np.empty(size, dtype = np.complex64)

class BoundaryIncidence(object):
    def __init__(self, size):
        self.phi = np.empty(size, dtype = np.complex64);
        self.v   = np.empty(size, dtype = np.complex64);
        
class BoundarySolution(object):
    
    def __init__(self, parent, k, aPhi, aV):
        self.parent = parent
        self.k      = k
        self.aPhi   = aPhi
        self.aV     = aV
    
    def __repr__(self):
        result = "Solution2D("
        result += "parent = " + repr(self.parent) + ", "
        result += "k = " + repr(self.k) + ", "
        result += "aPhi = " + repr(self.aPhi) + ", "
        result += "aV = " + repr(self.aV) + ")"
        return result
    
    def __str__(self):
        res =  "Density of medium:      {} kg/m^3\n".format(self.parent.density)
        res += "Speed of sound:         {} m/s\n".format(self.parent.c)
        res += "Wavenumber (Frequency): {} ({} Hz)\n\n".format(self.k, wavenumberToFrequency(self.k))
        res += "index          Potential                   Pressure                    Velocity              Intensity\n"
        for i in range(self.aPhi.size):
            pressure = soundPressure(self.k, self.aPhi[i], c=self.parent.c, density=self.parent.density)
            intensity = AcousticIntensity(pressure, self.aV[i])
            res += "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e}\n".format( \
                    i+1, self.aPhi[i].real, self.aPhi[i].imag, pressure.real, pressure.imag,                      \
                    self.aV[i].real, self.aV[i].imag, intensity)
        return res

    def radiationRatio(self):
        solver = self.parent
        power = 0.0
        bpower = 0.0
        for i in range(self.aPhi.size):
            pressure = soundPressure(self.k, self.aPhi[i], c=self.parent.c, density=self.parent.density)
            power += AcousticIntensity(pressure, self.aV[i])
            bpower += (self.parent.density * self.parent.c * np.abs(self.aV[i])**2)
        return 2 * power / bpower
