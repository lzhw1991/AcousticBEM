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
