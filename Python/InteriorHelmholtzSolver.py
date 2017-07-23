from Geometry import *
from AcousticProperties import *
import numpy as np

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
    
class InteriorHelmholtzSolver(object):
    
    def __init__(self, aVertex = None, aElement = None, c = 344.0, density = 1.205):
        assert not (aVertex is None), "Cannot construct InteriorHelmholtzProblem2D without valid vertex array."
        self.aVertex = aVertex
        assert not (aElement is None), "Cannot construct InteriorHelmholtzProblem2D without valid element array."
        self.aElement = aElement
        self.c       = c
        self.density = density
        
    def __repr__(self):
        result = "InteriorHelmholtzSolover("
        result += "aVertex = " + repr(self.aVertex) + ", "
        result += "aElement = " + repr(self.aElement) + ", "
        result += "c = " + repr(self.c) + ", "
        result += "rho = " + repr(self.rho) + ")"
        return result

    @classmethod
    def SolveLinearEquation(cls, Ai, Bi, ci, alpha, beta, f):
        A = np.copy(Ai)
        B = np.copy(Bi)
        c = np.copy(ci)

        x = np.empty(c.size, dtype=np.complex)
        y = np.empty(c.size, dtype=np.complex)

        gamma = norm(B, np.inf) / norm(A, np.inf)
        swapXY = np.empty(c.size, dtype=bool)
        for i in range(c.size):
            if np.abs(beta[i]) < gamma * np.abs(alpha[i]):
                swapXY[i] = False
            else:
                swapXY[i] = True

        for i in range(c.size):
            if swapXY[i]:
                for j in range(alpha.size):
                    c[j] += f[i] * B[j,i] / beta[i]
                    B[j, i] = -alpha[i] * B[j, i] / beta[i]
            else:
                for j in range(alpha.size):
                    c[j] -= f[i] * A[j, i] / alpha[i]
                    A[j, i] = -beta[i] * A[j, i] / alpha[i]

        A -= B
        y = np.linalg.solve(A, c)

        for i in range(c.size):
            if swapXY[i]:
                x[i] = (f[i] - alpha[i] * y[i]) / beta[i]
            else:
                x[i] = (f[i] - beta[i] * y[i]) / alpha[i]

        for i in range(c.size):
            if swapXY[i]:
                temp = x[i]
                x[i] = y[i]
                y[i] = temp

        return x, y

    def solveBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        mu = mu or (1j / (k + 1))
        assert boundaryCondition.f.size == self.aElement.shape[0]
        A, B = self.computeBoundaryMatrices(k, mu)
        c = np.empty(self.aElement.shape[0], dtype=complex)
        for i in range(self.aElement.shape[0]):
            c[i] = boundaryIncidence.phi[i] + mu * boundaryIncidence.v[i]

        phi, v = self.SolveLinearEquation(B, A, c,
                                          boundaryCondition.alpha,
                                          boundaryCondition.beta,
                                          boundaryCondition.f)
        return BoundarySolution(self, k, phi, v)

    
def printInteriorSolution(solution, aPhiInterior):
    print "\nSound pressure at the interior points\n"
    print "index          Potential                    Pressure               Magnitude         Phase\n"
    for i in range(aPhiInterior.size):
        pressure = soundPressure(solution.k, aPhiInterior[i], c=solution.parent.c, density=solution.parent.density)
        magnitude = SoundMagnitude(pressure)
        phase = SignalPhase(pressure)
        print "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase)
