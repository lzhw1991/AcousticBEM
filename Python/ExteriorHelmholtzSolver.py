from Geometry import *
from BoundaryData import *
from AcousticProperties import *
import numpy as np
    
class ExteriorHelmholtzSolver(object):
    
    def __init__(self, aVertex = None, aElement = None, c = 344.0, density = 1.205):
        assert not (aVertex is None), "Cannot construct ExteriorHelmholtzProblem2D without valid vertex array."
        self.aVertex = aVertex
        assert not (aElement is None), "Cannot construct ExteriorHelmholtzProblem2D without valid element array."
        self.aElement = aElement
        self.c       = c
        self.density = density
        # compute the centers of the discrete elements (depending on king, i.e line segment or triangle).
        # The solver computes the velocity potential at at these center points. 
        if (self.aElement.shape[1] ==  2):
            self.aCenters = 0.5 * (self.aVertex[self.aElement[:, 0]] + self.aVertex[aElement[:, 1]])
        elif (self.aElement.shape[1] == 3):
            self.aCenters = 1.0/3.0 * (self.aVertex[self.aElement[:, 0]] +\
                                       self.aVertex[self.aElement[:, 1]] +\
                                       self.aVertex[self.aElement[:, 2]])
    def __repr__(self):
        result = "ExteriorHelmholtzSolover("
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
            # Note, the only difference between the interior solver and this
            # one is the sign of the assignment below.
            c[i] = -(boundaryIncidence.phi[i] + mu * boundaryIncidence.v[i])

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
