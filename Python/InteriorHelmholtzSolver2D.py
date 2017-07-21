import numpy as np
from numpy.linalg import norm
from scipy.special import hankel1

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
        res += "Wavenumber (Frequency): {} ({} Hz)\n\n".format(self.k, self.parent.wavenumberToFrequency(self.k))
        res += "index          Potential                   Pressure                    Velocity              Intensity\n"
        for i in range(self.aPhi.size):
            pressure = self.parent.soundPressure(self.k, self.aPhi[i])
            intensity = self.parent.AcousticIntensity(pressure, self.aV[i])
            res += "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e}\n".format( \
                    i+1, self.aPhi[i].real, self.aPhi[i].imag, pressure.real, pressure.imag,                      \
                    self.aV[i].real, self.aV[i].imag, intensity)
        return res
    
class InteriorHelmholtzSolver2D(object):
    
    def __init__(self, aVertex = None, aElement = None, c = 344.0, density = 1.205):
        assert not (aVertex is None), "Cannot construct InteriorHelmholtzProblem2D without valid vertex array."
        self.aVertex = aVertex
        assert not (aElement is None), "Cannot construct InteriorHelmholtzProblem2D without valid element array."
        self.aElement = aElement
        self.c       = c
        self.density = density
        
    def __repr__(self):
        result = "InteriorHelmholtzProblem2D("
        result += "aVertex = " + repr(self.aVertex) + ", "
        result += "aElement = " + repr(self.aElement) + ", "
        result += "c = " + repr(self.c) + ", "
        result += "rho = " + repr(self.rho) + ")"
        return result

    def wavenumberToFrequency(self, k):
        return 0.5 * k * self.c / np.pi

    def frequencyToWavenumber(self, f):
        return 2.0 * np.pi * f / self.c

    @classmethod
    def Normal2D(cls, pointA, pointB):
        diff = pointA - pointB
        len = norm(diff)
        return np.array([diff[1]/len, -diff[0]/len])
    
    @classmethod
    def ComplexQuad(cls, func, start, end):
        samples = np.array([[0.980144928249, 5.061426814519E-02], 
                            [0.898333238707, 0.111190517227], 
                            [0.762766204958, 0.156853322939], 
                            [0.591717321248, 0.181341891689], 
                            [0.408282678752, 0.181341891689],
                            [0.237233795042, 0.156853322939], 
                            [0.101666761293, 0.111190517227],
                            [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)
        vec = end - start
        sum = 0.0
        for n in range(samples.shape[0]):
            x = start + samples[n, 0] * vec
            sum += samples[n, 1] * func(x)
        return sum * norm(vec)
    
    @classmethod
    def ComputeL(cls, k, p, qa, qb, pOnElement):
        qab = qb - qa
        if pOnElement:
            if k == 0.0:
                ra = norm(p - qa)
                rb = norm(p - qb)
                re = norm(qab)
                return 0.5 / np.pi * (re - (ra * np.log(ra) + rb * np.log(rb)))
            else:
                def func(x):
                    R = norm(p - x)
                    return 0.5 / np.pi * np.log(R) + 0.25j * hankel1(0, k * R)
                return cls.ComplexQuad(func, qa, p)  + cls.ComplexQuad(func, p, qa) \
                     + cls.ComputeL(0.0, p, qa, qb, True)
        else:
            if k == 0.0:
                return -0.5 / np.pi * cls.ComplexQuad(lambda q: np.log(norm(p - q)), qa, qb)
            else:
                return 0.25j * cls.ComplexQuad(lambda q: hankel1(0, k * norm(p - q)), qa, qb)
        return 0.0
        
    @classmethod
    def ComputeM(cls, k, p, qa, qb, pOnElement):
        qab = qb - qa
        vecq = cls.Normal2D(qa, qb)
        if pOnElement:
            return 0.0
        else:
            if k == 0.0:
                def func(x):
                    r = p - x
                    return np.dot(r, vecq) / np.dot(r, r)
                return -0.5 / np.pi * cls.ComplexQuad(func, qa, qb)
            else:
                def func(x):
                    r = p - x
                    R = norm(r)
                    return hankel1(1, k * R) * np.dot(r, vecq) / R
                return 0.25j * k * cls.ComplexQuad(func, qa, qb)
        return 0.0

    @classmethod
    def ComputeMt(cls, k, p, vecp, qa, qb, pOnElement):
        qab = qb - qa
        if pOnElement:
            return 0.0
        else:
            if k == 0.0:
                def func(x):
                    r = p - x
                    return np.dot(r, vecp) / np.dot(r, r)
                return -0.5 / np.pi * cls.ComplexQuad(func, qa, qb)
            else:
                def func(x):
                    r = p - x
                    R = norm(r)
                    return hankel1(1, k * R) * np.dot(r, vecp) / R
                return -0.25j * k * cls.ComplexQuad(func, qa, qb)
    
    @classmethod
    def ComputeN(cls, k, p, vecp, qa, qb, pOnElement):
        qab = qb- qa
        if pOnElement:
            ra = norm(p - qa)
            rb = norm(p - qb)
            re = norm(qab)
            if k == 0.0:
                return -(1.0 / ra + 1.0 / rb) / (re * 2.0 * np.pi) * re
            else:
                vecq = cls.Normal2D(qa, qb)
                k2 = k * k
                def func(x):
                    r = p - x
                    R2 = np.dot(r, r)
                    R = np.sqrt(R2)
                    drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2
                    dpnu = np.dot(vecp, vecq)
                    c1 =  0.25j * k / R * hankel1(1, k * R)                                  - 0.5 / (np.pi * R2)
                    c2 =  0.50j * k / R * hankel1(1, k * R) - 0.25j * k2 * hankel1(0, k * R) - 1.0 / (np.pi * R2)
                    c3 = -0.25  * k2 * np.log(R) / np.pi
                    return c1 * dpnu + c2 * drdudrdn + c3
                return cls.ComputeN(0.0, p, vecp, qa, qb, True) - 0.5 * k2 * cls.ComputeL(0.0, p, qa, qb, True) \
                     + cls.ComplexQuad(func, qa, p) + cls.ComplexQuad(func, p, qb)
        else:
            sum = 0.0j
            vecq = cls.Normal2D(qa, qb)
            un = np.dot(vecp, vecq)
            if k == 0.0:
                def func(x):
                    r = p - x
                    R2 = np.dot(r, r)
                    drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / R2
                    return (un + 2.0 * drdudrdn) / R2
                return 0.5 / np.pi * cls.ComplexQuad(func, qa, qb)
            else:
                def func(x):
                    r = p - x
                    drdudrdn = -np.dot(r, vecq) * np.dot(r, vecp) / np.dot(r, r)
                    R = norm(r)
                    return hankel1(1, k * R) / R * (un + 2.0 * drdudrdn) - k * hankel1(0, k * R) * drdudrdn 
                return 0.25j * k * cls.ComplexQuad(func, qa, qb)   
        
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

    def computeBoundaryMatrices(self, k, mu):
        A = np.empty((self.aElement.shape[0], self.aElement.shape[0]), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.aElement.shape[0]):
            pa = self.aVertex[self.aElement[i, 0]]
            pb = self.aVertex[self.aElement[i, 1]]
            pab = pb - pa
            center = 0.5 * (pa + pb)
            centerNormal = self.Normal2D(pa, pb)
            for j in range(self.aElement.shape[0]):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                elementL  = self.ComputeL(k, center, qa, qb, i==j)
                elementM  = self.ComputeM(k, center, qa, qb, i==j)
                elementMt = self.ComputeMt(k, center, centerNormal, qa, qb, i==j)
                elementN  = self.ComputeN(k, center, centerNormal, qa, qb, i==j)
                
                A[i, j] = elementL + mu * elementMt
                B[i, j] = elementM + mu * elementN

            A[i,i] -= 0.5 * mu
            B[i,i] += 0.5

        return A, B
    
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

    
    def solveInterior(self, solution, aIncidentInteriorPhi, aInteriorPoints):
        assert aIncidentInteriorPhi.shape == aInteriorPoints.shape[:-1], \
            "Incident phi vector and interior points vector must match"

        aResult = np.empty(aInteriorPoints.shape[0], dtype=complex)

        for i in range(aIncidentInteriorPhi.size):
            p  = aInteriorPoints[i]
            sum = aIncidentInteriorPhi[i]
            for j in range(solution.aPhi.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                elementL  = self.ComputeL(solution.k, p, qa, qb, False)
                elementM  = self.ComputeM(solution.k, p, qa, qb, False)

                sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
            aResult[i] = sum
        return aResult

    def soundPressure(self, k, phi, t = 0.0):
        angularVelocity = k * self.c
        return (1j * self.density * angularVelocity  * np.exp(-1.0j*angularVelocity*t)
                * phi).astype(np.complex64)

    @staticmethod
    def SoundMagnitude(pressure):
        return np.log10(np.abs(pressure / 2e-5)) * 20

    @staticmethod
    def AcousticIntensity(pressure, velocity):
        return 0.5 * (np.conj(pressure) * velocity).real

    @staticmethod
    def SignalPhase(pressure):
        return np.arctan2(pressure.imag, pressure.real)


def printInteriorSolution(solution, aPhiInterior):
    print "\nSound pressure at the interior points\n"
    print "index          Potential                    Pressure               Magnitude         Phase\n"
    for i in range(aPhiInterior.size):
        pressure = solution.parent.soundPressure(solution.k, aPhiInterior[i])
        magnitude = solution.parent.SoundMagnitude(pressure)
        phase = solution.parent.SignalPhase(pressure)
        print "{:5d}  {: 1.4e}+ {: 1.4e}i   {: 1.4e}+ {: 1.4e}i    {: 1.4e} dB       {:1.4f}".format( \
            i+1, aPhiInterior[i].real, aPhiInterior[i].imag, pressure.real, pressure.imag, magnitude, phase)
