from InteriorHelmholtzSolver2D import *


class CircularIntegratorPi(object):
    """ 
    Integrator class for integrating the upper half-circle or in other
    words integrate a function along the unit acr over angles 
    theta in [0, pi].
    """
    samples = np.array([[0.980144928249, 5.061426814519E-02], 
                        [0.898333238707, 0.111190517227], 
                        [0.762766204958, 0.156853322939], 
                        [0.591717321248, 0.181341891689], 
                        [0.408282678752, 0.181341891689],
                        [0.237233795042, 0.156853322939], 
                        [0.101666761293, 0.111190517227],
                        [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)

    def __init__(self, segments):
        self.segments = segments
        nSamples = segments * self.samples.shape[0]
        self.rotationFactors = np.empty((nSamples, 2), dtype=np.float32)
        
        factor = np.pi / self.segments
        for i in range(nSamples):
            arcAbscissa = i / self.samples.shape[0] + self.samples[i % self.samples.shape[0], 0]
            arcAbscissa *= factor
            self.rotationFactors[i, :] = np.cos(arcAbscissa), np.sin(arcAbscissa)

    def integrate(self, func):
        sum = 0.0
        for n in range(self.rotationFactors.shape[0]):
            sum += self.samples[n % self.samples.shape[0], 1] * func(self.rotationFactors[n, :])
        return sum * np.pi / self.segments

def ComplexQuadGenerator(func, start, end):
    """
    This is a variation on the basic complex quadrature function from the
    base class. The difference is, that the abscissa values y**2 have been
    substituted for x. Kirkup doesn't explain the details of why this
    is helpful for the case of this kind of 2D integral evaluation, but points
    to his PhD thesis and another reference that I have no access to.
    """
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
        x = start + samples[n, 0]**2 * vec
        sum += samples[n, 1] * func(x) * samples[n, 0]

    return 2.0 * sum * norm(vec)

class InteriorHelmholtzSolverRAD(InteriorHelmholtzSolver2D):
    
    def __init__(self, *args, **kwargs):
        super(InteriorHelmholtzSolverRAD, self).__init__(*args, **kwargs)

    @classmethod
    def ComplexQuadCone(cls, func, start, end, segments = 1):
        delta = 1.0 / segments * (end - start)
        sum = 0.0
        for s in range(segments):
            sum += cls.ComplexQuad(func, start + s * delta, start + (s+1) * delta)
            
        return sum


    @classmethod
    def ComputeL(cls, k, p, qa, qb, pOnElement):
        qab = qb - qa
        # subdived circular integral into sections of
        # similar size as qab
        q = 0.5 * (qa + qb)
        nSections = 1 + int(q[0] * np.pi / norm(qab))
        if pOnElement:
            ap = p - qa
                
            if k == 0.0:
                def generatorFunc(x):
                    circle = CircularIntegratorPi(2 * nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        rr = q3 - p3
                        return 1.0 / norm(rr)

                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)

                return ComplexQuadGenerator(generatorFunc, p, qa) + ComplexQuadGenerator(generatorFunc, p, qb)

            else:
                def generatorFunc(x):
                    circle = CircularIntegratorPi(2 * nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                    
                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        rr = q3 - p3
                        RR = norm(rr)
                        return (np.exp(1.0j * k * RR) - 1.0) / RR

                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)

                return cls.ComputeL(0.0, p, qa, qb, True) + cls.ComplexQuad(generatorFunc, qa, qb)

        else:
            if k == 0.0:
                def generatorFunc(x):
                    circle = CircularIntegratorPi(nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        rr = q3 - p3
                        return 1.0 / norm(rr)

                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)

                return cls.ComplexQuad(generatorFunc, qa, qb)

            else:
                def generatorFunc(x):
                    circle = CircularIntegratorPi(nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                    
                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        rr = q3 - p3
                        RR = norm(rr)
                        return np.exp(1.0j * k * RR) / RR

                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)


                return cls.ComplexQuad(generatorFunc, qa, qb)

        return 0.0
        
    @classmethod
    def ComputeM(cls, k, p, qa, qb, pOnElement):
        qab = qb - qa
        vec_q = cls.Normal2D(qa, qb)

        # subdived circular integral into sections of
        # similar size as qab
        q = 0.5 * (qa + qb)
        nSections = 1 + int(q[0] * np.pi / norm(qab))
        
        if k == 0.0:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                
                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                    rr = q3 - p3
                    
                    return -np.dot(rr, vec_q3) / (norm(rr) * np.dot(rr, rr))
                
                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            if pOnElement:
                return cls.ComplexQuad(generatorFunc, qa, p) + cls.ComplexQuad(generatorFunc, p, qb)
            else:
                return cls.ComplexQuad(generatorFunc, qa, qb)
        
        else:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                
                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                    rr = q3 - p3
                    RR = norm(rr)
                    return (1j * k * RR - 1.0) * np.exp(1j * k * RR) * np.dot(rr, vec_q3) / (RR *  np.dot(rr, rr))
                
                return circle.integrate(circleFunc) * r / (2.0 * np.pi)
            
            if pOnElement:
                return cls.ComplexQuad(generatorFunc, qa, p) + cls.ComplexQuad(generatorFunc, p, qb)
            else:
                return cls.ComplexQuad(generatorFunc, qa, qb)
        
        return 0.0
        
    @classmethod
    def ComputeMt(cls, k, p, vecp, qa, qb, pOnElement):
        qab = qb - qa

        # subdived circular integral into sections of
        # similar size as qab
        q = 0.5 * (qa + qb)
        nSections = 1 + int(q[0] * np.pi / norm(qab))
        
        if k == 0.0:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                
                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    rr = q3 - p3
                    dotRnP = vecp[0] * rr[0] + vec[1] * rr[2]
                    return dotRnP / (norm(rr) * np.dot(rr, rr))
                
                return circle.integrate(circleFunc) * r / (2.0 * np.pi)

            if pOnElement:
                return cls.ComplexQuad(generatorFunc, qa, p) + cls.ComplexQuad(generatorFunc, p, qb)
            else:
                return cls.ComplexQuad(generatorFunc, qa, qb)

        else:
            def generatorFunc(x):
                circle = CircularIntegratorPi(nSections)
                r = x[0]
                z = x[1]
                p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                
                def circleFunc(x):
                    q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                    rr = q3 - p3
                    RR = norm(rr)
                    dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                    return -(1j * k * RR - 1.0) * np.exp(1j * k * RR) * dotRnP / (RR *  np.dot(rr, rr))
                
                return circle.integrate(circleFunc) * r / (2.0 * np.pi)
            
            
            if pOnElement:
                return cls.ComplexQuad(generatorFunc, qa, p) + cls.ComplexQuad(generatorFunc, p, qb)
            else:
                return cls.ComplexQuad(generatorFunc, qa, qb)
    
    @classmethod
    def ComputeN(cls, k, p, vecp, qa, qb, pOnElement):
        qab = qb - qa
        vec_q = cls.Normal2D(qa, qb)

        # subdived circular integral into sections of
        # similar size as qab
        q = 0.5 * (qa + qb)
        nSections = 1 + int(q[0] * np.pi / norm(qab))

        if pOnElement:
            if k == 0.0:
                vecp3 = np.array([vecp[0], 0.0, vecp[1]], dtype=np.float32)
                def coneFunc(x, direction):
                    circle = CircularIntegratorPi(nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)

                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        vec_q3 = np.sqrt(0.5) * np.array([x[0], x[1], direction], dtype=np.float32)
                        dnpnq = np.dot(vecp3, vec_q3)
                        rr = q3 - p3
                        RR = norm(rr)
                        dotRNP = np.dot(rr, vecp3)
                        dotRNQ = -np.dot(rr, vec_q3)
                        RNPRNQ = dotRNP * dotRNQ / np.dot(rr, rr)
                        return (dnpnq + 3.0 * RNPRNQ) / (RR * np.dot(rr, rr))
                    
                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)
                        
                lenAB = norm(qab)
                # deal with the cone at the qa side of the generator
                direction = np.sign(qa[1] - qb[1])
                if direction == 0.0:
                    direction = 1.0
                tip_a = np.array([0.0, qa[1] + direction * qa[0]], dtype=np.float32)
                nConeSectionsA = int(qa[0] * np.sqrt(2.0) / lenAB) + 1
                coneValA = cls.ComplexQuadCone(lambda x: coneFunc(x, direction), qa, tip_a, nConeSectionsA)
                
                # deal with the cone at the qb side of the generator
                direction = np.sign(qb[1] - qa[1])
                if direction == 0.0:
                    direction = -1.0
                tip_b = np.array([0.0, qb[1] + direction * qb[0]], dtype=np.float32)
                nConeSectionsB = int(qb[0] * np.sqrt(2.0) / lenAB) + 1
                coneValB = cls.ComplexQuadCone(lambda x: coneFunc(x, direction), qb, tip_b, nConeSectionsB)

                return -(coneValA + coneValB)
            
            else:
                def generatorFunc(x):
                    circle = CircularIntegratorPi(nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                
                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                        rr = q3 - p3
                        RR = norm(rr)
                        DNPNQ = vecp[0] * vec_q3[0] + vecp[1] * vec_q3[2]
                        dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                        dotRnQ = -np.dot(rr, vec_q3)
                        RNPRNQ = dotRnP * dotRnQ / np.dot(rr, rr)
                        RNPNQ = -(DNPNQ + RNPRNQ) / RR
                        IKR = 1j * k * RR
                        FPG0 = 1.0 / RR
                        FPGR = np.exp(IKR) / np.dot(rr, rr) * (IKR - 1.0)
                        FPGR0 = -1.0 / np.dot(rr, rr)
                        FPGRR = np.exp(IKR) * (2.0 - 2.0 * IKR - (k*RR)**2) / (RR * np.dot(rr, rr))
                        FPGRR0 = 2.0 / (RR * np.dot(rr, rr))
                        return (FPGR - FPGR0) * RNPNQ + (FPGRR - FPGRR0) * RNPRNQ \
                            + k**2 * FPG0 / 2.0
                
                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)

                return cls.ComputeN(0.0, p, vecp, qa, qb, True) - k**2 * cls.ComputeL(0.0, p, qa, qb, True) / 2.0 \
                    + cls.ComplexQuad(generatorFunc, qa, p) + cls.ComplexQuad(generatorFunc, p, qb)
            
        else:
            if k == 0.0:
                def generatorFunc(x):
                    circle = CircularIntegratorPi(nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                
                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                        rr = q3 - p3
                        RR = norm(rr)
                        DNPNQ = vecp[0] * vec_q3[0] + vecp[1] * vec_q3[2]
                        dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                        dotRnQ = -np.dot(rr, vec_q3)
                        RNPRNQ = dotRnP * dotRnQ / np.dot(rr, rr)
                        RNPNQ = -(DNPNQ + RNPRNQ) / RR
                        IKR = 1j * k * RR
                        FPGR = -1.0 / np.dot(rr, rr)
                        FPGRR = 2.0 / (RR * np.dot(rr, rr))
                        return FPGR * RNPNQ + FPGRR * RNPRNQ

                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)

                return cls.ComplexQuad(generatorFunc, qa, qb)
            else:
                def generatorFunc(x):
                    circle = CircularIntegratorPi(nSections)
                    r = x[0]
                    z = x[1]
                    p3 = np.array([p[0], 0.0, p[1]], dtype=np.float32)
                
                    def circleFunc(x):
                        q3 = np.array([r * x[0], r * x[1], z], dtype=np.float32)
                        vec_q3 = np.array([vec_q[0] * x[0], vec_q[0] * x[1], vec_q[1]], dtype=np.float32)
                        rr = q3 - p3
                        RR = norm(rr)
                        DNPNQ = vecp[0] * vec_q3[0] + vecp[1] * vec_q3[2]
                        dotRnP = vecp[0] * rr[0] + vecp[1] * rr[2]
                        dotRnQ = -np.dot(rr, vec_q3)
                        RNPRNQ = dotRnP * dotRnQ / np.dot(rr, rr)
                        RNPNQ = -(DNPNQ + RNPRNQ) / RR
                        IKR = 1j * k * RR
                        FPGR = np.exp(IKR) / np.dot(rr, rr) * (IKR - 1.0)
                        FPGRR = np.exp(IKR) * (2.0 - 2.0 * IKR - (k*RR)**2) / (RR * np.dot(rr, rr))
                        return FPGR * RNPNQ + FPGRR * RNPRNQ
                
                    return circle.integrate(circleFunc) * r / (2.0 * np.pi)

                return cls.ComplexQuad(generatorFunc, qa, qb)


    def solveBoundary(self, k, boundaryCondition, boundaryIncidence, mu = None):
        mu = mu or (1j / (k + 1))
        wkspc1 = np.empty((boundaryCondition.f.size, boundaryCondition.f.size), dtype=complex)
        wkspc2 = np.empty(wkspc1.shape, dtype=complex)
        wkspc6 = np.empty(boundaryCondition.f.size, dtype=complex)

        elementL  = 0.0j
        elementM  = 0.0j
        elementMt = 0.0j
        elementN  = 0.0j

        quadSamples = np.array([[0.980144928249, 5.061426814519E-02], 
                            [0.898333238707, 0.111190517227], 
                            [0.762766204958, 0.156853322939], 
                            [0.591717321248, 0.181341891689], 
                            [0.408282678752, 0.181341891689],
                            [0.237233795042, 0.156853322939], 
                            [0.101666761293, 0.111190517227],
                            [1.985507175123E-02, 5.061426814519E-02]], dtype=np.float32)
        agqon = np.empty(16, dtype=np.float32)
        wgqon = np.empty(16, dtype=np.float32)
        agqoff = quadSamples[:, 0]
        wgqoff = quadSamples[:, 1]
        agqon[:8] = agqoff
        agqon[8:] = agqoff
        wgqon[:8] = wgqoff
        wgqon[8:] = wgqoff

        for i in range(boundaryCondition.f.size):
            pa = self.aVertex[self.aElement[i, 0]]
            pb = self.aVertex[self.aElement[i, 1]]
            pab = pb - pa
            center = 0.5 * (pa + pb)
            centerNormal = self.Normal2D(pa, pb)
            for j in range(boundaryCondition.f.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                # Select quadrature rule for H3ALC
                #  Select the quadrature rule AGQON-WGQON in the case when the
                #  point p lies on the element, otherwise select AGQOFF-WGQOFF
                #  [Note that the overall method would benefit from selecting from
                #  a wider set of quadrature rules, and an appropriate method
                #  of selection]
                if i==j:
                    agq = agqon
                    wgq = wgqon
                else:
                    agq = agqoff
                    wgq = wgqoff

                # Quadrature rule in the theta direction is constructed outof individual
                # Gauss rules so that the lentgh of each is approximately equal to the
                # length of the element at the generator.
                RADMID = 0.5 * (qa[0] + qb[0]) # midpoint between the to radii
                SGLEN = np.dot(qa - qb, qa - qb) 
                GLEN = np.sqrt(SGLEN)
                CIRMID = np.pi * RADMID
                NDIV = 1 + int(CIRMID / GLEN)
                TDIV = 1.0/NDIV
                wtq = np.empty(NDIV * agq.size, dtype=np.float32)
                atq = np.empty(NDIV * agq.size, dtype=np.float32)
                for m in range(NDIV):
                    for n in range(agq.size):
                        wtq[m * agq.size + n] = wgq[n] / NDIV
                        atq[m * agq.size + n] = agq[n] / NDIV + TDIV * k
                
                elementL  = self.ComputeL(k, center, qa, qb, i==j)
                elementM  = self.ComputeM(k, center, qa, qb, i==j)
                elementMt = self.ComputeMt(k, center, centerNormal, qa, qb, i==j)
                elementN  = self.ComputeN(k, center, centerNormal, qa, qb, i==j)
                
                wkspc1[i, j] = elementL + mu * elementMt
                wkspc2[i, j] = elementM + mu * elementN

            wkspc1[i,i] -= 0.5 * mu
            wkspc2[i,i] += 0.5
            wkspc6[i] = boundaryIncidence.phi[i] + mu * boundaryIncidence.v[i]

        phi, v = self.SolveLinearEquation(wkspc2, wkspc1, wkspc6,
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

