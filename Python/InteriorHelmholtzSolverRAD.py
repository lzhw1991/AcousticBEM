bOptimized = True
if bOptimized:
    from HelmholtzIntegralsRAD_C import *
else:
    from HelmholtzIntegralsRAD import *        
from InteriorHelmholtzSolver2D import *


class InteriorHelmholtzSolverRAD(InteriorHelmholtzSolver2D):
    
    def __init__(self, *args, **kwargs):
        super(InteriorHelmholtzSolverRAD, self).__init__(*args, **kwargs)

 
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
            centerNormal = Normal2D(pa, pb)
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
                
                elementL  = ComputeL(k, center, qa, qb, i==j)
                elementM  = ComputeM(k, center, qa, qb, i==j)
                elementMt = ComputeMt(k, center, centerNormal, qa, qb, i==j)
                elementN  = ComputeN(k, center, centerNormal, qa, qb, i==j)
                
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

                elementL  = ComputeL(solution.k, p, qa, qb, False)
                elementM  = ComputeM(solution.k, p, qa, qb, False)

                sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
            aResult[i] = sum
        return aResult

