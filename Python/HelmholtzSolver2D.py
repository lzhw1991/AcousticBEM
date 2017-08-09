from HelmholtzSolver import *

bOptimized = False
if bOptimized:
    from HelmholtzIntegrals2D_C import *
else:
    from HelmholtzIntegrals2D import *        
        
from Geometry import *
from AcousticProperties import *
import numpy as np


class HelmholtzSolver2D(HelmholtzSolver):
    def __init__(self, *args, **kwargs):
        super(HelmholtzSolver2D, self).__init__(*args, **kwargs)
        self.aCenters = 0.5 * (self.aVertex[self.aElement[:, 0]] + self.aVertex[self.aElement[:, 1]])
        # lenght of the boundary elements (for the 3d shapes this is replaced by aArea
        self.aLength = np.linalg.norm(self.aVertex[self.aElement[:, 0]] - self.aVertex[self.aElement[:, 1]])

    def computeBoundaryMatrices(self, k, mu, orientation):
        A = np.empty((self.aElement.shape[0], self.aElement.shape[0]), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.aElement.shape[0]):
            pa = self.aVertex[self.aElement[i, 0]]
            pb = self.aVertex[self.aElement[i, 1]]
            pab = pb - pa
            center = 0.5 * (pa + pb)
            centerNormal = Normal2D(pa, pb)
            for j in range(self.aElement.shape[0]):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                elementL  = ComputeL(k, center, qa, qb, i==j)
                elementM  = ComputeM(k, center, qa, qb, i==j)
                elementMt = ComputeMt(k, center, centerNormal, qa, qb, i==j)
                elementN  = ComputeN(k, center, centerNormal, qa, qb, i==j)
                
                A[i, j] = elementL + mu * elementMt
                B[i, j] = elementM + mu * elementN

            if orientation == 'interior':
                # interior variant, signs are reversed for exterior
                A[i,i] -= 0.5 * mu
                B[i,i] += 0.5
            elif orientation == 'exterior':
                A[i,i] += 0.5 * mu
                B[i,i] -= 0.5
            else:
                assert False, 'Invalid orientation: {}'.format(orientation)
                
        return A, B

    def computeBoundaryMatricesInterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'interior')
    
    def computeBoundaryMatricesExterior(self, k, mu):
        return self.computeBoundaryMatrices(k, mu, 'exterior')

    
    def solveSamples(self, solution, aIncidentPhi, aSamples, orientation):
        assert aIncidentPhi.shape == aSamples.shape[:-1], \
            "Incident phi vector and sample points vector must match"

        aResult = np.empty(aSamples.shape[0], dtype=complex)

        for i in range(aIncidentPhi.size):
            p  = aSamples[i]
            sum = aIncidentPhi[i]
            for j in range(solution.aPhi.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                elementL  = ComputeL(solution.k, p, qa, qb, False)
                elementM  = ComputeM(solution.k, p, qa, qb, False)
                if orientation == 'interior':
                    sum += elementL * solution.aV[j] - elementM * solution.aPhi[j]
                elif orientation == 'exterior':
                    sum -= elementL * solution.aV[j] - elementM * solution.aPhi[j]
                else:
                    assert False, 'Invalid orientation: {}'.format(orientation)
            aResult[i] = sum
        return aResult

    def solveInterior(self, solution, aIncidentInteriorPhi, aInteriorPoints):
        return self.solveSamples(solution, aIncidentInteriorPhi, aInteriorPoints, 'interior')
    
    def solveExterior(self, solution, aIncidentExteriorPhi, aExteriorPoints):
        return self.solveSamples(solution, aIncidentExteriorPhi, aExteriorPoints, 'exterior')
