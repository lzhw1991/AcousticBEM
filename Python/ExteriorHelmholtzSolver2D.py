from ExteriorHelmholtzSolver import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals2D_C import *
else:
    from HelmholtzIntegrals2D import *        
        
from Geometry import *
from AcousticProperties import *
import numpy as np

class ExteriorHelmholtzSolver2D(ExteriorHelmholtzSolver):

    def __init__(self, *args, **kwargs):
        super(ExteriorHelmholtzSolver2D, self).__init__(*args, **kwargs)

    def computeBoundaryMatrices(self, k, mu):
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

            # the following two lines are the only difference between
            # the exterior and interor problems, in particular the difference
            # is that the inplace addtion and subtraction trade places between
            # the two variants. TODO: Investigate how this could be implemented
            # with less cut-and-paste code.
            A[i,i] += 0.5 * mu
            B[i,i] -= 0.5

        return A, B
    
    def solveExterior(self, solution, aIncidentExteriorPhi, aExteriorPoints):
        assert aIncidentExteriorPhi.shape == aExteriorPoints.shape[:-1], \
            "Incident phi vector and exterior points vector must match"

        aResult = np.empty(aExteriorPoints.shape[0], dtype=complex)

        for i in range(aIncidentExteriorPhi.size):
            p  = aExteriorPoints[i]
            sum = aIncidentExteriorPhi[i]
            for j in range(solution.aPhi.size):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]

                elementL  = ComputeL(solution.k, p, qa, qb, False)
                elementM  = ComputeM(solution.k, p, qa, qb, False)

                # similarly to the computeBoundaryMatrices method above, the
                # only difference between the interior solver and this, the exterior
                # solver is that the signs in the assignment below trade places.
                # TODO: Investigate if it's possible to reduce redundant code in these
                # two solvers.
                sum -= elementL * solution.aV[j] - elementM * solution.aPhi[j]
            aResult[i] = sum
        return aResult

