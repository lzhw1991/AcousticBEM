from ExteriorHelmholtzSolver import *

bOptimized = True
if bOptimized:
    from HelmholtzIntegrals3D_C import *
else:
    from HelmholtzIntegrals3D import *        
        
from Geometry import *
from AcousticProperties import *
import numpy as np

class ExteriorHelmholtzSolver3D(ExteriorHelmholtzSolver):

    def __init__(self, *args, **kwargs):
        super(ExteriorHelmholtzSolver3D, self).__init__(*args, **kwargs)

    def computeBoundaryMatrices(self, k, mu):
        A = np.empty((self.aElement.shape[0], self.aElement.shape[0]), dtype=complex)
        B = np.empty(A.shape, dtype=complex)

        for i in range(self.aElement.shape[0]):
            pa = self.aVertex[self.aElement[i, 0]]
            pb = self.aVertex[self.aElement[i, 1]]
            pc = self.aVertex[self.aElement[i, 2]]            
            p = (pa + pb + pc) / 3.0
            centerNormal = Normal3D(pa, pb, pc)
            for j in range(self.aElement.shape[0]):
                qa = self.aVertex[self.aElement[j, 0]]
                qb = self.aVertex[self.aElement[j, 1]]
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(k, self.aCenters[i], qa, qb, qc, i==j)
                elementM  = ComputeM(k, self.aCenters[i], qa, qb, qc, i==j)
                elementMt = ComputeMt(k, self.aCenters[i], centerNormal, qa, qb, qc, i==j)
                elementN  = ComputeN(k, self.aCenters[i], centerNormal, qa, qb, qc, i==j)

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
                qc = self.aVertex[self.aElement[j, 2]]

                elementL  = ComputeL(solution.k, p, qa, qb, qc, False)
                elementM  = ComputeM(solution.k, p, qa, qb, qc, False)

                # similarly to the computeBoundaryMatrices method above, the
                # only difference between the interior solver and this, the exterior
                # solver is that the signs in the assignment below trade places.
                # TODO: Investigate if it's possible to reduce redundant code in these
                # two solvers.
                sum -= elementL * solution.aV[j] - elementM * solution.aPhi[j]
            aResult[i] = sum
        return aResult

